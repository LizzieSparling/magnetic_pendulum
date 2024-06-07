from functools import cached_property, partial

from .coordinate_systems import CurvilinearCoordinateSystem
import sympy as sp
import numpy as np

from numbers import Real
from typing import Tuple, Dict, Union, List

import logging
logger = logging.getLogger(__name__)


class Force:
    def __init__(self, vector: np.ndarray[sp.Expr], coordsystem: CurvilinearCoordinateSystem) -> None:
        """
        vector: should be 3-tuple in terms of coordsystem's .unit_vectors and .U (scalars)
        """
        ### probably adds checks in for safety - not the focus right now
        self.F = vector
        self.coordsystem = coordsystem
    
    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, Force):
            raise NotImplementedError(f"must add to type of Force. tried {type(other)}")
        if self.coordsystem != other.coordsystem:
            raise NotImplementedError("must add to Force with same coordinate system")
        return Force(self.F + other.F, self.coordsystem)

    def __radd__(self, other):
        return self.__add__(other)


class Energy:
    def __init__(self, scalar: sp.Expr, coordsystem: CurvilinearCoordinateSystem) -> None:
        """
        scalar: should be a scalar expression in terms of coordsystem's .U (scalars)
        """
        ### probably adds checks in for safety - not the focus right now
        self.E = scalar
        self.coordsystem = coordsystem

    def to_force(self):
        # grad(E) w.r.t the basis U
        vector = -np.vectorize(sp.diff)(self.E, self.coordsystem.U)/self.coordsystem.lame_coefficients
        return Force(vector, self.coordsystem)

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, Energy):
            raise NotImplementedError(f"must add to type of Energy. tried {type(other)}")
        if self.coordsystem != other.coordsystem:
            raise NotImplementedError("must add to Energy with same coordinate system")
        return Energy(self.E + other.E, self.coordsystem)

    def __radd__(self, other):
        return self.__add__(other)
    

class Model:
    def __init__(self, coordsystem: CurvilinearCoordinateSystem, constants: List[sp.Symbol], name = "") -> None:
        """
        constants: dictionary of sympy symbol to either a fixed value or None (representing no value)
                    not completely sure how to use this yet..
        """
        self.coordsystem = coordsystem
        self.Force = partial(Force, coordsystem=self.coordsystem)
        self.Energy = partial(Energy, coordsystem=self.coordsystem)

        # for const_symbol in constants:
        #     setattr(self, f'c_{const_symbol.name}', const_symbol)
        # self.constants = list(constants)
        self.constants = []
        for c in constants:
            self._add_const(c)

        self.name = name
        self.objs = []

    def _add_obj(self, object):
        self.objs.append(object)
    
    def _add_const(self, c: sp.Symbol):
        if c in self.constants:
            logger.info('%s already exists as constant in model.', c.name)
        setattr(self, f'c_{c.name}', c)
        self.constants.append(c)


class Model_Object:
    def __init__(self, Model: Model, mass: Union[sp.Symbol, Real],
                 position_symbols: np.ndarray[sp.Symbol],
                 force_dict: Dict[str, Force]={},
                 energy_dict: Dict[str, Energy]={},
                 name = "",
                 
        ) -> None:
        self.Model = Model
        self.Model._add_obj(self)
        self.name = name
        self.mass = mass
        if not position_symbols:
            position_symbols = self.Model.coordsystem.U
        self.position_symbols = np.array(position_symbols)

        self.force_dict = force_dict
        self.energy_dict = energy_dict
        
    
    def _add_energy(self, energy: Union[sp.Expr, Energy], name: str):
        # energy = self.Model.Energy(scalar)
        # force = energy.to_force()
        # self.add_force(force, name)

        if not isinstance(energy, Energy):
            energy = self.Model.Energy(energy)
        
        if name in self.energy_dict:
            logger.warning(
                '%s already exists as an energy for %s in %s. overwriting.',
                name, self.name, self.Model.name)
        # setattr(self, f'e_{name}', energy)
        self.energy_dict[name] = energy

    def _add_force(self, force: Union[np.ndarray[sp.Expr], Force], name: str):
        if not isinstance(force, Force):
            force = self.Model.Force(force)
        
        if name in self.force_dict:
            logger.warning(
                '%s already exists as force for %s in %s. overwriting.',
                name, self.name, self.Model.name)
        # setattr(self, f'f_{name}', force)
        self.force_dict[name] = force

    def add_forces(self, **forces_dict):
        for name, force in forces_dict.items():
            self._add_force(force, name)
    
    def add_energies(self, **energies_dict):
        for name, energy in energies_dict.items():
            self._add_energy(energy, name)

    def overall_force(self) -> Force:
        """includes energies also"""
        total_base_forces = sum((f for f in self.force_dict.values()),
                                start=self.Model.Force(np.array([0,0,0]))
                            )
        # total_force_from_energy = sum(e for e in self.energy_dict.values()).to_force()
        total_energy = sum((e for e in self.energy_dict.values()),
                                start=self.Model.Energy(0)
                            )
        total_force_from_energy = total_energy.to_force()

        return total_base_forces + total_force_from_energy
    
    @cached_property
    def position_substitutions(self):
        position_replacement = list(zip(self.Model.coordsystem.U, self.position_symbols))
        velocity_replacement = [(sp.Derivative(pos_u, self.Model.coordsystem.t),
                                sp.diff(pos_u, self.Model.coordsystem.t))
                                for pos_u in self.position_symbols]
        return position_replacement + velocity_replacement

    def acceleration_equations(self) -> List[sp.Expr]:
        placeholder_force_over_mass = np.array(sp.symbols('F_1 F_2 F_3'))

        eqn =  self.Model.coordsystem.acceleration_components(
            placeholder_force_over_mass
        )

        subs = self.position_substitutions
    
        accelerations = [None, None, None]
        for i, (e, f, placeholder) in enumerate(zip(
            eqn, self.overall_force().F, placeholder_force_over_mass
        )):
            # acceleration_in_direction = sp.simplify(e.subs(subs))
            acceleration_in_direction = e.subs(subs)
            overall_force_over_mass = (f/self.mass).subs(subs)

            acceleration_in_direction = acceleration_in_direction.subs(placeholder, overall_force_over_mass)

            free_vars = (set(acceleration_in_direction.free_symbols)
                         - set(self.Model.constants)
                         - {self.Model.coordsystem.t}
                        )
            if free_vars:
                logger.warning(f'found free variables in acceleration wrt {self.Model.coordsystem.U[i]}:'\
                               f'{free_vars}. assuming acceleration in this direction is 0')
                acceleration_in_direction = sp.Expr(0)

            accelerations[i] = acceleration_in_direction

        return accelerations