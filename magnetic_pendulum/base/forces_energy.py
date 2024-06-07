from functools import cached_property, partial
from operator import itemgetter

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
    def __init__(self, coordsystem: CurvilinearCoordinateSystem,
                 constants_list: List[sp.Symbol]=[],
                 force_dict: Dict[str, Force]={},
                 energy_dict: Dict[str, Energy]={},
                 name = "") -> None:
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
        self.constants = {}
        self._add_consts(constants_list)

        self.name = name
        self.objs = []

        self.force_dict = {}
        self.energy_dict = {}
        self.add_forces(**force_dict)
        self.add_energies(**energy_dict)

    def _add_obj(self, object):
        self.objs.append(object)

    def __add_const(self, c: sp.Symbol):
        if self.constants.get(c.name):
            logger.warning('attempting to overwrite %s as constant in model.', c.name)
        self.constants[c.name] = c
    
    def _add_consts(self, constants_list):
        for c in constants_list:
            self.__add_const(c)


    def _add_energy(self, energy: Union[sp.Expr, Energy], name: str):
        if not isinstance(energy, Energy):
            energy = self.Energy(energy)
        
        if name in self.energy_dict:
            logger.warning(
                '%s already exists as energy in %s. overwriting.',
                name, self.name)
        # setattr(self, f'e_{name}', energy)
        self.energy_dict[name] = energy

    def _add_force(self, force: Union[np.ndarray[sp.Expr], Force], name: str):
        if not isinstance(force, Force):
            force = self.Force(force)
        
        if name in self.force_dict:
            logger.warning(
                '%s already exists as force in %s. overwriting.',
                name, self.name)
        # setattr(self, f'f_{name}', force)
        self.force_dict[name] = force

    def add_forces(self, **force_dict):
        for name, force in force_dict.items():
            self._add_force(force, name)
    
    def add_energies(self, **energy_dict):
        for name, energy in energy_dict.items():
            self._add_energy(energy, name)


class Model_Object:
    def __init__(self, Model: Model, mass: Union[sp.Symbol, Real],
                 position_overwrite: List[Union[sp.Symbol, None]]=[None, None, None],
                 forces_include: List[str] = None,
                 energy_include: List[str] = None,
                 constant_subs: Dict[str, Real] = {},
                 name = "",
                 
        ) -> None:
        self.Model = Model
        self.Model._add_obj(self)
        self.name = name
        self.mass = mass
        if not position_overwrite:
            position_overwrite = self.Model.coordsystem.U
        self.position_overwrite = np.array(position_overwrite)

        self.forces_include = forces_include
        self.energy_include = energy_include

        self.constant_subs = constant_subs

        coordsys = self.Model.coordsystem
        self.position = [new_u if new_u else u for (u, new_u) in zip(coordsys.U, self.position_overwrite)]
        self.velocity = [sp.Derivative(pos, coordsys.t) for pos in self.position]


    def overall_force(self) -> Force:
        """includes energies also"""

        if self.forces_include is None:
            forces = self.Model.force_dict.values()
        elif self.forces_include:
            forces = itemgetter(self.forces_include)(self.Model.force_dict)
        else:
            forces = []
        #### these NEEED to be combined - this is so ugly
        if self.energy_include is None:
            energies = self.Model.energy_dict.values()
        elif self.energy_include:
            energies = itemgetter(self.energy_include)(self.Model.energy_dict)
        else:
            energies = []

        total_base_forces = sum((f for f in forces),
                                start=self.Model.Force(np.array([0,0,0]))
                            )
        # total_force_from_energy = sum(e for e in self.energy_dict.values()).to_force()
        total_energy = sum((e for e in energies),
                                start=self.Model.Energy(0)
                            )
        total_force_from_energy = total_energy.to_force()

        return total_base_forces + total_force_from_energy
    
    @cached_property
    def position_func_substitutions(self):
        # subs = []
        # for u, new_u in zip(self.Model.coordsystem.U, self.position_overwrite):
        #     if new_u:
        #         # sub in new position
        #         subs.append((u, new_u))
        #         # sub in new velocity
        #         new_u_diff = sp.Derivative(new_u, self.Model.coordsystem.t)
        #         subs.append((new_u_diff, new_u_diff.doit()))

        # return subs
        t = self.Model.coordsystem.t
        position_subs = list(zip(self.Model.coordsystem.U, self.position))
        velocity_subs = [(v, v.doit()) for v in self.velocity]
        return position_subs + velocity_subs


    def acceleration_equations(self) -> List[sp.Expr]:
        placeholder_force_over_mass = np.array(sp.symbols('F_1 F_2 F_3'))

        eqn =  self.Model.coordsystem.acceleration_components(
            placeholder_force_over_mass
        )

        subs = self.position_func_substitutions
    
        accelerations = [None, None, None]
        for i, (e, f, placeholder) in enumerate(zip(
            eqn, self.overall_force().F, placeholder_force_over_mass
        )):
            # acceleration_in_direction = sp.simplify(e.subs(subs))
            acceleration_in_direction = e.subs(subs)
            overall_force_over_mass = (f/self.mass).subs(subs)

            acceleration_in_direction = acceleration_in_direction.subs(placeholder, overall_force_over_mass)

            free_vars = (set(acceleration_in_direction.free_symbols)
                         - set(self.Model.constants.values())
                         - {self.Model.coordsystem.t}
                        )
            if free_vars:
                logger.warning(f'found free variables in acceleration wrt {self.Model.coordsystem.U[i]}:'\
                               f'{free_vars}. assuming acceleration in this direction is 0')
                acceleration_in_direction = sp.S.Zero

            accelerations[i] = acceleration_in_direction

        return accelerations

    def check_const_substitutions(self, subs):
        consts = set(self.Model.constants)

        left_over = set(subs) - consts
        missing = consts - set(subs)
        if left_over:
            logger.info(f'constants specified but unused: {left_over}')
        if missing:
            logger.warning(f'some constants still unspecified: {missing}')

    def get_acceleration_with_subs(self, **value_subs):
        # coordsys = self.Model.coordsystem

        overlap = set(value_subs).intersection(set(self.constant_subs))
        if overlap:
            logger.warning(f'overwriting already defined constant values: {overlap}')
            print(self.constant_subs)
        value_subs = self.constant_subs | value_subs
        
        self.check_const_substitutions(value_subs)

        fs = []
        for e in self.acceleration_equations():
            expr = e.subs(value_subs.items())
            inputs = self.position + self.velocity
            fs.append(sp.lambdify(inputs, expr, 'numpy', dummify=True))

        return fs