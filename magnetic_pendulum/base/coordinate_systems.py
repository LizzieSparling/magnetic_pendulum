import sympy as sp
from sympy import sin, cos
from sympy.vector import CoordSys3D, VectorAdd
from sympy.physics.vector import dynamicsymbols

import numpy as np

from functools import cached_property
from typing import List


class CurvilinearCoordinateSystem:
    def __init__(self, symbols: tuple[sp.Symbol],
                 cartesian_position_formula: sp.Expr,
                 C: CoordSys3D=CoordSys3D('C'), name: str="") -> None:
        """
        t: sp.Symbol which defines the dynamics of our system
        symbols: 3-tuple of sp.Symbol which we use to define position (these will be converted into functions of t)
        position_vector: 3-tuple, giving the cartesian coordinates of a in the system, in terms of `symbols`
        (optional) C: standard cartesian coordinate system we define this in terms of
        """
        self.name = name
        self.C = C  # standard Cartesian system
        self.t = dynamicsymbols._t
        self.old_symbols = np.array(symbols)

        # tuple of symbols as functions of t
        self.U = np.array(dynamicsymbols(
            ' '.join(s.name for s in self.old_symbols), positive=True, real=True
        ))

        position_vector: VectorAdd = (
            cartesian_position_formula[0]*C.i
            + cartesian_position_formula[1]*C.j
            + cartesian_position_formula[2]*C.k
        )
        self.position_vector = position_vector.subs(zip(self.old_symbols, self.U))

    @cached_property
    def _direction_vectors(self) -> List[VectorAdd]:
        get_vector = lambda u: sp.simplify(sp.diff(self.position_vector, u))
        return np.vectorize(get_vector)(self.U)

    @cached_property
    def lame_coefficients(self) -> List[sp.Expr]:
        magnitude = lambda v: sp.simplify(sp.sqrt(v.dot(v)))
        return np.vectorize(magnitude)(self._direction_vectors)

    @cached_property
    def unit_vectors(self) -> List[VectorAdd]:
        return np.vectorize(sp.simplify)(self._direction_vectors / self.lame_coefficients)
    
    def resolve_to_unit_vectors(self, v: VectorAdd) -> sp.Expr:
        "get component of vector `v` in direction `self.unit_vectors[i]`"   
        wrt_unit = lambda unit: sp.simplify(v.dot(unit))
        return np.vectorize(wrt_unit)(self.unit_vectors)

    @cached_property
    def velocity_vector(self) -> List[sp.Derivative]:
        """(similar to acceleration_vector)"""
        v = sp.diff(self.position_vector, self.t)
        return self.resolve_to_unit_vectors(v)
        
    @cached_property
    def acceleration_vector(self) -> List[sp.Derivative]:
        """
        list giving acceleration in directions of each **unit_vector**
        (note: this distinction becomes clear in spherical coordinates, for example)
        """
        a = sp.diff(self.position_vector, self.t, 2)
        return self.resolve_to_unit_vectors(a)
    
    def acceleration_components(self, RHS) -> sp.Expr:
        """
        gives acceleration of each of `self.U`, given some RHS (should be force/mass)
        RHS: 3-tuple giving the scalar (???) force/mass in the direction of each unit_vector
        """  
        def rearrange_acceleration_scalar(a, rhs, u):
            u_a_solns = sp.solve(sp.Eq(a, rhs), sp.diff(u, self.t, 2))
            if len(u_a_solns) != 1:
                raise ValueError(f'could not find acceleration in direction of {u} in terms of {sp.diff(u, self.t, 2)}.\nreturned solns: {u_a_solns}')
            return sp.simplify(u_a_solns[0])

        return np.vectorize(rearrange_acceleration_scalar)(self.acceleration_vector, RHS, self.U)

    def get_distance(self, to_v, from_v=None):
        """MUST BOTH be in same coord system"""
        to_pos = self.position_vector.subs(zip(self.U, to_v))
        return self.get_distance_to_cartesian(to_pos, from_v)
    
    def get_distance_to_cartesian(self, to_pos: VectorAdd, from_v=None):
        """from_v in current coord system, to_pos in cartesian"""
        if from_v:
            from_pos = self.position_vector.subs(zip(self.U, from_v))
        else:
            from_pos = self.position_vector
        return sp.simplify((to_pos - from_pos).magnitude())



def _get_cartesian_system() -> CurvilinearCoordinateSystem:
    symbols = (x, y, z) = sp.symbols('x y z')
    position_vector = (x, y, z)

    return CurvilinearCoordinateSystem(
        symbols=symbols,
        cartesian_position_formula=position_vector,
        name='Cartesian'
    )

def _get_cylindrical_system() -> CurvilinearCoordinateSystem:
    symbols = (r, θ, z) = sp.symbols('r θ z')
    position_vector = ((r*cos(θ)), (r*sin(θ)), (z))

    return CurvilinearCoordinateSystem(
        symbols=symbols,
        cartesian_position_formula=position_vector,
        name='Cylindrical'
    )

def _get_spherical_system() -> CurvilinearCoordinateSystem:
    symbols = (r, θ, φ) = sp.symbols('r θ φ')
    position_vector = ((r*sin(θ)*cos(φ)), (r*sin(θ)*sin(φ)), (r*cos(θ)))

    return CurvilinearCoordinateSystem(
        symbols=symbols,
        cartesian_position_formula=position_vector,
        name='Spherical'
    )

def _get_inverted_spherical_system() -> CurvilinearCoordinateSystem:
    """standard spherical, but θ corresponds to *negative* z-axis"""
    symbols = (r, θ, φ) = sp.symbols('r θ φ')
    position_vector = ((r*sin(θ)*cos(φ)), (r*sin(θ)*sin(φ)), (-r*cos(θ)))

    return CurvilinearCoordinateSystem(
        symbols=symbols,
        cartesian_position_formula=position_vector,
        name='InvSpherical'
    )


def get_predefined_system(name) -> CurvilinearCoordinateSystem:
    """[not well implemented, but does the job]
    get a coordinate system from predefined set
    """
    if name == 'Cartesian':
        return _get_cartesian_system()
    elif name == 'Cylindrical':
        return _get_cylindrical_system()
    elif name == 'Spherical':
        return _get_spherical_system()
    elif name == 'InvSpherical':
        return _get_inverted_spherical_system()
    else:
        raise ValueError(f"no predefined system with name '{name}'")
    