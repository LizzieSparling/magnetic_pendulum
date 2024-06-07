from functools import partial
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from base import CurvilinearCoordinateSystem, Model, Model_Object, get_predefined_system
from operator import itemgetter

class Pendulum(Model):

    def __init__(self, coordsystem, name='Pendulum', *args, **kwargs):

        def new_constants():
            consts1 = list(sp.symbols('m g L b'))
            return consts1

        super().__init__(
            coordsystem,
            name=name,
            *args, **kwargs
        )

        self._add_consts(new_constants())

        self.add_forces(damping = self.damping()
                        # tension = self.tension()
                        )
        self.add_energies(GPE = self.gravity())
    

    def gravity(self):
        m, g = itemgetter('m', 'g')(self.constants)
        coordsys = self.coordsystem
        # dot product position with z axis for height
        h = coordsys.position_vector.dot(coordsys.C.k)
        # energy = m*g*h (ignoring energy minimums)
        return m*g*h

    def damping(self):
        b = itemgetter('b')(self.constants)
        coordsys = self.coordsystem
        # damping force is negatively proportional to velocity
        return -b * coordsys.velocity_vector
    
    # def tension(self):
    #     pos = self.coordsystem.position_vector
    #     sp.simplify(pos.normalize())*sp.symbols('T')

    def kinetic_energy(self):
        m = itemgetter('m')(self.constants)
        coordsys = self.coordsystem
        speed = sp.diff(coordsys.position_vector, coordsys.t).magnitude()
        return self.Energy(m*(speed**2)/2)

    
    def create_bob(self, position_overwrite, forces_include=None, energy_include=None):
        m = itemgetter('m')(self.constants)
        
        bob = Model_Object(
            self, m, position_overwrite,
            forces_include=forces_include,
            energy_include=energy_include,
            name=f'bob_{len(self.objs)}'
        )
        self._add_obj(bob)
        return bob
    
    def _get_ddt(self, bob: Model_Object=None, **value_subs):
        def ddt(_, y_array):
            u1, u2, u3, u1v, u2v, u3v = y_array
            return np.array([
                u1v,
                u2v,
                u3v,
                fs[0](*y_array),
                fs[1](*y_array),
                fs[2](*y_array)
            ])

        if not bob:
            bob = self.create_bob()
        fs = bob.get_acceleration_with_subs(**value_subs)
        return ddt

    

    def get_path(self, starting_conditions, t_max=100, dt=0.001, **value_subs):
        t_vals = np.arange(0, t_max, dt)
        ddt = self._get_ddt(**value_subs)
        result = solve_ivp(ddt, (0, t_max), starting_conditions, t_eval=t_vals, method='RK45')
        return result.y

    def path_to_trajectory(self, path):
        coordsys = self.coordsystem

        trajectory = np.zeros((3, path.shape[1]))
        for i in range(3):
            trajectory[i] = sp.lambdify(
                coordsys.U,
                coordsys.position_vector.dot(coordsys.C.base_vectors()[i]),
                modules='numpy')(
                    path[0], path[1], path[2]
                )
        return trajectory

class MagneticPendulum(Pendulum):

    def __init__(self, num_magnets, coordsystem, name='MagneticPendulum', *args, **kwargs):

        def new_constants():
            consts1 = list(sp.symbols('R h'))
            self.polarity_list = list(sp.symbols(f'p_0:{self.num_magnets}'))
            return consts1 + self.polarity_list

        super().__init__(
            coordsystem,
            name=name,
            *args, **kwargs
        )

        self.num_magnets = num_magnets
        self._add_consts(new_constants())
        self.magnet_positions_C = self.get_evenly_spaced_magnets()
        
        self.add_energies(Vmag1 = self.magnetic_1_potential_energy())

    def get_evenly_spaced_magnets(self):
        L, R, h = itemgetter('L', 'R', 'h')(self.constants)
        coordsys = self.coordsystem
        
        φmag_list = np.array([(2*sp.pi/self.num_magnets) * i for i in range(self.num_magnets)])
        position = lambda φmag: (
            (R*sp.cos(φmag))*coordsys.C.i
            + (R*sp.sin(φmag))*coordsys.C.j
            - (L+h)*coordsys.C.k
        )
        return list(np.vectorize(position)(φmag_list))

    def magnetic_1_potential_energy(self):

        def single_magnet(polarity, magnet_position_C):
            # we assume that the following equation holds per magnet, as given in paper:
            # Vmag_i = (-1/3) * p_i/dist(bob, magnet_i)**3
            return -sp.Rational(1,3) * polarity/(coordsys.get_distance_to_cartesian(magnet_position_C))**3

        coordsys = self.coordsystem
        # sum up the contribution over all magnets
        return sum(
            single_magnet(polarity, magnet_position_C)
            for polarity, magnet_position_C in zip(self.polarity_list, self.magnet_positions_C)
        )


class MagneticPendulumInvSpherical(MagneticPendulum):
    def __init__(self, num_magnets, name='MagneticPendulumInvSpherical', *args, **kwargs):
        super().__init__(
            num_magnets,
            coordsystem=get_predefined_system('InvSpherical'),
            name=name,
            *args, **kwargs
        )

    def create_bob(self):
        L = itemgetter('L')(self.constants)
        pos = (L, None, None)
        return super().create_bob(position_overwrite=pos)

    def _get_ddt(self, bob: Model_Object=None, **value_subs):
        """this implementation is so hideous but I'm too tired to think about it"""
        def ddt(_, y_array):
            _, θ, ϕ, _, θv, ϕv = y_array
            _ = None
            return np.array([
                0,
                θv,
                ϕv,
                0,
                fs[1](_, θ, ϕ, _, θv, ϕv),
                fs[2](_, θ, ϕ, _, θv, ϕv)
            ])

        if not bob:
            bob = self.create_bob()
        fs = bob.get_acceleration_with_subs(**value_subs)
        return ddt    


class MagneticPendulumXY(MagneticPendulum):
    def __init__(self, num_magnets, name='MagneticPendulumXY', *args, **kwargs):
        super().__init__(
            num_magnets,
            coordsystem=get_predefined_system('Cartesian'),
            name=name,
            *args, **kwargs
        )

    def gravity(self):
        """we define gravity slightly different on the plane ..."""
        return (self.coordsystem.U[0]**2 + self.coordsystem.U[1]**2)/2

    def create_bob(self):
        L = itemgetter('L')(self.constants)
        pos = (None, None, -L)
        return super().create_bob(position_overwrite=pos)
    
    def _get_ddt(self, bob: Model_Object=None, **value_subs):
        """this implementation is so hideous but I'm too tired to think about it"""
        def ddt(_, y_array):
            x, y, _, xv, yv, _ = y_array
            _ = None
            return np.array([
                xv,
                yv,
                0,
                fs[0](x, y, _, xv, yv, _),
                fs[1](x, y, _, xv, yv, _),
                0
            ])

        if not bob:
            bob = self.create_bob()
        fs = bob.get_acceleration_with_subs(**value_subs)
        return ddt