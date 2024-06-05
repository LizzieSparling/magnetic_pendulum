import numpy as np

class MagneticPendulum:
    def __init__(self, magnets, b, h, initial_pos, initial_vel):
        self.magnets = np.array(magnets, dtype=float)  # Magnet positions in numpy array
        self.b = b  # Damping constant
        self.h = h  # Distance of bob above x-y plane
        self.initial_pos = np.array(initial_pos, dtype=float)  # Initial position (x, y)
        self.initial_vel = np.array(initial_vel, dtype=float)  # Initial velocity (dx, dy)
        self.strength = 5

    def _distance(self, magnet_pos, pos):
        """Magnet distances : first derived equation."""
        return np.sqrt((magnet_pos[0] - pos[0])**2 + (magnet_pos[1] - pos[1])**2 + self.h**2)

    def _magnetic_force(self, magnet_pos, pos):
        """Magnetic force vector : second derived equation."""
        dist = self._distance(magnet_pos, pos)
        return self.strength * (magnet_pos - pos) * (1 / dist**5)

    def _gravitational_force(self, pos):
        """Gravitational force on bob."""
        return -pos

    def _damping_force(self, vel):
        """Damping force on bob."""
        return -self.b * vel

    def _total_force(self, t, state):
        """Total force acting on bob."""
        pos = state[:2]
        vel = state[2:]
        magnetic_forces = np.sum([self._magnetic_force(m, pos) for m in self.magnets], axis=0)
        total_force = self._gravitational_force(pos) + self._damping_force(vel) + magnetic_forces
        return np.concatenate((vel, total_force))

    def initial_conditions(self):
        """Return initial conditions for the RK45 solver."""
        return np.concatenate((self.initial_pos, self.initial_vel))
