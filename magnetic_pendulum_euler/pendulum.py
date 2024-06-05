import numpy as np

class MagneticPendulum:
    def __init__(self, magnets, b, h, initial_pos, initial_vel):
        self.magnets = np.array(magnets, dtype=float)  # Magnet positions in numpy array
        self.b = b  # Damping constant
        self.h = h  # Distance of bob above x-y plane
        self.pos = np.array(initial_pos, dtype=float)  # Initial position (x, y)
        self.vel = np.array(initial_vel, dtype=float)  # Initial velocity (dx, dy)
        self.strength = 5


    def _distance(self, magnet_pos):
        """Magnet distances : first derived equation."""
        return np.sqrt((magnet_pos[0] - self.pos[0])**2 + (magnet_pos[1] - self.pos[1])**2 + self.h**2)

    def _magnetic_force(self, magnet_pos):
        """Magnetic force vector : second derived equation."""
        dist = self._distance(magnet_pos)
        return self.strength * (magnet_pos - self.pos) * (1 / dist**5)

    def _gravitational_force(self):
        """Gravitational force on bob."""
        return -self.pos

    def _damping_force(self):
        """Damping force on bob."""
        return -self.b * self.vel

    def _total_force(self):
        """Total force acting on bob."""
        magnetic_forces = np.sum([self._magnetic_force(m) for m in self.magnets], axis=0)
        return self._gravitational_force() + self._damping_force() + magnetic_forces

    def update(self, dt):
        """At each time interval, record position and velocity of bob using Euler's Method."""
        force = self._total_force()
        acceleration = force
        self.vel += acceleration * dt
        self.pos += self.vel * dt
        return self.pos
