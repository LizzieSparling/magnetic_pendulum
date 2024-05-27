
from magnetic_pendulum import MagneticPendulum, run_simulation, plot_trajectory
import numpy as np

# Define the magnets in a square centered at the origin
R = 0.6  # Square radius
magnets = [(R, R), (-R, R), (-R, -R), (R, -R)]

b = 0.05  # Damping coefficient
h = 0.5  # Height above the x-y plane

initial_pos = [0.95, -0.6]  # Random fixed initial position
initial_vel = (0, 0)  # Initial velocity
print(f"Initial position: {initial_pos}")

# Create the pendulum
pendulum = MagneticPendulum(magnets, b, h, initial_pos, initial_vel)

dt = 0.01  # Time step
max_steps = 10000  # Maximum number of steps
trajectory = run_simulation(pendulum, dt, max_steps)

plot_trajectory(trajectory, magnets)