from pendulumRK import MagneticPendulum
from simulationRK import run_simulation
from plotRK import plot_trajectory
import numpy as np

# Define the magnets in a square centered at the origin
R = 0.4  # Square radius
magnets = [(R, R), (-R, R), (-R, -R), (R, -R)]

b = 0.05  # Damping coefficient
h = 0.5  # Height above the x-y plane

initial_pos = [0.9, -0.6]  # Random fixed initial position
initial_vel = (0, 0)  # Initial velocity
print(f"Initial position: {initial_pos}")

# Create the pendulum
pendulum = MagneticPendulum(magnets, b, h, initial_pos, initial_vel)

dt = 0.01  # Time step

trajectory = run_simulation(pendulum, dt)

plot_trajectory(trajectory, magnets)

