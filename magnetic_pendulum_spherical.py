import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import g


# PARAMETERS
L = g  # Length of pendulum
b = 0.1  # Damping coefficient
k = 10  # Magnetic strength coefficient
h = 0.5  # Height of the pendulum above the xy-plane
d = 0.5  # Radius of magnets from the origin on the xy-plane
m = 10  # Mass of pendulum bob


# Initial conditions
theta0 = np.pi / 3  # 45 degrees
phi0 = np.pi / 3  # 45 degrees
theta_dot0 = 0.0  # Initially stationary
phi_dot0 = 0.0  # Initially stationary
initial_conditions = [theta0, phi0, theta_dot0, phi_dot0]


dt = 0.01  # Time step
t_max = 100  # Maximum time
t_vals = np.arange(0, t_max, dt)


magnet_positions = [
    (d, 0, -(L + h)),
    (-d, 0, -(L + h)),
    (0, -d, -(L + h)),
    (0, d, -(L + h))
]


def magnet_dist(x_p, y_p, z_p, magnet_pos):
    x_m, y_m, z_m = magnet_pos
    return np.sqrt((x_p - x_m)**2 + (y_p - y_m)**2 + (z_p - z_m)**2)

def dVmag_dtheta_magnet(theta, phi, x_p, y_p, z_p, magnet_pos):
    r_i = magnet_dist(x_p, y_p, z_p, magnet_pos)
    x_m, y_m, z_m = magnet_pos
    dri_d_theta = (L * np.cos(theta) * np.cos(phi) * (x_p - x_m)
                        + L * np.cos(theta) * np.sin(phi) * (y_p - y_m)
                        + L * np.sin(theta) * (z_p - z_m)) / r_i
    dV_dtheta_magnet = 4 * k * dri_d_theta / r_i**5
    return dV_dtheta_magnet

def dVmag_dtheta(theta, phi, x_p, y_p, z_p):
    # Partial derivative of Vmag with respect to theta 
    return sum([dVmag_dtheta_magnet(theta, phi, x_p, y_p, z_p, mag_pos) for mag_pos in magnet_positions])

def eval_dV_dphi_magnet(theta, phi, x_p, y_p, z_p, magnet_pos):
    r_i = magnet_dist(x_p, y_p, z_p, magnet_pos)
    x_m, y_m, z_m = magnet_pos
    d_r_i_d_phi = (-L * np.sin(theta) * np.sin(phi) * (x_p - x_m)
                    + L * np.sin(theta) * np.cos(phi) * (y_p - y_m)) / r_i
    dV_dphi_magnet = 4 * k * d_r_i_d_phi / r_i**5
    return dV_dphi_magnet

def dVmag_dphi(theta, phi, x_p, y_p, z_p):
    # Partial derivative of Vmag with respect to phi
    return sum([eval_dV_dphi_magnet(theta, phi, x_p, y_p, z_p, mag_pos) for mag_pos in magnet_positions])


def derivatives(t, y):
    theta, phi, theta_dot, phi_dot = y
     
    x_p = L * np.sin(theta) * np.cos(phi)
    y_p = L * np.sin(theta) * np.sin(phi)
    z_p = -L * np.cos(theta)

    # Partial derivatives of the magnetic potential energy
    dV_dtheta = dVmag_dtheta(theta, phi, x_p, y_p, z_p)
    dV_dphi = dVmag_dphi(theta, phi, x_p, y_p, z_p)

    # Euler Lagrange equations, isolating second derivatives
    theta_double_dot = (m * L**2 * np.sin(theta) * np.cos(theta) * phi_dot**2
                        - m * g * L * np.sin(theta)
                        - dV_dtheta
                        - b * theta_dot) / (m * L**2)
    phi_double_dot = (-2 * m * L**2 * np.sin(theta) * np.cos(theta) * theta_dot * phi_dot
                      - dV_dphi
                      - b * phi_dot) / (m * L**2 * np.sin(theta)**2)
    return [theta_dot, phi_dot, theta_double_dot, phi_double_dot]


# Numerical solution
solution = solve_ivp(derivatives, [0, t_max], initial_conditions, t_eval=t_vals, method='RK45')
theta = solution.y[0]
phi = solution.y[1]


# Convert spherical coordinates to Cartesian coordinates for plotting
x = L * np.sin(theta) * np.cos(phi)
y = L * np.sin(theta) * np.sin(phi)
z = -L * np.cos(theta)


# Plotting the trajectory
ax = plt.figure(figsize=(8, 8)).add_subplot(projection='3d')
ax.plot(x, y, z, label='Trajectory')
ax.scatter(x[-1], y[-1], z[-1], color='orange', s=10, label='Final Position', zorder=5)
ax.scatter(x[0], y[0], z[0], color='cyan', s=10, label='Initial Position', zorder=5)



# Magnet positions
for magnet_pos in magnet_positions:
    ax.scatter(*magnet_pos, color='red', s=10, label='Magnet' if 'Magnet' not in ax.get_legend_handles_labels()[1] else "", zorder=4)


ax.set_xlim(-L,L)
ax.set_ylim(-L,L)
ax.set_zlim(-L,L)

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

print()