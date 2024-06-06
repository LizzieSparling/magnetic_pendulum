import matplotlib.pyplot as plt

def plot_trajectory(trajectory, magnets, initial_pos):
    plt.figure(figsize=(8, 8))
    plt.title(f'Magnetic Pendulum Trajectory')

    # Plot the trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Pendulum Path', zorder=1)  

    # Plot the magnet positions
    plt.scatter(*zip(*magnets), color='red', label='Magnets', zorder=2)

    # Plot trajectory starting point
    plt.scatter(initial_pos[0], initial_pos[1], color='orange', label='Initial Position', zorder=3)

    # Plot the final position of the pendulum
    final_position = trajectory[-1]
    plt.scatter(final_position[0], final_position[1], color='cyan', label='Final Position', zorder=4)

    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()