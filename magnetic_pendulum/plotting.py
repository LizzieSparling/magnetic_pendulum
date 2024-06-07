import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_trajectory(trajectory, magnet_positions, L):
    
    ax = plt.figure(figsize=(8, 8)).add_subplot(projection='3d')
    # plt.title(f'Magnetic Pendulum Trajectory')
    ax.set_title(f'Magnetic Pendulum Trajectory')
    # Plot the trajectory
    ax.plot(*trajectory, label='Pendulum Path', zorder=1)  

    # Plot the magnet positions
    ax.scatter(*zip(*magnet_positions), color='red', label='Magnets', zorder=2)  

    # Plot the final & initial positions of the pendulum
    final_position = trajectory[:,-1]
    initial_position = trajectory[:,0]
    ax.scatter(*final_position, color='cyan', label='Final Position', s=10, zorder=3)
    ax.scatter(*initial_position, color='black', label='Initial Position', s=10, zorder=3)

    ax.legend()
    ax.grid(True)
    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    ax.set_zlim(-L,L)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()
    return ax

def plot_trajectory_animation(trajectory, magnet_positions, L, dt=0.01, timestep=None):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    # plt.title(f'Magnetic Pendulum Trajectory')
    ax.set_title(f'Magnetic Pendulum Trajectory')
    # Plot the trajectory
    ax.plot(*trajectory, label='Pendulum Path', color='gray', alpha=0.1, zorder=1)  

    # Plot the magnet positions
    ax.scatter(*zip(*magnet_positions), color='red', label='Magnets', zorder=2)  

    # Plot the final & initial positions of the pendulum
    final_position = trajectory[:,-1]
    initial_position = trajectory[:, 0]
    ax.scatter(*final_position, color='cyan', label='Final Position', s=10, zorder=3)
    ax.scatter(*initial_position, color='black', label='Initial Position', s=10, zorder=3)

    ax.legend()
    ax.grid(True)
    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    ax.set_zlim(-L,L)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    line, = ax.plot([], [], [], 'o-', lw=2)
    trace, = ax.plot([], [], [], '.-', lw=1, ms=2)
    time_template = 'time = %.5f'
    time_text = ax.text(0.05, 0.9, 0.9, '', transform=ax.transAxes)

    animation_trajectory = trajectory.T
    if timestep:
        animation_trajectory = animation_trajectory[::timestep].copy()
    

    def animate(i):
        i = i+100
        this = animation_trajectory[i:i+1]
        history = animation_trajectory[:i]

        line.set_data_3d(this.T)
        trace.set_data_3d(history.T)
        time_text.set_text(time_template % (i*dt*timestep))
        return line, trace, time_text

    N = len(animation_trajectory)
    ani = animation.FuncAnimation(
        fig, animate, N, interval=dt/3, blit=True, cache_frame_data=False)
#     plt.show()

    return ani