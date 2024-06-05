import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def euler_method(x0, v0, dt, t_max):
    """Euler's method to solve x'' = -x."""
    num_steps = int(t_max / dt)
    t_values = np.linspace(0, t_max, num_steps)
    x_values = np.zeros(num_steps)
    v_values = np.zeros(num_steps)
    
    x_values[0] = x0
    v_values[0] = v0
    
    for i in range(1, num_steps):
        v_values[i] = v_values[i - 1] - x_values[i - 1] * dt
        x_values[i] = x_values[i - 1] + v_values[i - 1] * dt
    
    return t_values, x_values

def exact_solution(t_values):
    """Exact solution x(t) = cos(t)."""
    return np.cos(t_values)

def rk_method(x0, v0, t_max):
    """Runge Kutta method to solve x'' = -x using solve_ivp."""
    def derivatives(t, y):
        return [y[1], -y[0]]
    
    initial_state = [x0, v0]
    solution = solve_ivp(derivatives, [0, t_max], initial_state, method='RK45', t_eval=np.linspace(0, t_max, int(t_max/0.1)))
    return solution.t, solution.y[0]

def plot_comparison(t_values, euler_x_values, exact_x_values, rk_t_values, rk_x_values):
    """Plot the Euler's method approximation vs. the exact solution vs. the RK solution."""
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, euler_x_values, label="Euler's Method", linestyle='dashed', color='blue')
    plt.plot(t_values, exact_x_values, label="Exact Solution", linestyle='solid', color='black')
    plt.plot(rk_t_values, rk_x_values, label="RK Method", linestyle='dashed', color='red')
    plt.xlabel('Time (t)')
    plt.ylabel('Position (x)')
    plt.title("Comparison of Euler's Method, Exact Solution, and RK Method")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Initial conditions
    x0 = 1
    v0 = 0
    dt = 0.1
    t_max = 10
    
    # Euler's method
    t_values, euler_x_values = euler_method(x0, v0, dt, t_max)
    
    # Exact solution
    exact_x_values = exact_solution(t_values)
    
    # RK method
    rk_t_values, rk_x_values = rk_method(x0, v0, t_max)
    
    # Plot comparison
    plot_comparison(t_values, euler_x_values, exact_x_values, rk_t_values, rk_x_values)

if __name__ == "__main__":
    main()
