from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np

#initialize parameters
r = 2 #distance of magnets from origin
h = 0.5
p = [-1,1,-1,1] #polarity of magnets, 1 is attractive
b = 0.05 #damping constant
X = [(r,0),(0,r),(-r,0),(0,-r)] #position vectors of the 4 magnets
def equations(vars):
    #this is the components wise equations that determines the 4 equilibrium points (equation 3 in the article)
    x, y = vars
    eq1=x
    eq2=y
    for i in range(4):
        eq1 -= p[i] * (X[i][0] - x) / ((X[i][0] - x) ** 2 + (X[i][1] - y) ** 2 + h ** 2) ** (5 / 2)
        eq2 -= p[i] * (X[i][1] - y) / ((X[i][0] - x) ** 2 + (X[i][1] - y) ** 2 + h ** 2) ** (5 / 2)
    return [eq1, eq2]

#these 2 lines finds the root of the above equations given an initial guess.
initial_guesses = [[0,-r],[0,r],[-r,0],[r,0],[0,0]]
roots = [root(equations, guess).x for guess in initial_guesses]


def stability_analysis(x_star):
    #this function analyzes the stability of a given point by checking the negativity of eigenvalues.

    J = np.zeros((2, 2)) #This is the jacobian
    for n in range(4):
        #loop through the sum
        pn = p[n]
        x,y=x_star[0],x_star[1]
        (xn, yn)=X[n]
        D = (xn-x)**2+(yn-y)**2+h**2 #|xn-x|^2+h^2 for simplicity
        J[0,0] += -pn*(D**2.5-5*D**1.5*(x-xn)**2/D**5) #partial px partial x
        J[0,1] += -pn*((xn-x)*5*D**1.5*(y-yn)/D**5) #partial px partial y
        J[1, 0] += -pn*((xn-x)*5*D**1.5*(y-yn)/D**5) #partial py partial x
        J[1, 1] += -pn*(D**2.5-5*D**1.5*(y-yn)**2/D**5) #partial py partial y

    I = np.eye(2)
    A = np.block([[np.zeros((2, 2)), I],[J-I, -b * I]]) #the umtimate linearized coefficient matrix
    eigenvalues = np.linalg.eig(A)[0] #find all eigenvalues
    stable = all(np.real(ev) < 0 for ev in eigenvalues) #determine if they are all negative
    return stable


def plot_stability(points):
    #plot all the eq points with colors distinguishing their stability
    stable_points = []
    unstable_points = []

    for point in points:
        if stability_analysis(point):
            stable_points.append(point)
        else:
            unstable_points.append(point)

    stable_points = np.array(stable_points)
    unstable_points = np.array(unstable_points)

    plt.figure(figsize=(10, 6))
    if len(stable_points) > 0:
        plt.scatter(stable_points[:, 0], stable_points[:, 1], color='g', label='Stable Points')
    if len(unstable_points) > 0:
        plt.scatter(unstable_points[:, 0], unstable_points[:, 1], color='r', label='Unstable Points')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Stability of Equilibrium Points')
    plt.grid(True)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()

plot_stability(roots)