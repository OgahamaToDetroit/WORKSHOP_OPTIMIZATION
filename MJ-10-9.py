import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# 1. Functions
# -----------------------------
def f2(x):  # Rosenbrock function
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def f4(x):  # Griewank function
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod([np.cos(x[i] / np.sqrt(i+1)) for i in range(len(x))])
    return 1 + sum_term - prod_term

# -----------------------------
# 2. Simple Swarm Optimization
# -----------------------------
def swarm_optimize(func, dim, bounds, n_agents=20, n_iter=20):
    agents = np.random.uniform(bounds[0], bounds[1], (n_agents, dim))
    best_positions = np.copy(agents)
    best_scores = np.array([func(a) for a in agents])

    global_best_idx = np.argmin(best_scores)
    global_best = agents[global_best_idx]

    history = [agents.copy()]

    for t in range(1, n_iter+1):
        for i in range(n_agents):
            # simple random move (not full PSO for simplicity)
            step = np.random.uniform(-0.1, 0.1, dim)
            candidate = agents[i] + step
            candidate = np.clip(candidate, bounds[0], bounds[1])
            score = func(candidate)

            if score < best_scores[i]:
                best_scores[i] = score
                best_positions[i] = candidate

        global_best_idx = np.argmin(best_scores)
        global_best = best_positions[global_best_idx]
        agents = best_positions + 0.5*(np.random.rand(n_agents, dim)-0.5)

        history.append(agents.copy())

    return global_best, history

# -----------------------------
# 3. Rosenbrock contour with agent positions (3.a)
# -----------------------------
def rosenbrock_contour():
    # run optimization in 2D
    best, history = swarm_optimize(f2, dim=2, bounds=[-2, 2], n_agents=20, n_iter=20)

    # make contour
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = 100 * (Y - X**2)**2 + (1 - X)**2

    fig, ax = plt.subplots(figsize=(6,6))
    cs = ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap="jet")
    ax.clabel(cs, inline=1, fontsize=8)
    scat = ax.scatter([], [], c="red", s=30)

    def update(frame):
        ax.clear()
        cs = ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap="jet")
        ax.set_title(f"Iteration {frame}")
        agents = history[frame]
        ax.scatter(agents[:,0], agents[:,1], c="red", s=30)
        return scat,

    ani = FuncAnimation(fig, update, frames=len(history), interval=500, repeat=False)
    plt.show()

# -----------------------------
# 4. Example usage
# -----------------------------
if __name__ == "__main__":
    # Test f4 (Griewank) with 30 dimensions
    x_test = np.zeros(30)
    print("f4(0) =", f4(x_test))  # should be 0

    # Test f2 (Rosenbrock) with [1,1]
    print("f2([1,1]) =", f2(np.array([1,1])))  # should be 0

    # Plot Rosenbrock contour with agents for 20 iterations
    rosenbrock_contour()