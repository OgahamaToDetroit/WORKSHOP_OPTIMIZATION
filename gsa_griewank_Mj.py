import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib.patches as mpatches

# Griewank function (f4)
def griewank(x):
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return 1 + sum_part - prod_part

# Gravitational Search Algorithm class
class GSA:
    def __init__(self, objective_func, bounds, num_agents=20, max_iter=100, G0=100, alpha=20):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.G0 = G0  # initial gravitational constant
        self.alpha = alpha  # decay parameter
        self.dim = len(bounds)
        
        # Initialize agents
        self.agents = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                       (self.num_agents, self.dim))
        
        # Initialize velocities
        self.velocities = np.zeros((self.num_agents, self.dim))
        
        # Evaluate initial fitness
        self.fitness = np.array([self.objective_func(agent) for agent in self.agents])
        
        # Find best and worst fitness
        self.best_fitness = np.min(self.fitness)
        self.worst_fitness = np.max(self.fitness)
        self.best_agent = self.agents[np.argmin(self.fitness)].copy()
        
        # History for visualization
        self.history = {
            'positions': [],
            'best_fitness': []
        }
        
        # Save initial state
        self.history['positions'].append(self.agents.copy())
        self.history['best_fitness'].append(self.best_fitness)
    
    def optimize(self):
        for iter in range(self.max_iter):
            # Update gravitational constant
            G = self.G0 * np.exp(-self.alpha * iter / self.max_iter)
            
            # Calculate masses
            if self.best_fitness == self.worst_fitness:
                # All agents have the same fitness
                masses = np.ones(self.num_agents)
            else:
                # Normalize fitness
                normalized_fitness = (self.fitness - self.worst_fitness) / (self.best_fitness - self.worst_fitness)
                masses = normalized_fitness / np.sum(normalized_fitness)
            
            # Calculate forces
            forces = np.zeros((self.num_agents, self.dim))
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:
                        # Euclidean distance between agents i and j (with small epsilon to avoid division by zero)
                        R = np.linalg.norm(self.agents[i] - self.agents[j]) + 1e-10
                        # Calculate force
                        force = G * masses[i] * masses[j] * (self.agents[j] - self.agents[i]) / R
                        forces[i] += force
            
            # Update velocities and positions
            acceleration = forces / (masses.reshape(-1, 1) + 1e-10)  # Add small value to avoid division by zero
            self.velocities = np.random.rand(self.num_agents, self.dim) * self.velocities + acceleration
            self.agents += self.velocities
            
            # Apply bounds
            for i in range(self.dim):
                self.agents[:, i] = np.clip(self.agents[:, i], 
                                           self.bounds[i, 0], self.bounds[i, 1])
            
            # Evaluate fitness
            self.fitness = np.array([self.objective_func(agent) for agent in self.agents])
            
            # Update best and worst fitness
            self.best_fitness = np.min(self.fitness)
            self.worst_fitness = np.max(self.fitness)
            self.best_agent = self.agents[np.argmin(self.fitness)].copy()
            
            # Save history for visualization
            self.history['positions'].append(self.agents.copy())
            self.history['best_fitness'].append(self.best_fitness)
            
        return self.best_agent, self.best_fitness

# Set up the problem
D = 2  # Using 2D for visualization
bounds = [[-600, 600]] * D
num_agents = 20
max_iter = 100

# Run GSA
gsa = GSA(griewank, bounds, num_agents, max_iter)
best_position, best_score = gsa.optimize()

print(f"Best position: {best_position}")
print(f"Best score: {best_score}")

# Create contour plot
def create_contour_plot():
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = griewank(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, label='Function Value')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of Griewank Function')
    plt.show()

create_contour_plot()

# Create animation of agent movement
def create_animation():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create contour background
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = griewank(np.array([X[i, j], Y[i, j]]))

    contour = ax.contour(X, Y, Z, 20, cmap='viridis', alpha=0.5)
    plt.colorbar(contour, ax=ax, label='Function Value')

    # Initialize scatter plot for agents
    scatter = ax.scatter([], [], c='red', s=50, alpha=0.7, label='Agents')
    best_agent_point = ax.scatter([], [], c='blue', s=100, marker='*', label='Best Agent')

    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Agent Movement in Griewank Function (GSA)')
    ax.legend()

    # Animation update function
    def update(frame):
        if frame < len(gsa.history['positions']):
            positions = gsa.history['positions'][frame]
            scatter.set_offsets(positions)
            
            # Find best agent at this iteration
            fitness_values = [griewank(pos) for pos in positions]
            current_best_agent = positions[np.argmin(fitness_values)] # This line was causing the error
            best_agent_point.set_offsets([current_best_agent])

            ax.set_title(f'Agent Movement in Griewank Function (Iteration {frame+1})')

        return scatter, best_agent_point

    ani = FuncAnimation(fig, update, frames=min(20, max_iter), interval=500, blit=True, repeat=False)
    plt.show()
    
    return ani

# Create and display the animation
ani = create_animation()

# Create convergence plot
def create_convergence_plot():
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(gsa.history['best_fitness'])+1), gsa.history['best_fitness'])
    plt.xlabel('Iteration')
    plt.ylabel('Best Function Value')
    plt.title('Convergence of GSA on Griewank Function')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
create_convergence_plot()

# For 3D surface plot
def create_3d_surface():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(bounds[0][0], bounds[0][1], 50) # Reduced resolution for faster plotting
    y = np.linspace(bounds[1][0], bounds[1][1], 50) # Reduced resolution for faster plotting
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = griewank(np.array([X[i, j], Y[i, j]]))

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8) # Changed alpha to 0.8
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.set_title('3D Surface Plot of Griewank Function')
    
    plt.show() # This line was missing

create_3d_surface()

# Run multiple times for statistical analysis
def run_multiple_times(n_runs=10):
    results = []
    for i in range(n_runs):
        gsa = GSA(griewank, bounds, num_agents, max_iter)
        best_position, best_score = gsa.optimize()
        results.append(best_score)
        print(f"Run {i+1}: Best score = {best_score}")

    print(f"\nAverage best score: {np.mean(results):.6f}")
    print(f"Standard deviation: {np.std(results):.6f}")
    
    return results

# Run multiple times for statistical analysis
print("Running GSA multiple times for statistical analysis:")
results = run_multiple_times(10)

# Create a combined visualization
def create_combined_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Contour plot with final agent positions
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = griewank(np.array([X[i, j], Y[i, j]]))

    contour = ax1.contour(X, Y, Z, 20, cmap='viridis', alpha=0.5)
    final_positions = gsa.history['positions'][-1]
    ax1.scatter(final_positions[:, 0], final_positions[:, 1], c='red', s=50, alpha=0.7)
    ax1.scatter(best_position[0], best_position[1], c='blue', s=100, marker='*')
    ax1.set_xlim(bounds[0][0], bounds[0][1])
    ax1.set_ylim(bounds[1][0], bounds[1][1])
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Final Agent Positions')
    
    # Convergence plot
    ax2.plot(range(1, len(gsa.history['best_fitness'])+1), gsa.history['best_fitness'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Function Value')
    ax2.set_title('Convergence of GSA on Griewank Function')
    ax2.set_yscale('log')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

create_combined_visualization()