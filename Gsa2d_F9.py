import numpy as np
import math

# --- 1. Define the Fitness Function ---
# This is the Rastrigin function for 2 dimensions.
# The global minimum is at f(0, 0) = 0.
def fitness_function(pos_2d):
    x = pos_2d[0]
    y = pos_2d[1]
    # Equation: f(x, y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
    term1 = x**2 - 10 * np.cos(2 * np.pi * x)
    term2 = y**2 - 10 * np.cos(2 * np.pi * y)
    return 20 + term1 + term2

# --- 2. Set Algorithm Parameters ---
N = 100                # Number of agents
D = 2                  # Number of dimensions
max_iter = 80        # Maximum number of iterations

# Parameters tuned for the 100-iteration run
G0 = 100               # Initial gravitational constant
alpha = 20             # A constant to control the decay of G
epsilon = 1e-7         # A small constant to avoid division by zero

# Define the search space boundaries (standard for Rastrigin)
lower_bound = -5.12
upper_bound = 5.12

# Kbest strategy parameters
kbest_initial = N
kbest_final = 1

# --- 3. Main GSA Algorithm ---
# --- Step 1: Initialization ---
# Initialize the positions and velocities of N agents in D dimensions.
positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(N, D)
velocities = np.zeros((N, D)) # Initial velocities are zero
masses = np.zeros(N)

# Variables to store the best solution found so far
best_agent_position = np.zeros(D)
best_agent_fitness = float('inf') # Initialize with a very large value

print(f"Starting GSA... (Searching for the minimum of Rastrigin function in 100 iterations)")
print("-" * 80)

# --- Start the main iteration loop ---
for t in range(max_iter):

    # --- Step 2: Fitness Evaluation and Mass Calculation ---
    fitness_values = np.array([fitness_function(p) for p in positions])

    # Find the best and worst fitness values in the current population
    best_fitness = np.min(fitness_values)
    worst_fitness = np.max(fitness_values)
    
    # Update the global best solution if a better one is found
    if best_fitness < best_agent_fitness:
        best_agent_fitness = best_fitness
        best_agent_position = positions[np.argmin(fitness_values)].copy()

    # Calculate the normalized mass (m) for each agent
    m_normalized = (fitness_values - worst_fitness) / (best_fitness - worst_fitness + epsilon)
    # Calculate the final mass (M) for each agent
    masses = m_normalized / (np.sum(m_normalized) + epsilon)

    # Update the gravitational constant G
    G = G0 * math.exp(-alpha * t / max_iter)

    # --- Kbest Strategy Implementation ---
    # Calculate the current value of kbest, decreasing linearly from N to 1
    kbest = round(kbest_initial - (kbest_initial - kbest_final) * (t / max_iter))

    # Sort agents based on their fitness to identify the k-best agents
    sorted_indices = np.argsort(fitness_values)

    # --- Step 3 & 4: Calculate Force and Acceleration ---
    forces = np.zeros((N, D))
    accelerations = np.zeros((N, D))

    # Calculate the total force acting on each agent
    for i in range(N):
        total_force_on_i = np.zeros(D)
        # Only the k-best agents exert force
        for j_idx in sorted_indices[0:kbest]:
            j = j_idx 
            if i != j:
                # Calculate the displacement vector and Euclidean distance
                displacement_vec = positions[j] - positions[i]
                distance = np.linalg.norm(displacement_vec)
                # Calculate the force vector F_ij
                force_ij = G * (masses[i] * masses[j] / (distance + epsilon)) * displacement_vec
                # Sum the forces from different agents with a random factor
                total_force_on_i += np.random.rand() * force_ij
        
        forces[i] = total_force_on_i
        # Calculate the acceleration vector a_i
        accelerations[i] = forces[i] / (masses[i] + epsilon)

    # --- Step 5: Update Velocity ---
    velocities = np.random.rand(N, 1) * velocities + accelerations

    # --- Step 6: Update Position ---
    positions = positions + velocities
    
    # Clip positions to ensure they stay within the defined boundaries
    positions = np.clip(positions, lower_bound, upper_bound)
    
    # Print the results for the current iteration
    pos_str = f"[{best_agent_position[0]:.6f}, {best_agent_position[1]:.6f}]"
    print(f"Iteration {t+1:3d} (kbest={kbest:2d}): Best Solution = {pos_str}, Fitness = {best_agent_fitness:.6f}")

# --- End of Algorithm ---
print("-" * 80)
print("GSA has finished.")
pos_str = f"[{best_agent_position[0]:.6f}, {best_agent_position[1]:.6f}]"
print(f"The best solution found is x, y = {pos_str}")
print(f"The minimum value of the function is f(x,y) = {best_agent_fitness:.6f}")