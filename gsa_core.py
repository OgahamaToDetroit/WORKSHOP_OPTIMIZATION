import numpy as np
import math

def run_gsa(objective_func, dim, bounds, max_iter, g0, alpha, n_agents=50, show_logs=False, seed=None):
    """
    A universal GSA engine that can optimize any given function.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # --- Parameters ---
    D = dim
    N = n_agents
    G0 = g0
    lower_bound, upper_bound = bounds
    epsilon = 1e-7
    kbest_initial = N
    kbest_final = 1
    
    # --- Initialization ---
    positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(N, D)
    velocities = np.zeros((N, D))
    best_agent_position = np.zeros(D)
    best_agent_fitness = float('inf')

    if show_logs:
        print(f"Starting GSA (D={D}, max_iter={max_iter})...")
        print("-" * 80)

    # --- Main Loop ---
    for t in range(max_iter):
        fitness_values = np.array([objective_func(p) for p in positions])
        
        current_best_fitness = np.min(fitness_values)
        current_worst_fitness = np.max(fitness_values)
        
        if current_best_fitness < best_agent_fitness:
            best_agent_fitness = current_best_fitness
            best_agent_position = positions[np.argmin(fitness_values)].copy()

        m_normalized = (fitness_values - current_worst_fitness) / (current_best_fitness - current_worst_fitness + epsilon)
        masses = m_normalized / (np.sum(m_normalized) + epsilon)

        G = G0 * math.exp(-alpha * t / max_iter)
        kbest = round(kbest_initial - (kbest_initial - kbest_final) * (t / max_iter))
        sorted_indices = np.argsort(fitness_values)

        forces = np.zeros((N, D))
        for i in range(N):
            total_force_on_i = np.zeros(D)
            for j_idx in sorted_indices[0:kbest]:
                j = j_idx
                if i != j:
                    displacement_vec = positions[j] - positions[i]
                    distance = np.linalg.norm(displacement_vec)
                    force_ij = G * (masses[i] * masses[j] / (distance + epsilon)) * displacement_vec
                    total_force_on_i += np.random.rand() * force_ij 
            forces[i] = total_force_on_i
        
        accelerations = forces / (masses[:, np.newaxis] + epsilon)
        velocities = np.random.rand(N, 1) * velocities + accelerations
        positions = positions + velocities
        positions = np.clip(positions, lower_bound, upper_bound)
        
        if show_logs and (t + 1) % 100 == 0:
            print(f"Iteration {t+1:4d}: Best Fitness = {best_agent_fitness:.8f}")

    if show_logs:
        print("-" * 80)
        print("GSA has finished.")
        print(f"The best fitness found is: {best_agent_fitness:.8f}")

    return best_agent_fitness, best_agent_position
