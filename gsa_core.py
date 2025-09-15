import numpy as np
import math

# BUG FIX: Added 'initial_state=None' to the function definition
def run_gsa(objective_func, dim, bounds, max_iter, g0, alpha, n_agents=50, initial_state=None):
    """
    A universal GSA engine that can accept an initial state and return a final state.
    """
    D = dim
    N = n_agents
    G0 = g0
    lower_bound, upper_bound = bounds
    epsilon = 1e-7
    kbest_initial = N
    kbest_final = 1
    
    # --- Initialization ---
    if initial_state is None:
        # Start from a random state if none is provided
        positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(N, D)
        velocities = np.zeros((N, D))
    else:
        # Continue from the previous state
        positions = initial_state['positions']
        velocities = initial_state['velocities']

    best_agent_fitness = float('inf')

    # --- Main Loop ---
    for t in range(max_iter):
        fitness_values = np.array([objective_func(p) for p in positions])
        
        current_best_fitness = np.min(fitness_values)
        if current_best_fitness < best_agent_fitness:
            best_agent_fitness = current_best_fitness

        current_worst_fitness = np.max(fitness_values)
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
        
    final_state = {'positions': positions, 'velocities': velocities}
    
    # We need to re-evaluate the final best fitness after the last move
    final_fitness_values = np.array([objective_func(p) for p in positions])
    final_best_fitness = np.min(final_fitness_values)

    return final_best_fitness, final_state

