import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- 1. Define the Fitness Function (N-Dimensional Rastrigin) ---
def fitness_function(position):
    return np.sum(position**2 - 10 * np.cos(2 * np.pi * position) + 10)

# --- Main GSA Algorithm ---
def run_gsa(show_logs=True, seed=None):
    # --- Parameters tuned for High Dimensions ---
    N = 100
    D = 20
    max_iter = 1000
    G0 = 200
    alpha = 10
    epsilon = 1e-7
    lower_bound = -5.12
    upper_bound = 5.12
    kbest_initial = N
    kbest_final = 1
    
    if seed is not None:
        np.random.seed(seed)
        
    positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(N, D)
    velocities = np.zeros((N, D))
    best_agent_position = np.zeros(D)
    best_agent_fitness = float('inf')

    positions_history = []
    best_fitness_history = []

    if show_logs:
        print(f"Starting GSA for Rastrigin Function (D={D})...")
        print("-" * 80)

    for t in range(max_iter):
        positions_history.append(positions.copy())
        fitness_values = np.array([fitness_function(p) for p in positions])
        
        current_best_fitness = np.min(fitness_values)
        if current_best_fitness < best_agent_fitness:
            best_agent_fitness = current_best_fitness
            best_agent_position = positions[np.argmin(fitness_values)].copy()

        best_fitness_history.append(best_agent_fitness)
        
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
        
        if show_logs and (t + 1) % 100 == 0:
            print(f"Iteration {t+1:4d}: Best Fitness = {best_agent_fitness:.8f}")

    if show_logs:
        print("-" * 80)
        print("GSA has finished.")
        print(f"The best fitness found is: {best_agent_fitness:.8f}")
        print("Known minimum is f(0,..,0) = 0.0")
        print("-" * 80)

    return best_agent_fitness, D, positions_history, best_fitness_history

if __name__ == "__main__":
    fitness, D_val, positions_history, best_fitness_history = run_gsa(show_logs=True, seed=42)
    
    print("Generating visualizations...")
    
    # --- Create output directory inside the rastrigin folder ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "gsa_rastrigin_nd_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved in '{output_dir}/'")

    # --- FIGURE A: Convergence Graph (Works for ANY dimension) ---
    fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
    ax_conv.plot(np.arange(1, len(best_fitness_history) + 1), best_fitness_history)
    ax_conv.set_title(f'Convergence of GSA on Rastrigin Function (D={D_val})')
    ax_conv.set_xlabel('Iteration'); ax_conv.set_ylabel('Best Fitness')
    ax_conv.set_yscale('log'); ax_conv.grid(True)
    plt.tight_layout()
    fig_conv.savefig(os.path.join(output_dir, f'convergence_{D_val}d.png'))
    print(f"Convergence plot for D={D_val} saved successfully.")

    # --- Visualization for 2D case ONLY ---
    if D_val == 2:
        print("Dimension is 2. Generating 2D-specific plots...")
        lower_bound = -5.12; upper_bound = 5.12
        x_plot = np.linspace(lower_bound, upper_bound, 400)
        y_plot = np.linspace(lower_bound, upper_bound, 400)
        X, Y = np.meshgrid(x_plot, y_plot)
        Z = fitness_function(np.array([X, Y]))

        # --- FIGURE D: Static Plot of Final Positions at 100 Iterations ---
        max_iter = len(positions_history)
        fig_final, ax_final = plt.subplots(figsize=(8, 7))
        ax_final.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        ax_final.set_title(f'Final Agent Positions at Iteration {max_iter}')
        ax_final.set_xlabel('x1'); ax_final.set_ylabel('x2')
        final_positions = positions_history[-1]
        ax_final.scatter(final_positions[:, 0], final_positions[:, 1], c='red', s=35, label=f'Agents at Iteration {max_iter}')
        ax_final.plot(0, 0, 'y*', markersize=15, label='Global Minimum')
        ax_final.legend()
        plt.tight_layout()
        fig_final.savefig(os.path.join(output_dir, 'final_positions_2d.png'))
        print("Final positions plot for D=2 saved successfully.")
    else:
        print(f"Dimension is {D_val}. Skipping 2D-specific plots.")

    plt.show()

