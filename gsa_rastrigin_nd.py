import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- 1. Define the Fitness Function (N-Dimensional Rastrigin) ---
def fitness_function(position):
    """
    Calculates the Rastrigin function value for a given position vector.
    This function is designed to work with any number of dimensions (D).
    Formula: f(x) = sum_{i=1 to D} [x_i^2 - 10*cos(2*pi*x_i) + 10]
    """
    d = len(position)
    total_sum = 0
    for i in range(d):
        xi = position[i]
        total_sum += (xi**2 - 10 * np.cos(2 * np.pi * xi) + 10)
    return total_sum

# --- 2. Set Algorithm Parameters (Tuned for High Dimensions) ---
N = 100                # << TUNED: Increased agents for better exploration
D = 20
max_iter = 1000

# --- GSA parameters tuned for more exploration ---
G0 = 200               # << TUNED: Higher G0 for stronger initial exploration
alpha = 10             # << TUNED: Slower decay to prolong exploration
epsilon = 1e-7

# Search space boundaries for Rastrigin function
lower_bound = -5.12
upper_bound = 5.12

# Kbest strategy parameters
kbest_initial = N
kbest_final = 1

# --- 3. Main GSA Algorithm ---
def run_gsa(show_logs=True, seed=42):
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
        current_worst_fitness = np.max(fitness_values)
        
        if current_best_fitness < best_agent_fitness:
            best_agent_fitness = current_best_fitness
            best_agent_position = positions[np.argmin(fitness_values)].copy()

        best_fitness_history.append(best_agent_fitness)

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
        print(f"Known minimum is f(0,..,0) = 0.0")
        print("-" * 80)
        
    return best_agent_fitness, best_agent_position, positions_history, best_fitness_history

if __name__ == "__main__":
    fitness, position, positions_history, best_fitness_history = run_gsa()
    
    # --- 4. Visualization Section (ONLY RUNS IF D=2) ---
    if D == 2:
        print("Dimension is 2. Generating visualizations...")

        output_dir = "gsa_rastrigin_2d_output"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output files will be saved in '{output_dir}/'")

        x_plot = np.linspace(lower_bound, upper_bound, 400)
        y_plot = np.linspace(lower_bound, upper_bound, 400)
        X, Y = np.meshgrid(x_plot, y_plot)
        Z = fitness_function([X, Y])

        # --- Convergence Graph ---
        fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
        ax_conv.plot(np.arange(1, max_iter + 1), best_fitness_history)
        ax_conv.set_title('Convergence of GSA on Rastrigin Function (D=2)')
        ax_conv.set_xlabel('Iteration'); ax_conv.set_ylabel('Best Fitness')
        ax_conv.set_yscale('log'); ax_conv.grid(True)
        plt.tight_layout()
        fig_conv.savefig(os.path.join(output_dir, 'convergence_rastrigin.png'))
        
        # --- Animation of Agent Movement ---
        fig_anim, ax_anim = plt.subplots(figsize=(8, 7))
        ax_anim.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        ax_anim.set_title('GSA Agent Movement on Rastrigin Function (D=2)')
        ax_anim.set_xlabel('x1'); ax_anim.set_ylabel('x2')
        ax_anim.plot(0, 0, 'r*', markersize=15, label='Global Minimum')
        scatter = ax_anim.scatter(positions_history[0][:, 0], positions_history[0][:, 1], c='red', s=25)
        iter_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, color='white', fontsize=12,
                                       bbox=dict(facecolor='black', alpha=0.5))
        ax_anim.legend()
        def update_anim(frame):
            scatter.set_offsets(positions_history[frame])
            iter_text.set_text(f'Iteration: {frame + 1}/{max_iter}')
            return scatter, iter_text
        animation = FuncAnimation(fig_anim, update_anim, frames=max_iter, interval=100, blit=True)
        animation.save(os.path.join(output_dir, 'gsa_rastrigin_anim.gif'), writer='pillow', fps=10)
        print("Animation saved successfully.")

        plt.show()
    else:
        print(f"Dimension is {D}. Visualization is skipped (only available for D=2).")
