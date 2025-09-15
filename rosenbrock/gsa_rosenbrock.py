import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os # Import the os module to handle file paths

# --- 1. Define the Fitness Function ---
def fitness_function(pos_2d):
    x1, x2 = pos_2d[0], pos_2d[1]
    return 100 * (x1**2 - x2)**2 + (1 - x1)**2

# --- Main GSA Algorithm Function ---
def run_gsa(show_logs=True, seed=None):
    # --- Algorithm Parameters (Paper-Compliant Values for Full Run) ---
    N = 20
    D = 2
    max_iter = 1000
    G0 = 100
    alpha = 20
    epsilon = 1e-7
    lower_bound = -2
    upper_bound = 2
    kbest_initial = N
    kbest_final = 1
    
    # --- Initialization ---
    if seed is not None:
        np.random.seed(seed)
        
    positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(N, D)
    velocities = np.zeros((N, D))
    best_agent_position = np.zeros(D)
    best_agent_fitness = float('inf')

    positions_history = []
    best_fitness_history = []

    if show_logs:
        print(f"Starting GSA... (Searching for the minimum of Rosenbrock function in {max_iter} iterations)")
        print("-" * 80)

    # --- Main Loop ---
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
            pos_str = f"[{best_agent_position[0]:.6f}, {best_agent_position[1]:.6f}]"
            print(f"Iteration {t+1:4d}: Best Solution = {pos_str}, Fitness = {best_agent_fitness:.8f}")

    if show_logs:
        print("-" * 80)
        print("GSA has finished.")
        pos_str = f"[{best_agent_position[0]:.6f}, {best_agent_position[1]:.6f}]"
        print(f"The best solution found is: {pos_str}")
        print(f"The minimum value found is: {best_agent_fitness:.8f}")
        print("Known minimum is at [1.0, 1.0] with fitness = 0.0")
        print("-" * 80)

    return best_agent_fitness, best_agent_position, positions_history, best_fitness_history

if __name__ == "__main__":
    fitness, position, positions_history, best_fitness_history = run_gsa(show_logs=True, seed=42)
    
    print("Generating visualizations...")

    # --- Create output directory ---
    output_dir = "gsa_rosenbrock_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved in '{output_dir}/'")

    # --- Prepare data for contour plots ---
    max_iter_full = 1000
    lower_bound = -2
    upper_bound = 2
    x_plot = np.linspace(lower_bound, upper_bound, 400)
    y_plot = np.linspace(lower_bound, upper_bound, 400)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = fitness_function([X, Y])

    # --- FIGURE A: Convergence Graph ---
    fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
    iterations_to_plot = np.arange(5, max_iter_full + 1)
    fitness_to_plot = best_fitness_history[4:]
    ax_conv.plot(iterations_to_plot, fitness_to_plot)
    ax_conv.set_title(f'Convergence of GSA (Iterations 5-{max_iter_full})')
    ax_conv.set_xlabel('Iteration k'); ax_conv.set_ylabel('Value of the Objective Function (Best Fitness)')
    ax_conv.set_yscale('log'); ax_conv.grid(True)
    plt.tight_layout()
    fig_conv.savefig(os.path.join(output_dir, 'convergence_1000iter.png'))

    # --- FIGURE B: Animation of First 20 Iterations ---
    fig_anim_20, ax_anim_20 = plt.subplots(figsize=(8, 7))
    ax_anim_20.contourf(X, Y, Z, levels=np.logspace(0, 3.5, 35), cmap='viridis', alpha=0.7)
    ax_anim_20.set_title('GSA Agent Movement (First 20 Iterations)')
    ax_anim_20.set_xlabel('x1'); ax_anim_20.set_ylabel('x2')
    ax_anim_20.plot(1, 1, 'r*', markersize=15, label='Global Minimum')
    scatter_20 = ax_anim_20.scatter(positions_history[0][:, 0], positions_history[0][:, 1], c='red', s=25)
    iter_text_20 = ax_anim_20.text(0.02, 0.95, '', transform=ax_anim_20.transAxes, color='white', fontsize=12,
                                   bbox=dict(facecolor='black', alpha=0.5))
    ax_anim_20.legend()
    def update_anim_20(frame):
        scatter_20.set_offsets(positions_history[frame])
        iter_text_20.set_text(f'Iteration: {frame + 1}/20')
        return scatter_20, iter_text_20
    animation_20 = FuncAnimation(fig_anim_20, update_anim_20, frames=20, interval=200, blit=True)
    animation_20.save(os.path.join(output_dir, 'gsa_rosenbrock_anim_20iter.gif'), writer='pillow', fps=5)
    print("Animation of first 20 iterations saved successfully.")

    # --- FIGURE C: Animation of Full 1000 Iterations ---
    fig_anim_1000, ax_anim_1000 = plt.subplots(figsize=(8, 7))
    ax_anim_1000.contourf(X, Y, Z, levels=np.logspace(0, 3.5, 35), cmap='viridis', alpha=0.7)
    ax_anim_1000.set_title(f'GSA Agent Movement (Full {max_iter_full} Iterations)')
    ax_anim_1000.set_xlabel('x1'); ax_anim_1000.set_ylabel('x2')
    ax_anim_1000.plot(1, 1, 'r*', markersize=15, label='Global Minimum')
    scatter_1000 = ax_anim_1000.scatter(positions_history[0][:, 0], positions_history[0][:, 1], c='red', s=25)
    iter_text_1000 = ax_anim_1000.text(0.02, 0.95, '', transform=ax_anim_1000.transAxes, color='white', fontsize=12,
                                       bbox=dict(facecolor='black', alpha=0.5))
    ax_anim_1000.legend()
    def update_anim_1000(frame):
        scatter_1000.set_offsets(positions_history[frame])
        iter_text_1000.set_text(f'Iteration: {frame + 1}/{max_iter_full}')
        return scatter_1000, iter_text_1000
    animation_1000 = FuncAnimation(fig_anim_1000, update_anim_1000, frames=max_iter_full, interval=20, blit=True)
    animation_1000.save(os.path.join(output_dir, 'gsa_rosenbrock_anim_1000iter.gif'), writer='pillow', fps=50)
    print(f"Animation of full {max_iter_full} iterations saved successfully.")

    # --- FIGURE D: Static Plot of Final Positions at 1000 Iterations ---
    fig_final_1000, ax_final_1000 = plt.subplots(figsize=(8, 7))
    ax_final_1000.contourf(X, Y, Z, levels=np.logspace(0, 3.5, 35), cmap='viridis', alpha=0.7)
    ax_final_1000.set_title(f'Final Agent Positions at Iteration {max_iter_full}')
    ax_final_1000.set_xlabel('x1'); ax_final_1000.set_ylabel('x2')
    final_positions_1000 = positions_history[-1]
    ax_final_1000.scatter(final_positions_1000[:, 0], final_positions_1000[:, 1], c='red', s=35, label=f'Agents at Iteration {max_iter_full}')
    ax_final_1000.plot(1, 1, 'y*', markersize=15, label='Global Minimum')
    ax_final_1000.legend()
    plt.tight_layout()
    fig_final_1000.savefig(os.path.join(output_dir, 'final_positions_1000iter.png'))
    
    plt.show()