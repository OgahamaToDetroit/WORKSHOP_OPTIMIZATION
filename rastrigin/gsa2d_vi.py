import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- 1. Define the Fitness Function ---
# Rastrigin function for 2 dimensions.
def fitness_function(pos_2d):
    x = pos_2d[0]
    y = pos_2d[1]
    term1 = x**2 - 10 * np.cos(2 * np.pi * x)
    term2 = y**2 - 10 * np.cos(2 * np.pi * y)
    return 20 + term1 + term2

# --- 2. Set Algorithm Parameters ---
N = 100
D = 2
max_iter = 100          # Set  iterations as 

# --- Adjusted Parameters for stronger attraction ---
G0 = 500               # Increased G0 significantly for stronger initial pull
alpha = 15             # Slightly reduced alpha to make G decay slower
epsilon = 1e-7

lower_bound = -5.12
upper_bound = 5.12

kbest_initial = N
kbest_final = 1        # Keep kbest_final at 1 to focus on the best agent at the end

# --- 3. Main GSA Algorithm ---
# --- Step 1: Initialization ---
positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(N, D)
velocities = np.zeros((N, D))
masses = np.zeros(N)

best_agent_position = np.zeros(D)
best_agent_fitness = float('inf')

# VISUALIZATION: Store the history of agent positions for plotting
positions_history = []

print(f"Starting GSA... (Searching for the minimum of Rastrigin function in {max_iter} iterations)")
print("-" * 80)

# --- Start the main iteration loop ---
for t in range(max_iter):
    # Store the current positions for visualization
    positions_history.append(positions.copy())

    # --- Step 2: Fitness Evaluation and Mass Calculation ---
    fitness_values = np.array([fitness_function(p) for p in positions])
    best_fitness = np.min(fitness_values)
    worst_fitness = np.max(fitness_values)
    
    if best_fitness < best_agent_fitness:
        best_agent_fitness = best_fitness
        best_agent_position = positions[np.argmin(fitness_values)].copy()

    m_normalized = (fitness_values - worst_fitness) / (best_fitness - worst_fitness + epsilon)
    masses = m_normalized / (np.sum(m_normalized) + epsilon)

    G = G0 * math.exp(-alpha * t / max_iter)

    kbest = round(kbest_initial - (kbest_initial - kbest_final) * (t / max_iter))
    sorted_indices = np.argsort(fitness_values)

    # --- Step 3 & 4: Calculate Force and Acceleration ---
    forces = np.zeros((N, D))
    accelerations = np.zeros((N, D))

    for i in range(N):
        total_force_on_i = np.zeros(D)
        for j_idx in sorted_indices[0:kbest]:
            j = j_idx
            if i != j:
                displacement_vec = positions[j] - positions[i]
                distance = np.linalg.norm(displacement_vec)
                # Ensure random_rand() is called once per force component if needed,
                # but for total_force_on_i, it's typically applied to the whole vector.
                force_ij = G * (masses[i] * masses[j] / (distance + epsilon)) * displacement_vec
                total_force_on_i += np.random.rand() * force_ij # Apply random factor here
        
        forces[i] = total_force_on_i
        accelerations[i] = forces[i] / (masses[i] + epsilon)

    # --- Step 5 & 6: Update Velocity and Position ---
    # Apply a random factor to current velocity components for more dynamism
    velocities = np.random.rand(N, D) * velocities + accelerations 
    positions = positions + velocities
    positions = np.clip(positions, lower_bound, upper_bound)
    
    pos_str = f"[{best_agent_position[0]:.6f}, {best_agent_position[1]:.6f}]"
    print(f"Iteration {t+1:3d} (kbest={kbest:2d}): Best Solution = {pos_str}, Fitness = {best_agent_fitness:.6f}")

# --- End of Algorithm ---
print("-" * 80)
print("GSA has finished.")
pos_str = f"[{best_agent_position[0]:.6f}, {best_agent_position[1]:.6f}]"
print(f"The best solution found is x, y = {pos_str}")
print(f"The minimum value of the function is f(x,y) = {best_agent_fitness:.6f}")
print("-" * 80)
print("Generating visualizations...")

# --- 4. Visualization Section ---

# --- Prepare data for the landscape plots ---
x_plot = np.linspace(lower_bound, upper_bound, 200)
y_plot = np.linspace(lower_bound, upper_bound, 200)
X, Y = np.meshgrid(x_plot, y_plot)
Z = np.array([fitness_function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = Z.reshape(X.shape)

# --- A) Static 3D Surface and 2D Contour Plot of the Function ---
fig_static = plt.figure(figsize=(16, 7))
ax1 = fig_static.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('3D Surface of Rastrigin Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
ax2 = fig_static.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
ax2.set_title('2D Contour Plot of Rastrigin Function')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig_static.colorbar(contour, ax=ax2, label='Fitness Value')
plt.tight_layout()

# --- B) Snapshots of Agent Positions ---
# สร้าง snapshot_iters ให้กระจายเท่าๆ กันใน 100 รอบ
snapshot_iters = np.linspace(1, max_iter, num=6, dtype=int)  # [1, 20, 40, 60, 80, 100]
fig_snapshots, axes = plt.subplots(1, len(snapshot_iters), figsize=(25, 5))
fig_snapshots.suptitle('Agent Positions at Different Iterations', fontsize=16)
for i, iter_num in enumerate(snapshot_iters):
    ax = axes[i]
    ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    # ตรวจสอบ index ไม่เกิน positions_history
    idx = min(iter_num - 1, len(positions_history) - 1)
    agent_pos = positions_history[idx]
    ax.scatter(agent_pos[:, 0], agent_pos[:, 1], color='red', s=15, zorder=2)
    ax.set_title(f'Iteration: {iter_num}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([lower_bound, upper_bound])
    ax.set_ylim([lower_bound, upper_bound])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- C) Animation of Agent Movement ---
fig_anim, ax_anim = plt.subplots(figsize=(8, 7))
ax_anim.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
ax_anim.set_xlim([lower_bound, upper_bound])
ax_anim.set_ylim([lower_bound, upper_bound])
ax_anim.set_xlabel('x')
ax_anim.set_ylabel('y')
ax_anim.set_title('GSA Agent Movement on Rastrigin Function')
scatter = ax_anim.scatter(positions_history[0][:, 0], positions_history[0][:, 1], color='red', s=20)
iter_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, color='white', fontsize=12)

def update(frame):
    scatter.set_offsets(positions_history[frame])
    iter_text.set_text(f'Iteration: {frame + 1}/{max_iter}')
    return scatter, iter_text

animation = FuncAnimation(fig_anim, update, frames=max_iter, interval=100, blit=True)

try:
    animation.save('gsa_animation.gif', writer='pillow', fps=10)
    print("Animation saved successfully as gsa_animation.gif")
except Exception as e:
    print(f"Could not save animation. Error: {e}")
    print("Please make sure you have Pillow installed (`pip install Pillow`)")

plt.show()