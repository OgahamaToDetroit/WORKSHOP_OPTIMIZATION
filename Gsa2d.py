import numpy as np
import math

# --- 1. กำหนดฟังก์ชันเป้าหมาย (Fitness Function) ---
def fitness_function(pos_2d):
    return pos_2d[0]**2 + pos_2d[1]**2

# --- 2. ตั้งค่าพารามิเตอร์ของ Algorithm ---
N = 30
D = 2
max_iter = 100
G0 = 10
alpha = 1
epsilon = 1e-7

# เพิ่มเข้ามา: กำหนดค่าเริ่มต้นและสุดท้ายของ Kbest
kbest_initial = N
kbest_final = 1

lower_bound = -100
upper_bound = 100

# --- 3. ส่วนของ Algorithm หลัก (GSA) ---
positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(N, D)
velocities = np.zeros((N, D))
masses = np.zeros(N)

best_agent_position = np.zeros(D)
best_agent_fitness = float('inf')

print(f"เริ่มต้นการทำงานของ GSA... (หาค่าต่ำสุดของฟังก์ชัน x^2 + y^2)")
print("-" * 60) # เพิ่มความยาวเส้น

for t in range(max_iter):
    fitness_values = np.array([fitness_function(p) for p in positions])

    best_fitness = np.min(fitness_values)
    worst_fitness = np.max(fitness_values)
    
    if best_fitness < best_agent_fitness:
        best_agent_fitness = best_fitness
        best_agent_position = positions[np.argmin(fitness_values)].copy()

    m_normalized = (fitness_values - worst_fitness) / (best_fitness - worst_fitness + epsilon)
    masses = m_normalized / (np.sum(m_normalized) + epsilon)

    G = G0 * math.exp(-alpha * t / max_iter)

    # --- การปรับปรุงด้วย Kbest ---
    # 1. คำนวณค่า kbest สำหรับรอบปัจจุบัน ให้ลดลงแบบเส้นตรง
    kbest = round(kbest_initial - (kbest_initial - kbest_final) * (t / max_iter))

    # 2. เรียงลำดับ Agent ตามค่า fitness จากดีที่สุดไปแย่ที่สุด
    sorted_indices = np.argsort(fitness_values)

    forces = np.zeros((N, D))
    accelerations = np.zeros((N, D))

    for i in range(N):
        total_force_on_i = np.zeros(D)
        #  วนลูปเฉพาะ Agent ที่ดีที่สุด kbest ตัวแรกเท่านั้น
        for j_idx in sorted_indices[0:kbest]:
            j = j_idx # j คือ index ของ agent ที่อยู่ในกลุ่ม kbest
            if i != j:
                displacement_vec = positions[j] - positions[i]
                distance = np.linalg.norm(displacement_vec)
                force_ij = G * (masses[i] * masses[j] / (distance + epsilon)) * displacement_vec
                total_force_on_i += np.random.rand() * force_ij
        
        forces[i] = total_force_on_i
        accelerations[i] = forces[i] / (masses[i] + epsilon)

    velocities = np.random.rand(N, 1) * velocities + accelerations
    positions = positions + velocities
    positions = np.clip(positions, lower_bound, upper_bound)
    
    pos_str = f"[{best_agent_position[0]:.6f}, {best_agent_position[1]:.6f}]"
    # เพิ่มการแสดงค่า kbest ในแต่ละรอบ
    print(f"รอบที่ {t+1:3d} (kbest={kbest:2d}): คำตอบที่ดีที่สุด = {pos_str}, ค่า Fitness = {best_agent_fitness:.6f}")

print("-" * 60)
print("GSA ทำงานเสร็จสิ้น")
pos_str = f"[{best_agent_position[0]:.6f}, {best_agent_position[1]:.6f}]"
print(f"คำตอบที่ดีที่สุดที่พบคือ x, y = {pos_str}")
print(f"ค่าต่ำสุดของฟังก์ชันคือ f(x,y) = {best_agent_fitness:.6f}")