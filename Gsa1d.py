import numpy as np
import math

# --- 1. กำหนดฟังก์ชันเป้าหมาย (Fitness Function) ---
# นี่คือ "ปัญหา" ที่เราต้องการหาคำตอบที่ดีที่สุด (ค่าต่ำสุด)
# ในที่นี้ เราใช้ฟังก์ชันง่ายๆ คือ f(x) = x^2 ซึ่งมีคำตอบที่ดีที่สุดคือ x = 0
def fitness_function(x):
    return x**2

# --- 2. ตั้งค่าพารามิเตอร์ของ Algorithm ---
N = 30                # จำนวน Agent (วัตถุ) ที่จะใช้ในการค้นหา
max_iter = 10      # จำนวนรอบสูงสุดที่จะให้อัลกอริทึมทำงาน
G0 = 100              # ค่าคงที่แรงโน้มถ่วงเริ่มต้น (G_0)
alpha = 20            # ค่าคงที่สำหรับลดค่า G
epsilon = 1e-7        # ค่าคงที่เล็กๆ เพื่อป้องกันการหารด้วยศูนย์

# กำหนดขอบเขตของพื้นที่ค้นหา (Search Space)
lower_bound = -100
upper_bound = 100

# --- 3. ส่วนของ Algorithm หลัก (GSA) ---

# --- ขั้นตอนที่ 1: การกำหนดค่าเริ่มต้น (Initialization) ---
# สร้าง Agent ขึ้นมา N ตัว และสุ่มตำแหน่งเริ่มต้น (x) และความเร็ว (v)
# np.random.rand(N) จะสุ่มค่าระหว่าง 0 ถึง 1 ออกมา N ค่า
positions = lower_bound + (upper_bound - lower_bound) * np.random.rand(N)
velocities = np.zeros(N) # ให้ความเร็วเริ่มต้นเป็น 0 ทั้งหมด
masses = np.zeros(N)

# ตัวแปรสำหรับเก็บคำตอบที่ดีที่สุดที่เคยเจอ
best_agent_position = 0
best_agent_fitness = float('inf') # ตั้งค่าเริ่มต้นให้เป็นค่าที่สูงมากๆ

print(f"เริ่มต้นการทำงานของ GSA... (หาค่าต่ำสุดของฟังก์ชัน x^2)")
print("-" * 40)

# --- เริ่มการวนซ้ำเพื่อค้นหาคำตอบ ---
for t in range(max_iter):

    # --- ขั้นตอนที่ 2: การประเมิน Fitness และคำนวณมวล ---
    # คำนวณค่า fitness ของ agent แต่ละตัวจากตำแหน่งปัจจุบัน
    fitness_values = np.array([fitness_function(p) for p in positions])

    # หาค่า fitness ที่ดีที่สุด (ต่ำสุด) และแย่ที่สุด (สูงสุด)
    best_fitness = np.min(fitness_values)
    worst_fitness = np.max(fitness_values)
    
    # อัปเดตคำตอบที่ดีที่สุดที่เคยเจอ (Global Best)
    if best_fitness < best_agent_fitness:
        best_agent_fitness = best_fitness
        best_agent_position = positions[np.argmin(fitness_values)]

    # คำนวณมวล (M) ตามสมการ
    # m_i(t) = (fit_i(t) - worst(t)) / (best(t) - worst(t))
    # M_i(t) = m_i(t) / sum(m_j(t))
    # Agent ที่มี fitness ดี (ค่าน้อย) จะมีมวลมากกว่า
    m_normalized = (fitness_values - worst_fitness) / (best_fitness - worst_fitness + epsilon)
    masses = m_normalized / (np.sum(m_normalized) + epsilon)

    # --- ส่วนของ Exploration และ Exploitation ---
    # ค่า G จะลดลงเรื่อยๆ เมื่อเวลาผ่านไป (t เพิ่มขึ้น)
    # ช่วงแรก: G มีค่ามาก -> แรงดึงดูดสูง -> Agent เคลื่อนที่เยอะ (เน้น Exploration)
    # ช่วงท้าย: G มีค่าน้อย -> แรงดึงดูดต่ำ -> Agent เคลื่อนที่ช้าลง (เน้น Exploitation)
    G = G0 * math.exp(-alpha * t / max_iter)

    # --- ขั้นตอนที่ 3 และ 4: คำนวณแรง, ความเร่ง ---
    forces = np.zeros(N)
    accelerations = np.zeros(N)

    # วนลูปเพื่อคำนวณแรงดึงดูดระหว่าง Agent ทุกคู่
    for i in range(N):
        total_force_on_i = 0
        for j in range(N):
            if i != j:
                # คำนวณแรงที่ Agent j กระทำต่อ Agent i ตามสมการ F_ij
                # F_ij(t) = G(t) * (M_i*M_j / (|x_j-x_i|+eps)) * (x_j - x_i)
                force_ij = G * (masses[i] * masses[j] / (abs(positions[j] - positions[i]) + epsilon)) * (positions[j] - positions[i])
                
                # รวมแรงจาก Agent อื่นๆ ที่สุ่มเข้ามา
                total_force_on_i += np.random.rand() * force_ij
        
        forces[i] = total_force_on_i
        # คำนวณความเร่งตามสมการ a_i(t) = F_i(t) / M_i(t)
        accelerations[i] = forces[i] / (masses[i] + epsilon)

    # --- ขั้นตอนที่ 5: คำนวณความเร็ว ---
    # อัปเดตความเร็วของ Agent แต่ละตัวตามสมการ v_i(t+1) = rand * v_i(t) + a_i(t)
    # การสุ่ม rand เข้าไปในความเร็ว ช่วยเพิ่มการสำรวจ (Exploration)
    velocities = np.random.rand(N) * velocities + accelerations

    # --- ขั้นตอนที่ 6: อัปเดตตำแหน่ง ---
    # ย้ายตำแหน่งของ Agent ไปยังตำแหน่งใหม่ตามความเร็ว
    positions = positions + velocities

    # ทำให้แน่ใจว่าตำแหน่งของ Agent ยังอยู่ในขอบเขตที่กำหนด
    positions = np.clip(positions, lower_bound, upper_bound)
    
    # แสดงผลลัพธ์ในแต่ละรอบ
    print(f"รอบที่ {t+1}: คำตอบที่ดีที่สุด = {best_agent_position:.12f}, ค่า Fitness = {best_agent_fitness:.12f}")

# --- สิ้นสุดการทำงาน ---
print("-" * 40)
print("GSA ทำงานเสร็จสิ้น")
print(f"คำตอบที่ดีที่สุดที่พบคือ x = {best_agent_position:.12f}")
print(f"ค่าต่ำสุดของฟังก์ชันคือ f(x) = {best_agent_fitness:.12f}")