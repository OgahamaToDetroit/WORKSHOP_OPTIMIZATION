import numpy as np

# --- 1. Define Fitness Functions ---

def rastrigin_func(position):
    """
    N-Dimensional Rastrigin function.
    Global minimum is f(0,..,0) = 0.
    """
    d = len(position)
    total_sum = 0
    for i in range(d):
        xi = position[i]
        total_sum += (xi**2 - 10 * np.cos(2 * np.pi * xi) + 10)
    return total_sum

def rosenbrock_func(position):
    """
    N-Dimensional Rosenbrock function.
    This implementation works for D >= 2.
    Global minimum is f(1,..,1) = 0.
    """
    d = len(position)
    total_sum = 0
    for i in range(d - 1):
        x_i = position[i]
        x_i_plus_1 = position[i+1]
        total_sum += 100 * (x_i_plus_1 - x_i**2)**2 + (1 - x_i)**2
    return total_sum

# --- 2. Function Configurations ---
# This dictionary holds all the parameters needed for each test function.
# It makes the tester script clean and easy to switch between functions.

FUNCTION_CONFIG = {
    'RASTRIGIN': {
        'func': rastrigin_func,
        'dim': 20,
        'bounds': (-5.12, 5.12),
        'max_iter': 1000,
        'g0': 200,
        'alpha': 10
    },
    'ROSENBROCK': {
        'func': rosenbrock_func,
        'dim': 2,
        'bounds': (-2, 2),
        'max_iter': 1000, # Using 1000 for consistency as per paper
        'g0': 100,
        'alpha': 20
    },
    # You can add configurations for f1 and f4 here in the future
    # 'SPHERE': { ... },
    # 'GRIEWANK': { ... },
}
