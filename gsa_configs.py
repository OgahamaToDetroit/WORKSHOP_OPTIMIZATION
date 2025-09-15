import numpy as np

# --- 1. Fitness Functions ---

def rastrigin_func(position):
    """N-Dimensional Rastrigin function."""
    return np.sum(position**2 - 10 * np.cos(2 * np.pi * position) + 10)

def rosenbrock_func(position):
    """N-Dimensional Rosenbrock function."""
    return np.sum(100.0 * (position[1:] - position[:-1]**2.0)**2.0 + (1 - position[:-1])**2.0)

# --- 2. Function Configurations Dictionary ---
# This centralizes all parameters for easy testing.

FUNCTION_CONFIG = {
    'ROSENBROCK_1000': {
        'name': 'Rosenbrock (1000 Iterations)',
        'func': rosenbrock_func,
        'dim': 2,
        'bounds': (-2, 2),
        'max_iter': 1000,
        'n_agents': 20,
        'g0': 100,
        'alpha': 20
    },
    'ROSENBROCK_100': {
        'name': 'Rosenbrock (100 Iterations)',
        'func': rosenbrock_func,
        'dim': 2,
        'bounds': (-2, 2),
        'max_iter': 100,
        'n_agents': 20,
        'g0': 100,
        'alpha': 20
    },
    'RASTRIGIN_20D': {
        'name': 'Rastrigin (20 Dimensions)',
        'func': rastrigin_func,
        'dim': 20,
        'bounds': (-5.12, 5.12),
        'max_iter': 1000,
        'n_agents': 100, # Tuned for high-dim
        'g0': 200,      # Tuned for high-dim
        'alpha': 10       # Tuned for high-dim
    },
    'RASTRIGIN_2D_VI': {
        'name': 'Rastrigin 2D (from gsa2d_vi.py)',
        'func': rastrigin_func,
        'dim': 2,
        'bounds': (-5.12, 5.12),
        'max_iter': 100,
        'n_agents': 100,
        'g0': 500,      # As per original file
        'alpha': 15       # As per original file
    }
}

