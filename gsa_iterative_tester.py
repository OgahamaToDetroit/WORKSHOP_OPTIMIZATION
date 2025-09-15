import numpy as np
import matplotlib.pyplot as plt
import os
from gsa_core import run_gsa
from gsa_configs import FUNCTION_CONFIG

def main():
    """
    Runs a continuous GSA experiment, broken into 10 blocks,
    passing the state from one block to the next to observe long-term convergence.
    """
    # --- CHOOSE WHICH CONFIGURATION TO TEST HERE ---
    CONFIG_TO_TEST = 'RASTRIGIN_20D' 
    # Options: 'ROSENBROCK_1000', 'ROSENBROCK_100', 'RASTRIGIN_20D', 'RASTRIGIN_2D_VI'
    # -----------------------------------------------

    num_blocks = 10
    config = FUNCTION_CONFIG[CONFIG_TO_TEST]
    
    print("="*60)
    print(f"  ITERATIVE RUN TEST ({num_blocks} Continuous Blocks)")
    print(f"  CONFIGURATION: {config['name']}")
    print("="*60)
    
    # This will store the final state of the agents after each block
    saved_state = None 
    
    # This will store the best fitness found at the end of each block
    fitness_over_blocks = []

    for i in range(num_blocks):
        print(f"\n----- Starting Block {i+1}/{num_blocks} (Iterations: {config['max_iter']}) -----")
        
        # The 'saved_state' is passed to the next run.
        # For the first run (i=0), saved_state is None, so GSA starts randomly.
        best_fitness, saved_state = run_gsa(
            objective_func=config['func'],
            dim=config['dim'],
            bounds=config['bounds'],
            max_iter=config['max_iter'],
            n_agents=config['n_agents'],
            g0=config['g0'],
            alpha=config['alpha'],
            initial_state=saved_state # Pass the previous state to the new run
        )
        
        fitness_over_blocks.append(best_fitness)
        print(f"  > Block {i+1} Complete. Best Fitness in this block = {best_fitness:.8f}")

    fitness_array = np.array(fitness_over_blocks)

    # --- Calculate and Print Statistics of the final fitness values from each block ---
    avg_fitness = np.mean(fitness_array)
    std_dev_fitness = np.std(fitness_array)

    print("\n" + "="*60)
    print("  STATISTICAL ANALYSIS OF BLOCK RESULTS")
    print("="*60)
    print(f"  Average of final fitness values: {avg_fitness:.8f}")
    print(f"  Standard Deviation of final values: {std_dev_fitness:.8f}")
    print(f"  Final Best Fitness after all blocks: {fitness_array[-1]:.8f}")
    print("\n" + "="*60)

    # --- Generate and Save Line Chart of Improvement ---
    output_dir = f"gsa_{CONFIG_TO_TEST.lower()}_iterative_run"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    blocks = np.arange(1, num_blocks + 1)
    total_iterations = blocks * config['max_iter']
    
    plt.plot(total_iterations, fitness_array, marker='o', linestyle='-', color='purple')
    
    plt.xlabel('Total Iterations Completed')
    plt.ylabel('Best Fitness Found')
    plt.title(f'Continuous Improvement of GSA on {config["name"]}')
    plt.xticks(total_iterations)
    plt.yscale('log')
    plt.grid(True)
    
    chart_path = os.path.join(output_dir, 'iterative_improvement_chart.png')
    plt.savefig(chart_path)
    print(f"Improvement chart saved to: {chart_path}")
    
    plt.show()

if __name__ == "__main__":
    main()

