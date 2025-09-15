import numpy as np
import matplotlib.pyplot as plt
import os
from gsa_core import run_gsa
from gsa_configs import FUNCTION_CONFIG

def main():
    """
    Runs the GSA algorithm 10 times independently to gather statistics 
    on its performance and stability.
    """
    # --- CHOOSE WHICH CONFIGURATION TO TEST HERE ---
    CONFIG_TO_TEST = 'RASTRIGIN_20D' # Options: 'ROSENBROCK_1000', 'ROSENBROCK_100', 'RASTRIGIN_20D', 'RASTRIGIN_2D_VI'
    # -----------------------------------------------

    num_runs = 10
    config = FUNCTION_CONFIG[CONFIG_TO_TEST]
    
    print("="*60)
    print(f"  STATISTICAL TEST ({num_runs} Independent Runs)")
    print(f"  CONFIGURATION: {config['name']}")
    print("="*60)
    
    all_fitness_results = []
    
    for i in range(num_runs):
        print(f"\n----- Starting Run {i+1}/{num_runs} -----")
        # In each run, initial_state is None, so it starts fresh every time.
        best_fitness, _ = run_gsa(
            objective_func=config['func'],
            dim=config['dim'],
            bounds=config['bounds'],
            max_iter=config['max_iter'],
            n_agents=config['n_agents'],
            g0=config['g0'],
            alpha=config['alpha']
        )
        all_fitness_results.append(best_fitness)
        print(f"  > Run {i+1} Complete. Final Fitness = {best_fitness:.8f}")

    fitness_array = np.array(all_fitness_results)

    # --- Calculate and Print Statistics ---
    avg_fitness = np.mean(fitness_array)
    std_dev_fitness = np.std(fitness_array)
    best_overall_fitness = np.min(fitness_array)
    worst_overall_fitness = np.max(fitness_array)

    print("\n" + "="*60)
    print("  STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    print(f"  Average (Mean): {avg_fitness:.8f}")
    print(f"  Standard Deviation: {std_dev_fitness:.8f}")
    print(f"  Best Fitness Found:   {best_overall_fitness:.8f}")
    print(f"  Worst Fitness Found:  {worst_overall_fitness:.8f}")
    print("\n" + "="*60)

    # --- Generate and Save Bar Chart of Results ---
    output_dir = f"statistical_runs_output/{CONFIG_TO_TEST.lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    runs = np.arange(1, num_runs + 1)
    plt.bar(runs, fitness_array, color='skyblue', edgecolor='black')
    plt.axhline(y=avg_fitness, color='r', linestyle='--', label=f'Average: {avg_fitness:.4f}')
    
    plt.xlabel('Run Number')
    plt.ylabel('Final Best Fitness')
    plt.title(f'GSA Performance Stability on {config["name"]}')
    plt.xticks(runs); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    chart_path = os.path.join(output_dir, f'statistical_results.png')
    plt.savefig(chart_path)
    print(f"Results chart saved to: {chart_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
