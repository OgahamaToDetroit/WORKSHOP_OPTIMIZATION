import numpy as np
import matplotlib.pyplot as plt
import os
from gsa_core import run_gsa
from gsa_functions import FUNCTION_CONFIG

def main():
    """
    Runs the GSA algorithm multiple times for a selected function to gather statistics.
    """
    # --- CHOOSE WHICH FUNCTION TO TEST HERE ---
    FUNCTION_TO_TEST = 'RASTRIGIN' # Options: 'RASTRIGIN', 'ROSENBROCK'
    # -----------------------------------------

    num_runs = 10
    config = FUNCTION_CONFIG[FUNCTION_TO_TEST]
    
    print("="*60)
    print(f"  STATISTICAL TEST FOR GSA ON {FUNCTION_TO_TEST} FUNCTION")
    print(f"  (D={config['dim']}, max_iter={config['max_iter']})")
    print("="*60)
    print(f"The algorithm will be run {num_runs} times to evaluate its consistency.")
    
    all_fitness_results = []
    
    for i in range(num_runs):
        print(f"\n----- Starting Run {i+1}/{num_runs} -----")
        
        # run_gsa is called with no seed, so each run is independent and random
        best_fitness, _ = run_gsa(
            objective_func=config['func'],
            dim=config['dim'],
            bounds=config['bounds'],
            max_iter=config['max_iter'],
            g0=config['g0'],
            alpha=config['alpha'],
            show_logs=False # Keep the output clean for the summary
        )
        
        all_fitness_results.append(best_fitness)
        print(f"  > Run {i+1} Complete. Result: Fitness = {best_fitness:.8f}")

    # Convert to NumPy array for easy calculations
    fitness_array = np.array(all_fitness_results)

    # --- Calculate and Print Statistics ---
    avg_fitness = np.mean(fitness_array)
    std_dev_fitness = np.std(fitness_array)
    best_overall_fitness = np.min(fitness_array)
    worst_overall_fitness = np.max(fitness_array)
    median_fitness = np.median(fitness_array)

    print("\n" + "="*60)
    print("  STATISTICAL ANALYSIS OF ALL RUNS")
    print("="*60)
    print(f"  Average (Mean): {avg_fitness:.8f}")
    print(f"  Standard Deviation: {std_dev_fitness:.8f}")
    print(f"  Median:         {median_fitness:.8f}")
    print(f"  Best Fitness Found:   {best_overall_fitness:.8f}")
    print(f"  Worst Fitness Found:  {worst_overall_fitness:.8f}")
    print("\n" + "="*60)

    # --- Generate and Save Bar Chart of Results ---
    output_dir = f"gsa_{FUNCTION_TO_TEST.lower()}_stats"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    runs = np.arange(1, num_runs + 1)
    plt.bar(runs, fitness_array, color='skyblue', edgecolor='black')
    
    # Add a line for the average value
    plt.axhline(y=avg_fitness, color='r', linestyle='--', label=f'Average: {avg_fitness:.4f}')
    
    plt.xlabel('Run Number')
    plt.ylabel('Final Best Fitness')
    plt.title(f'Distribution of GSA Results for {FUNCTION_TO_TEST} ({num_runs} Runs)')
    plt.xticks(runs)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    chart_path = os.path.join(output_dir, f'results_distribution.png')
    plt.savefig(chart_path)
    print(f"Results distribution chart saved to: {chart_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
