import numpy as np
from gsa_rastrigin_nd import run_gsa

def main():
    """
    Runs the GSA algorithm for the Rastrigin function multiple times 
    to gather statistics on its performance, as required by the project.
    """
    num_runs = 10
    
    print("="*60)
    print("  STATISTICAL TEST FOR GSA ON RASTRIGIN FUNCTION (D=20)")
    print("="*60)
    print(f"The algorithm will be run {num_runs} times to evaluate its consistency.")
    
    all_fitness_results = []
    
    for i in range(num_runs):
        print(f"\n----- Starting Run {i+1}/{num_runs} -----")
        
        # We run without logs to keep the main output clean for the test.
        # A different seed is used for each run by default (seed=None).
        best_fitness, _, _, _ = run_gsa(show_logs=False, seed=None) 
        
        all_fitness_results.append(best_fitness)
        
        print(f"  > Run {i+1} Complete. Result: Fitness = {best_fitness:.8f}")

    # Convert the list of results to a NumPy array for easier calculations
    fitness_array = np.array(all_fitness_results)

    # --- Calculate and Print Statistics ---
    print("\n" + "="*60)
    print("  STATISTICAL ANALYSIS OF ALL RUNS")
    print("="*60)
    
    avg_fitness = np.mean(fitness_array)
    std_dev_fitness = np.std(fitness_array)
    best_overall_fitness = np.min(fitness_array)
    worst_overall_fitness = np.max(fitness_array)
    median_fitness = np.median(fitness_array)

    print(f"Number of Runs: {num_runs}")
    print(f"\n-- Fitness Statistics --")
    print(f"  Average (Mean): {avg_fitness:.8f}")
    print(f"  Standard Deviation: {std_dev_fitness:.8f}")
    print(f"  Median:         {median_fitness:.8f}")
    print(f"\n-- Performance Extremes --")
    print(f"  Best Fitness Found:   {best_overall_fitness:.8f}")
    print(f"  Worst Fitness Found:  {worst_overall_fitness:.8f}")
    print("\n" + "="*60)
    print("--- Test Complete ---")


if __name__ == "__main__":
    main()
