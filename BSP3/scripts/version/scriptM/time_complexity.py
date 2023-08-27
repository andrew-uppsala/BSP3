import subprocess
import time
import csv

def measure_time_for_generations(generations):
    """
    Measures the average time for 10 runs of main.py with the given number of generations.
    """
    total_time = 0.0
    iterations = 10

    for _ in range(iterations):
        start_time = time.perf_counter()
        subprocess.run(["python3", "main.py", str(generations)])
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

    average_time = total_time / iterations
    return average_time

def run_time_complexity_analysis(generations_uprange):
    """
    Runs the time complexity analysis and stores the result in a CSV file.
    """
    csv_filename = "time_complexity_results.csv"

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Generations', 'Average Time']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        for i in range(1, generations_uprange + 1):
            avg_time = measure_time_for_generations(i)
            print(f"Average time for {i} generations: {avg_time:.5f} seconds")
            csv_writer.writerow({'Generations': i, 'Average Time': avg_time})

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 time_complexity.py <generations_uprange>")
        return

    generations_uprange = int(sys.argv[1])
    run_time_complexity_analysis(generations_uprange)

if __name__ == "__main__":
    import sys
    main()
