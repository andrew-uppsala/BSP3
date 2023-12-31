import sys
import subprocess
import csv
import os
import re

'''Run the script with python3 R_analysis.py <lower_bound> <upper_bound> <step> to start the analysis. 
Replace <lower_bound>, <upper_bound>, and <step> with the values you desire.
The script assumes that the output files generated by main.py are named sequentially, starting with output0.txt for the first run, 
output1.txt for the second run, and so on.
This script writes the results to a CSV file named R_analysis.csv. It overwrites the file if it already exists. 
You might want to include code to generate a unique filename if you wish to keep previous results.
Make sure to place this script (R_analysis.py) in the same directory as main.py and the output files 
to ensure that it can find and execute everything correctly.'''


def scan_output_file(file, csv_writer, r_value):
    min_risk_penalty, max_risk_penalty = float('inf'), float('-inf')
    min_fitness, max_fitness = float('inf'), float('-inf')

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "risk penalty of all solutions is :" in line:
                risk_penalty = float(line.split(":")[1].strip())
                min_risk_penalty = min(min_risk_penalty, risk_penalty)
                max_risk_penalty = max(max_risk_penalty, risk_penalty)
            elif "fitness of all solutions is :" in line:
                fitness = float(line.split(":")[1].strip())
                min_fitness = min(min_fitness, fitness)
                max_fitness = max(max_fitness, fitness)

    csv_writer.writerow({
        "R Value": r_value,
        "Min Risk Penalty": min_risk_penalty,
        "Max Risk Penalty": max_risk_penalty,
        "Min Fitness": min_fitness,
        "Max Fitness": max_fitness
    })

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 R_analysis.py <lower_bound> <upper_bound> <step>")
        return

    lower_bound = float(sys.argv[1])
    upper_bound = float(sys.argv[2])
    step = float(sys.argv[3])

    with open('R_analysis.csv', 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=["R Value", "Min Risk Penalty", "Max Risk Penalty", "Min Fitness", "Max Fitness"])
        csv_writer.writeheader()

        r_value = lower_bound
        counter = 0
        while r_value <= upper_bound:
            print(f"Running with R = {r_value}")
            subprocess.run(["python3", "main.py", f"R={r_value}"])
            file_name = f"output{counter}.txt"  # Assuming output files are named this way
            scan_output_file(file_name, csv_writer, r_value)
            counter += 1
            r_value += step

if __name__ == "__main__":
    main()
