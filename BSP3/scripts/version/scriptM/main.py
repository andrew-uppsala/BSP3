import sys
import subprocess
import fileinput
import re

# Path to the full_code.py and before_cost.py files
FULL_CODE_PATH = "full_code.py"
BEFORE_COST_PATH = "before_cost.py"
OUTPUT_FILE = "output.txt"

# Function to read the original parameter value from full_code.py
def read_original_value(param_name):
    with open(FULL_CODE_PATH, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:250], start=1):
            if '#' in line:
                line = line[:line.index('#')]  # Remove comments
            if param_name in line:
                match = re.search(rf"{param_name}\s*=\s*(.+)", line)
                if match:
                    original_value = match.group(1).strip()
                    return original_value, i
    return None, None

# Function to modify the parameter value in full_code.py
def modify_parameter(param_name, new_value):
    for line in fileinput.input(FULL_CODE_PATH, inplace=True):
        if param_name in line:
            line = line.replace(f"{param_name} =", f"{param_name} = {new_value}")
        sys.stdout.write(line)

# Function to restore the original parameter value in full_code.py
def restore_original_value(param_name, original_value):
    modify_parameter(param_name, original_value)

def store_output(output_lines):
    with open(OUTPUT_FILE, "w") as f:
        for line in output_lines:
            f.write(line)

def run_program(param_name, new_value):
    original_value, line_number = read_original_value(param_name)
    if original_value is None:
        print("Parameter not found in the first 250 lines.")
        return
    print(f"Parameter {param_name} found at line {line_number}")
    modify_parameter(param_name, new_value)
    # Capture the standard output
    process = subprocess.Popen(["python3", BEFORE_COST_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    output_lines = stdout.decode().splitlines()
    print(f"Parameter {param_name} restored to {original_value}")
    restore_original_value(param_name, original_value)
    with open(OUTPUT_FILE, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

def main():
    if len(sys.argv) == 1:
        # No parameter definition provided, run before_cost.py directly
        subprocess.run(["python3", BEFORE_COST_PATH])
        return
    elif len(sys.argv) != 2:
        print("Usage: python3 main.py <parameter_definition>")
        return
    param_definition = sys.argv[1]
    param_name, new_value = param_definition.split("=")
    param_name = param_name.strip()
    new_value = new_value.strip()
    run_program(param_name, new_value)

def store_output(output_lines):
    with open(OUTPUT_FILE, "w") as f:
        for line in output_lines:
            f.write(line)

if __name__ == "__main__":
    main()
