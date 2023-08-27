import sys
import subprocess
import fileinput
import re
import os

# Path to the full_code.py and before_cost.py files
FULL_CODE_PATH = "full_code.py"
BEFORE_COST_PATH = "before_cost.py"

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
    original_stdout = sys.stdout
    with fileinput.input(FULL_CODE_PATH, inplace=True) as f:
        for line in f:
            if param_name in line:
                line = line.replace(f"{param_name} =", f"{param_name} = {new_value}")
            sys.stdout.write(line)
    sys.stdout = original_stdout  # Restore original stdout

# Function to restore the original parameter value in full_code.py
def restore_original_value(param_name, original_value):
    modify_parameter(param_name, original_value)

def store_output(output_lines):
    output_filename = "output.txt"
    counter = 1

    # Check if the file exists
    while os.path.isfile(output_filename):
        output_filename = f"output{counter}.txt"
        counter += 1

    with open(output_filename, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

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

    # Store the output in a unique file
    store_output(output_lines)

def read_original_num_gen():
    with open(BEFORE_COST_PATH, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines, start=1):
            if '#' in line:
                line = line[:line.index('#')]
            if "num_gen" in line:
                match = re.search(r"num_gen\s*=\s*(.+)", line)
                if match:
                    original_value = match.group(1).strip()
                    return original_value
    return None

def modify_num_gen(new_value):
    original_stdout = sys.stdout
    with fileinput.input(BEFORE_COST_PATH, inplace=True) as f:
        for line in f:
            if "num_gen" in line:
                line = line.replace("num_gen =", f"num_gen = {new_value}")
            sys.stdout.write(line)
    sys.stdout = original_stdout  # Restore original stdout

def run_program_without_param_change():
    # Capture the standard output
    process = subprocess.Popen(["python3", BEFORE_COST_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    output_lines = stdout.decode().splitlines()

    # Store the output in a unique file
    store_output(output_lines)

def main():
    original_num_gen = read_original_num_gen()

    if len(sys.argv) == 1:
        subprocess.run(["python3", BEFORE_COST_PATH])
        return
    elif len(sys.argv) == 2:
        # Check if the second argument contains an "=" sign (indicating it's a parameter to change)
        if "=" in sys.argv[1]:
            param_definition = sys.argv[1]
            param_name, new_value = param_definition.split("=")
            param_name = param_name.strip()
            new_value = new_value.strip()
            run_program(param_name, new_value)
        else:
            # If not, assume it's a new number of generations
            new_num_gen = sys.argv[1].strip()
            if original_num_gen is not None:
                modify_num_gen(new_num_gen)
                run_program_without_param_change()
                modify_num_gen(original_num_gen)  # Restore original value

    elif len(sys.argv) == 3:
        param_definition = sys.argv[1]
        param_name, new_value = param_definition.split("=")
        param_name = param_name.strip()
        new_value = new_value.strip()
        
        new_num_gen = sys.argv[2].strip()
        if original_num_gen is not None:
            modify_num_gen(new_num_gen)

        run_program(param_name, new_value)

        # Restore original num_gen value if it was modified
        if original_num_gen is not None:
            modify_num_gen(original_num_gen)
    else:
        print("Usage: python3 main.py [<param_definition>] [<num_gen_value>]")
        return

if __name__ == "__main__":
    main()
