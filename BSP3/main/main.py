import argparse
import subprocess
import os

def has_custom_parameter():
    parser = argparse.ArgumentParser(description="Check for custom parameter")
    parser.add_argument("--custom_parameter", nargs="*", default=None)
    args = parser.parse_args()

    return args.custom_parameter is not None, args.custom_parameter

def create_temp_parameters(parameter_updates, parameters_path):
    temp_parameters = {}
    
    # Load default parameters
    with open(parameters_path, "r") as f:
        exec(f.read(), temp_parameters)
    
    # Apply updates
    for param in parameter_updates:
        if "=" in param:
            name, value = param.split("=")
            temp_parameters[name.strip()] = value.strip()
    
    # Create temp_parameters.py
    with open("temp_parameters.py", "w") as f:
        for param, value in temp_parameters.items():
            f.write(f"{param} = {value}\n")

def main():
    # Change working directory to the location of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    
    is_custom, parameter_updates = has_custom_parameter()
    parameters_path = os.path.join(script_directory, "parameters.py")

    if is_custom:
        create_temp_parameters(parameter_updates, parameters_path)
        # Change working directory back to the top-level directory (BSP3)
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subprocess.run(["python3", "main/body.py"])
        
        # Clean up temp_parameters.py
        os.remove("temp_parameters.py")
    else:
        # Change working directory back to the top-level directory (BSP3)
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subprocess.run(["python3", "main/body.py"])

if __name__ == "__main__":
    main()
