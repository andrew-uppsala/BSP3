<<<<<<< HEAD
# BSP3
=======
Checking for temp_parameters:

In the main.py script, the create_temp_parameters() function is responsible for creating a temporary parameters file, temp_parameters.py, with the updated parameter values. This function is only called when you pass command-line arguments to main.py. If you don't provide any command-line arguments, the temp_parameters.py file won't be created, and body.py will import the regular parameters.py file.

create_temp_parameters() Function:

The create_temp_parameters() function is used to generate the temporary parameters file. It reads the existing parameter definitions from the parameters.py file and updates them based on the command-line arguments you provide. For example, if you run python3 main.py h=0.06, it will update the h parameter in the temp_parameters.py file to have a value of 0.06. Any parameters that are not mentioned in the command-line arguments will retain their values from parameters.py.

Here's a step-by-step explanation of the process:

You run "python3 main.py h=0.06" in the command line.

The has_custom_parameter() function detects that there are command-line arguments, and create_temp_parameters() is called.

create_temp_parameters() reads the original parameter definitions from parameters.py and updates the value of h to 0.06 in the temp_parameters.py file.

The subprocess runs body.py.

Inside body.py, it will check whether temp_parameters.py exists. If it does, it will import parameters from temp_parameters.py; otherwise, it will import them from parameters.py.
>>>>>>> 98ca81e (Initial commit)
