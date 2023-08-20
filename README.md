<<<<<<< HEAD
# BSP3


!!! Please notice that due to the poor design of the program, the code will not run independently on another machine unless the paths are replaced manually. This is a very poor practice, but changing it seemed to break other parts of the code, so for now it is not possible to modify it to make it portable. However, as far as my tasks are concerned, the main function behaves as expected.

The code needs to be run from the equivalent of the directory /Users/abnerandreymartinezzamudio/Downloads/BSP3/BSP3/scripts/version/scriptM. 

=======
Checking for temp_parameters:

The main.py script is responsible for checking if a command-line argument has been passed. If this is the case, the function will temporarily modify the parameter passed as argument and then call the program as a subprocess, then restore the original parameter. Otherwise, it will simply call the program.

Here's a step-by-step explanation of the process:

You run "python3 main.py h=0.06" in the command line.

The read_original_value function detects that there are command-line arguments, so it searched for that variable in the first 250 lines of full_code.py. Then, it updates the value of h to 0.06.

The subprocess runs 

Finally, a text file is produced in the same directory containing the outputs, and the parameter is restored to its original form. 

An example output file is included for reference. 


