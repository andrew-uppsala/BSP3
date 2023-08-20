import re
import sys

def declutter_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    decluttered_lines = []
    is_comment_block = False
    empty_lines = 0

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith('#'):
            continue  # Skip commented lines

        if stripped_line == '':
            empty_lines += 1
            if empty_lines <= 2:
                decluttered_lines.append(line)
        else:
            empty_lines = 0
            decluttered_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(decluttered_lines)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python declutter.py <input_filename>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = "decluttered.py"

    declutter_file(input_filename, output_filename)
    print(f"File '{input_filename}' decluttered and saved as '{output_filename}'.")
