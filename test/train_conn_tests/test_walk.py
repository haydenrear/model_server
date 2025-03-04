import os
import unittest


import os
import shutil
import traceback

# Example usage:
source_directory = "/Users/hayde/IdeaProjects/drools"
destination_directory = "/Volumes/RHEL-8-3-0-/obsidian/drools-backup-dec-3-2023"

def copy_files(src_dir, dest_dir):
    os.chdir(src_dir)
    for root, dirs, files in os.walk("."):
        for file in filter(lambda x: x is not None and len(x) != 0 and x != '.', files):
            if root != '.':
                src_file = os.path.join(src_dir, root, file)
            else:
                src_file = os.path.join(src_dir, file)
            if root == '.':
                dest_file = os.path.join(dest_dir, file)
            else:
                dest_file = os.path.join(dest_dir, root.replace('./', ''), file)
            try:
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copyfile(src_file, dest_file)
            except Exception as e:
                print(f"Error: {e}.")
                error_log_file = os.path.join(dest_dir, "error_log.txt")
                with open(error_log_file, "a") as error_log:
                    error_log.write(f"Error copying {src_file} to {dest_file}:\n")
                    error_log.write(traceback.format_exc())
                    error_log.write("\n")



# copy_files(source_directory, destination_directory)



