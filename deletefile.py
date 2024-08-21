import os
import shutil
n=8
# Define the base folder name and the range
base_folder_name = 'MMGP_OL'
folder_range = range(0, 0+n)

# Define subfolders to delete content from
subfolders_to_empty = ['Re500', 'vortex']

# Loop through each main folder
for i in folder_range:
    folder_name = f"{base_folder_name}{i}"

    # Loop through each subfolder
    for subfolder in subfolders_to_empty:
        # Create full path to the subfolder
        full_subfolder_path = os.path.join(folder_name, subfolder)

        # Check if the subfolder exists
        if os.path.exists(full_subfolder_path):
            # Delete all files in the subfolder
            for filename in os.listdir(full_subfolder_path):
                file_path = os.path.join(full_subfolder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"Subfolder {full_subfolder_path} does not exist.")
