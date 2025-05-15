import os
import shutil

# Base directory containing all the folders
base_dir = r"D:\Machine learning\Arch technologhy\YOLO11\original_dataset_100_epochs\segmentation"

# Loop through each folder in the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)

        for file in files:
            file_path = os.path.join(folder_path, file)
            new_path = os.path.join(base_dir, file)

            # Move the file to base directory
            shutil.move(file_path, new_path)
            print(f"Moved '{file}' to '{base_dir}'")

        # After moving, delete the now-empty folder
        if not os.listdir(folder_path):  # Check if folder is empty
            os.rmdir(folder_path)
            print(f"Deleted empty folder '{folder_name}'")
