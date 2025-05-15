import os

# Set the base directory
base_dir = r"D:\Machine learning\Arch technologhy\YOLO11\orignal_dataset_polygon_to_bounding_box\segmented_results\New folder"

# Loop through each folder in the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Ensure we're working with a folder
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        
        # Process only if there's exactly one file (assumed to be the image)
        if len(files) == 1:
            original_file = files[0]
            original_path = os.path.join(folder_path, original_file)
            
            # Get the file extension
            _, ext = os.path.splitext(original_file)
            
            # Set the new filename
            new_filename = folder_name + ext
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(original_path, new_path)
            print(f"Renamed '{original_file}' to '{new_filename}' in folder '{folder_name}'")
        else:
            print(f"Skipped folder '{folder_name}' (contains {len(files)} files)")
