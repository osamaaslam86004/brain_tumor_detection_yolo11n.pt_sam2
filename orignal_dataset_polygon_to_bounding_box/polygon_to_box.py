import os

import cv2
import numpy as np


def is_bbox_format(line):
    """
    Checks if a line in a label file represents a bounding box.

    Args:
        line (str): A string representing a line from a label file.

    Returns:
        bool: True if the line represents a bounding box (has 5 space-separated values), False otherwise.
    """
    return len(line.strip().split()) == 5


def convert_polygon_to_bbox(coords, img_w, img_h):
    """
    Converts a polygon annotation to a bounding box annotation in YOLO format.

    Args:
        coords (list): A list of normalized coordinates representing the polygon's vertices (x1, y1, x2, y2, ...).
        img_w (int): The width of the image in pixels.
        img_h (int): The height of the image in pixels.

    Returns:
        list: A list containing the normalized bounding box coordinates [x_center, y_center, width, height].
              These values are normalized relative to the image width and height (range 0.0 to 1.0).
    """
    # Convert normalized polygon to absolute pixel coords
    polygon = []
    for i in range(0, len(coords), 2):
        x = int(float(coords[i]) * img_w)
        y = int(float(coords[i + 1]) * img_h)
        polygon.append([x, y])

    polygon_np = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(polygon_np)  # Get bounding rectangle

    # Convert back to normalized bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h

    # Clamp to [0.0, 1.0]
    # ensures that the values remain within the valid range for normalized
    x_center = max(0.0, min(x_center, 1.0))
    y_center = max(0.0, min(y_center, 1.0))
    norm_w = max(0.0, min(norm_w, 1.0))
    norm_h = max(0.0, min(norm_h, 1.0))

    return [x_center, y_center, norm_w, norm_h]


def process_labels_folder(labels_folder, image_folder):
    """
    Processes all label files in a folder, converting polygon annotations to bounding box annotations if necessary.

    Args:
        labels_folder (str): The path to the folder containing the label files (e.g., in YOLO format).
        image_folder (str): The path to the folder containing the corresponding image files.
    """
    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            label_path = os.path.join(labels_folder, filename)
            image_path = os.path.join(
                image_folder, filename.replace(".txt", ".jpg")
            )  # Assumes .jpg images, adjust if needed

            if not os.path.exists(image_path):
                print(f"Image not found for {filename}, skipping.")
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image {image_path}, skipping")
                continue
            img_h, img_w = img.shape[:2]

            with open(label_path, "r") as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                parts = (
                    line.strip().split()
                )  #  return ['1', '0.5', '0.6', '0.2', '0.1'].
                if not parts:  # Skip empty lines
                    continue
                class_id = parts[0]

                if is_bbox_format(line):
                    updated_lines.append(line.strip())  # Already in bbox format
                else:
                    coords = parts[1:]
                    if (
                        len(coords) % 2 != 0 or len(coords) < 6
                    ):  # Validate polygon coordinates.  Need at least 3 points (6 coords).
                        print(
                            f"Invalid polygon format in {filename}: {line.strip()}. Skipping."
                        )
                        continue

                    bbox = convert_polygon_to_bbox(coords, img_w, img_h)
                    updated_line = f"{class_id} {' '.join([f'{v:.6f}' for v in bbox])}"
                    updated_lines.append(updated_line)

            # Overwrite with updated annotations (all in bbox format)
            unique_lines = list(set(updated_lines))  # Remove duplicates
            with open(label_path, "w") as f:
                print(unique_lines)
                f.write("\n".join(unique_lines))


# === CONFIGURE PATHS ===
labels_path = r"D:\Machine learning\Arch technologhy\YOLO11\orignal_dataset_polygon_to_bounding_box\train\labels"
images_path = r"D:\Machine learning\Arch technologhy\YOLO11\orignal_dataset_polygon_to_bounding_box\train\images"

process_labels_folder(labels_path, images_path)
