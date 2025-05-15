# import os

# import cv2
# import numpy as np
# from ultralytics import YOLO


# def augment_image(img):
#     augmented_images = []

#     # Flip and rotate
#     imgFH = cv2.flip(img, 1)
#     augmented_images.append(imgFH)

#     imgFV = cv2.flip(img, 0)
#     augmented_images.append(imgFV)

#     imgRFH = cv2.rotate(imgFH, cv2.ROTATE_90_CLOCKWISE)
#     augmented_images.append(imgRFH)

#     imgRFV = cv2.rotate(imgFV, cv2.ROTATE_90_CLOCKWISE)
#     augmented_images.append(imgRFV)

#     # Get image dimensions
#     (height, width) = img.shape[:2]

#     # Translations
#     # Use a smaller translation factor
#     tx = int(0.2 * width)
#     ty = int(0.2 * height)

#     translations = [
#         (tx, -ty),
#         (tx, ty),
#         (-tx, -ty),
#         (-tx, ty),
#     ]

#     base_variants = [imgFH, imgFV]
#     for base_img in base_variants:
#         for tx, ty in translations:
#             M = np.float32([[1, 0, tx], [0, 1, ty]])
#             translated_img = cv2.warpAffine(base_img, M, (width, height))
#             augmented_images.append(translated_img)

#     return augmented_images


# def save_augmented_data(image_path, output_img_dir, output_lbl_dir, model):
#     filename = os.path.basename(image_path)
#     name, _ = os.path.splitext(filename)

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"[❌] Failed to load image: {image_path}")
#         return

#     augmented_images = augment_image(img)

#     for i, aug_img in enumerate(augmented_images):
#         img_filename = f"{name}_aug_{i}.jpg"
#         img_output_path = os.path.join(output_img_dir, img_filename)
#         cv2.imwrite(img_output_path, aug_img)

#         results = model(aug_img)
#         boxes = results[0].boxes

#         label_filename = f"{name}_aug_{i}.txt"
#         label_output_path = os.path.join(output_lbl_dir, label_filename)

#         with open(label_output_path, "w") as f:
#             if boxes is not None:
#                 for box in boxes.data.tolist():
#                     cls_id, x1, y1, x2, y2, conf = (
#                         int(box[5]),
#                         box[0],
#                         box[1],
#                         box[2],
#                         box[3],
#                         box[4],
#                     )
#                     x_center = ((x1 + x2) / 2) / aug_img.shape[1]
#                     y_center = ((y1 + y2) / 2) / aug_img.shape[0]
#                     w = (x2 - x1) / aug_img.shape[1]
#                     h = (y2 - y1) / aug_img.shape[0]
#                     if (
#                         0 <= x_center <= 1
#                         and 0 <= y_center <= 1
#                         and 0 <= w <= 1
#                         and 0 <= h <= 1
#                     ):
#                         f.write(
#                             f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
#                         )

#         # Final check: warn if label file is empty
#         if os.path.getsize(label_output_path) == 0:
#             print(f"[⚠️] No labels written for: {img_filename}")
#         else:
#             print(f"[✅] Saved {img_filename} and {label_filename}")


# # --- CONFIGURE THESE PATHS ---
# image_path = "/content/train/images/"
# output_img_dir = "/content/augmented_images/"
# output_lbl_dir = "/content/augmented_labels/"

# os.makedirs(output_img_dir, exist_ok=True)
# os.makedirs(output_lbl_dir, exist_ok=True)

# model = YOLO("yolo11n.pt")

# if __name__ == "__main__":
#     image_files = [
#         f
#         for f in os.listdir(image_path)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]
#     print(f"[INFO] Found {len(image_files)} images to process.")

#     for filename in image_files:
#         full_image_path = os.path.join(image_path, filename)
#         save_augmented_data(full_image_path, output_img_dir, output_lbl_dir, model)

#     # Final verification
#     total_augmented = len(os.listdir(output_img_dir))
#     total_labels = len(os.listdir(output_lbl_dir))
#     print(f"\n[SUMMARY] Augmented Images: {total_augmented} | Labels: {total_labels}")
#     if total_augmented != total_labels:
#         print(
#             "[❗] Warning: Some augmented images may not have corresponding label files."
#         )
#     else:
#         print("[✅] All augmented images have matching labels.")


import os

import cv2
import numpy as np
from ultralytics import YOLO


def augment_image(img):
    augmented_images = []

    # Flip and rotate
    imgFH = cv2.flip(img, 1)
    augmented_images.append(imgFH)

    imgFV = cv2.flip(img, 0)
    augmented_images.append(imgFV)

    imgRFH = cv2.rotate(imgFH, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(imgRFH)

    imgRFV = cv2.rotate(imgFV, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(imgRFV)

    # Get image dimensions
    (height, width) = img.shape[:2]

    # Translations
    # Use a smaller translation factor
    tx = int(0.2 * width)
    ty = int(0.2 * height)

    translations = [
        (tx, -ty),
        (tx, ty),
        (-tx, -ty),
        (-tx, ty),
    ]

    base_variants = [imgFH, imgFV]
    for base_img in base_variants:
        for tx, ty in translations:
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_img = cv2.warpAffine(base_img, M, (width, height))
            augmented_images.append(translated_img)

    return augmented_images


def save_augmented_data(image_path, output_img_dir, output_lbl_dir, model):
    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[❌] Failed to load image: {image_path}")
        return

    augmented_images = augment_image(img)

    for i, aug_img in enumerate(augmented_images):
        img_filename = f"{name}_aug_{i}.jpg"
        img_output_path = os.path.join(output_img_dir, img_filename)
        cv2.imwrite(img_output_path, aug_img)

        results = model(aug_img)
        boxes = results[0].boxes

        label_filename = f"{name}_aug_{i}.txt"
        label_output_path = os.path.join(output_lbl_dir, label_filename)

        with open(label_output_path, "w") as f:
            if boxes is not None:
                for box in boxes.data.tolist():
                    cls_id, x1, y1, x2, y2, conf = (
                        int(box[5]),
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        box[4],
                    )
                    x_center = ((x1 + x2) / 2) / aug_img.shape[1]
                    y_center = ((y1 + y2) / 2) / aug_img.shape[0]
                    w = (x2 - x1) / aug_img.shape[1]
                    h = (y2 - y1) / aug_img.shape[0]
                    if (
                        0 <= x_center <= 1
                        and 0 <= y_center <= 1
                        and 0 <= w <= 1
                        and 0 <= h <= 1
                    ):
                        f.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                        )

        # Final check: warn if label file is empty
        if os.path.getsize(label_output_path) == 0:
            print(f"[⚠️] No labels written for: {img_filename}")
        else:
            print(f"[✅] Saved {img_filename} and {label_filename}")


# --- CONFIGURE THESE PATHS ---
image_path = "/content/train/images/"
output_img_dir = "/content/augmented_images/"
output_lbl_dir = "/content/augmented_labels/"
full_image_path = (
    "/content/train/images/glioma_7_jpg.rf.97275e860ec20218c57eac34d50b0a71.jpg"
)

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

model = YOLO("/content/best.pt")

if __name__ == "__main__":
    image_files = [
        f
        for f in os.listdir(image_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"[INFO] Found {len(image_files)} images to process.")

    save_augmented_data(full_image_path, output_img_dir, output_lbl_dir, model)

    # Final verification
    total_augmented = len(os.listdir(output_img_dir))
    total_labels = len(os.listdir(output_lbl_dir))
    print(f"\n[SUMMARY] Augmented Images: {total_augmented} | Labels: {total_labels}")
    if total_augmented != total_labels:
        print(
            "[❗] Warning: Some augmented images may not have corresponding label files."
        )
    else:
        print("[✅] All augmented images have matching labels.")
