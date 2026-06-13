import os
import cv2

def resize_and_rename_images(folder_path, size=(224, 224)):
    # Get all image filenames
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()  # Optional: sort for consistency

    for index, filename in enumerate(files):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Skipping unreadable file: {filename}")
            continue

        # Resize
        resized_img = cv2.resize(img, size)

        # New name: 1.jpg, 2.jpg, ...
        new_name = f"{index+1}.jpg"
        new_path = os.path.join(folder_path, new_name)

        # Save resized image
        cv2.imwrite(new_path, resized_img)

        # Remove original if name is different
        if filename != new_name:
            os.remove(img_path)

    print(f"✅ Done resizing and renaming in: {folder_path}")

# === Run for both folders ===
# resize_and_rename_images("dataset/handwritten")
# resize_and_rename_images("dataset/docs")
# resize_and_rename_images("dataset/onlinetest")
