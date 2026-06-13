import os

test_dir = "test_images"  # Change this path if your test images are somewhere else

# List all image files
image_extensions = (".jpg", ".jpeg", ".png")
images = [f for f in os.listdir(test_dir) if f.lower().endswith(image_extensions)]

# Sort filenames to keep consistent order (optional)
images.sort()

# Rename images sequentially
for idx, filename in enumerate(images, start=1):
    ext = os.path.splitext(filename)[1].lower()  # Keep original extension
    new_name = f"{idx}{ext}"
    old_path = os.path.join(test_dir, filename)
    new_path = os.path.join(test_dir, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print(f"✅ Renamed {len(images)} images sequentially in '{test_dir}'.")
