import os
from PIL import Image

# ===== CONFIGURATION ===== #
INPUT_DIR = "Data/Cardio/GEData/冠脉数据分类/冠脉数据分类/type1cLabelpredict"  # Folder with original images
OUTPUT_DIR = "Data/Cardio/GEData/冠脉数据分类/冠脉数据分类/labeled_train_images"  # Folder to save grayscale images

# Create output directory if it doesn’t exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert all images to grayscale
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        try:
            img = Image.open(input_path).convert("L")  # Convert to grayscale (L mode)
            img.save(output_path)
            print(f" Converted: {filename}")
        except Exception as e:
            print(f" Failed to convert {filename}: {e}")

print(" All images converted to grayscale and saved in:", OUTPUT_DIR)
