import os
import cv2
import pandas as pd

# Function to resize and extract pixel values
def process_image(image_path):
    if (image_path[-18:]==".ipynb_checkpoints"):
      return
    image = cv2.imread(image_path)
    print(image_path)
    resized_image = cv2.resize(image, (28, 28))
    # Extract R, G, B channels
    b, g, r = cv2.split(resized_image)
    return r, g, b

# Folder containing the subfolders with images
root_folder = '/content/HAM10000_Class_segregated'
# Initialize lists to store data
data = []
image_names = []
folder_names = []

# Loop through each subfolder
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)

    # Loop through images in each subfolder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Process the image and get pixel values
        if(image_path[-18:]==".ipynb_checkpoints"):
          continue
        r, g, b = process_image(image_path)

        # Flatten pixel values and add to data list
        pixel_values = list(r.flatten()) + list(g.flatten()) + list(b.flatten())
        data.append(pixel_values)

        # Add image name and folder name
        image_names.append(image_name)
        folder_names.append(folder_name)

# Create a DataFrame from the collected data
df = pd.DataFrame(data, columns=[f"Pixel_{i}" for i in range(28*28*3)])
df['Image_Name'] = image_names
df['class'] = folder_names

# Save the DataFrame to a CSV file
df.to_csv('/content/drive/MyDrive/image_data_final.csv', index=False)