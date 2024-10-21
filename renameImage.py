import os
import shutil

# Define the directories
source_folder = 'C:/Paul/VSPython/dataset_frames/images/val'  # Source folder
destination_folder = 'C:/Paul/VSPython/dataset_frames/images/trialonly'  # Destination folder

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Set the starting frame number
starting_frame = 364

# Get a sorted list of image filenames from the source folder
image_files = sorted(os.listdir(source_folder))  # Sorting ensures proper order

# Iterate through the sorted filenames and rename starting from frame364
for count, filename in enumerate(image_files, start=starting_frame):  # Start counting from frame364
    # Get the full path of the source image
    src = os.path.join(source_folder, filename)
    
    # Generate a new filename in the format 'frame' + count (e.g., frame364, frame365, ...)
    new_filename = f"frame{count}.jpg"  # Assuming the images are in .jpg format
    dest = os.path.join(destination_folder, new_filename)
    
    # Copy and rename the image
    shutil.copy(src, dest)

print("Images have been renamed and moved successfully!")
