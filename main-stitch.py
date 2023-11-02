import os

import cv2

# Target folder
folder = "outside_images/"

# Output folder for stitched images
output_folder = "stitched/"
os.makedirs(output_folder, exist_ok=True)

# Load existing stitched images and determine the next available number
existing_files = os.listdir(output_folder)
existing_indices = [
    int(filename.split("_")[1].split(".")[0]) for filename in existing_files
]
next_index = max(existing_indices) + 1 if existing_indices else 1

# Load images
filenames = os.listdir(folder)
images = []
for file in filenames:
    # Get image
    img = cv2.imread(os.path.join(folder, file))

    # Save
    images.append(img)

# Use the built-in stitcher
stitcher = cv2.Stitcher.create()
(status, stitched) = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    # Convert the stitched image to grayscale
    gray_stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    # Find the coordinates of the non-black pixels
    coords = cv2.findNonZero(gray_stitched)

    # Get the bounding box of the non-black region
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the stitched image to remove the black edges
    cropped_stitched = stitched[y : y + h, x : x + w]

    # Normalize the image
    normalized_stitched = cv2.normalize(cropped_stitched, None, 0, 255, cv2.NORM_MINMAX)

    # Save the normalized stitched image with the next available number
    output_filename = os.path.join(output_folder, f"stitched_{next_index}.png")
    cv2.imwrite(output_filename, normalized_stitched)
    print(f"Stitched image saved as {output_filename}")
else:
    print("Image stitching failed")

cv2.waitKey(0)
