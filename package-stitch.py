"""
 Copyright (c) 2023 Mayur Sinalkar

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import os

import cv2
from stitching import Stitcher

folder = "sunny_phone_camera/"
stitcher = Stitcher(detector="orb")
# Load images
filenames = os.listdir(folder)
images = []
for file in filenames:
    # Get image
    img = cv2.imread(os.path.join(folder, file))

    # Save
    images.append(img)

panorama = stitcher.stitch(images)

cv2.imwrite("panorama.png", panorama)
