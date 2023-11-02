import os

import cv2
import numpy as np


def stitch_images(images):
    # Initialize feature detector (SIFT)
    detector = cv2.SIFT_create()

    # Initialize feature matcher (Flann)
    matcher = cv2.FlannBasedMatcher_create()

    # Initialize the result with the first image
    result = images[0]

    # Initialize feature detector (SIFT) for the first image
    gray_first = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    keypoints_first, descriptors_first = detector.detectAndCompute(gray_first, None)
    descriptors = [descriptors_first]
    locations = [keypoints_first]

    for i in range(1, len(images)):
        # Initialize feature detector (SIFT) for the current image
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        keypoints, descriptor = detector.detectAndCompute(gray, None)

        # Match the descriptors with the previous image
        matches = matcher.knnMatch(descriptor, descriptors[-1], k=2)

        # Apply ratio test to select good matches
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # Get keypoint locations for good matches
        points1 = np.float32([locations[-1][m.trainIdx].pt for m in good_matches])
        points2 = np.float32([keypoints[m.queryIdx].pt for m in good_matches])

        # Find the perspective transform (homography)
        H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        # Warp the current image individually
        warped = cv2.warpPerspective(images[i], H, (result.shape[1], result.shape[0]))

        # Ensure the result has the same height as the warped image
        if result.shape[0] < warped.shape[0]:
            result = cv2.vconcat(
                [
                    result,
                    np.zeros(
                        (
                            warped.shape[0] - result.shape[0],
                            result.shape[1],
                            result.shape[2],
                        ),
                        dtype=np.uint8,
                    ),
                ]
            )

        # Combine the result with the warped image
        result = cv2.hconcat([result, warped])

        # Add the descriptor to the list
        descriptors.append(descriptor)
        locations.append(keypoints)

    return result


def process_images_in_batches(folder_path, batch_size, output_folder):
    # List all image files in the folder
    image_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".jpg")
    ]

    # Process images one at a time and create the batch panoramas
    current_batch = []
    batch_panoramas = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        current_batch.append(image)

        if len(current_batch) >= batch_size:
            batch_panorama = stitch_images(current_batch)
            batch_panoramas.append(batch_panorama)
            current_batch = []

    # Process the remaining images in the last batch
    if current_batch:
        batch_panorama = stitch_images(current_batch)
        batch_panoramas.append(batch_panorama)

    # Save batch panoramas to the output folder
    os.makedirs(output_folder, exist_ok=True)
    for i, panorama in enumerate(batch_panoramas):
        output_path = os.path.join(output_folder, f"panorama_batch_{i}.jpg")
        cv2.imwrite(output_path, panorama)

    return batch_panoramas


def stitch_batch_panoramas(batch_panoramas, output_folder):
    # Combine the batched panoramas into a final panorama
    final_panorama = stitch_images(batch_panoramas)

    # Save the final panorama
    output_path = os.path.join(output_folder, "final_panorama.jpg")
    cv2.imwrite(output_path, final_panorama)


# Define the folder containing the images
folder_path = "frames_output"

# Define the batch size (you can adjust this based on available memory)
batch_size = 25

# Define the output folder for batch panoramas
output_folder = "batch_panoramas"

# Process images in batches and create the batch panoramas
batch_panoramas = process_images_in_batches(folder_path, batch_size, output_folder)

# Stitch the batch panoramas to create the final panorama
stitch_batch_panoramas(batch_panoramas, output_folder)
