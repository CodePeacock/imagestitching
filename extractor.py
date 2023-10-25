"""
 Copyright (c) 2023 Mayur Sinalkar

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

# @title Extract All frames from video

import os

import cv2


def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(name=output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(filename=video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0

    # Initialize the GPU accelerated video reader
    cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(cuda_enabled)
    if cuda_enabled:
        print("GPU acceleration enabled.")
        gpu_frame_reader = cv2.cuda_BkgSubtractorMOG2.create()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Use GPU acceleration to process the frame
        if cuda_enabled:
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)
            fgmask_gpu = gpu_frame_reader.apply(frame_gpu)
            fgmask = fgmask_gpu.download()
        else:
            fgmask = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        # Save the frame as a .jpg image
        output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename=output_path, img=fgmask)

    # Release video capture and GPU resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video.mp4"
    output_folder = "frames_output"

    extract_frames(video_path, output_folder)
