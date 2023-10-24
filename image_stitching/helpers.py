import logging
import pathlib
from typing import Generator, List

import cv2
import numpy

DOC = """helper functions for loading frames and displaying them"""


def display(title, img, max_size=500000):
    """
    resizes the image before it displays it,
    this stops large stitches from going over the screen!
    """
    assert isinstance(img, numpy.ndarray), "img must be a numpy array"
    assert isinstance(title, str), "title must be a string"
    scale = numpy.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)


def save_frames_as_images(video_path: pathlib.Path, output_directory: pathlib.Path):
    """Save frames from a video as images."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_filename = output_directory / f"frame{frame_count:04d}.png"
        cv2.imwrite(str(frame_filename), frame)

        frame_count += 1

    cap.release()


def load_frames(image_directory: pathlib.Path):
    """Load saved frames from images and yield them one by one."""
    image_files = sorted(
        image_directory.glob("*.png")
    )  # Adjust the file extension as needed

    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        if frame is not None:
            yield frame
