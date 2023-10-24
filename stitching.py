import argparse
import logging
import pathlib

import cv2

from image_stitching import ImageStitcher
from image_stitching.helpers import load_frames, save_frames_as_images


def parse_args():
    parser = argparse.ArgumentParser(description="Image Stitching")
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video",
    )
    parser.add_argument("--display", action="store_true", help="Display result")
    parser.add_argument("--save", action="store_true", help="Save result to file")
    parser.add_argument(
        "--save-path",
        default="panorama.png",
        type=str,
        help="Path to save result",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Create an output directory for saving frames as images
    image_output_dir = pathlib.Path("frames")
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Call the function to save frames as images
    save_frames_as_images(args.video_path, image_output_dir)

    stitcher = ImageStitcher()

    # Call the function to load and process saved frames
    for frame in load_frames()(image_output_dir):
        stitcher.add_image(frame)

    result = stitcher.image()

    if args.display:
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save:
        logging.info(f"Saving final image to {args.save_path}")
        cv2.imwrite(args.save_path, result)


if __name__ == "__main__":
    main()
