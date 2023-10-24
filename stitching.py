import argparse
import logging
import concurrent.futures

import cv2

from image_stitching import ImageStitcher
from image_stitching import load_frames
from image_stitching import display

DOC = """This script lets us stich images together and display or save the results"""


def parse_args():
    """parses the command line arguments"""
    parser = argparse.ArgumentParser(description=DOC)
    parser.add_argument(
        "paths", type=str, nargs="+", help="paths to images, directories, or videos"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")

    parser.add_argument("--display", action="store_true", help="display result")
    parser.add_argument("--save", action="store_true", help="save result to file")
    parser.add_argument(
        "--save-path", default="stitched.png", type=str, help="path to save result"
    )

    return parser.parse_args()


def process_image(stitcher, frame):
    stitcher.add_image(frame)
    return stitcher.image()


result = None
if __name__ == "__main__":
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    stitcher = ImageStitcher()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        frames = load_frames(args.paths)
        results = list(
            executor.map(lambda frame: process_image(stitcher, frame), frames)
        )

    if args.display:
        for idx, result in enumerate(results):
            logging.info(f"displaying image {idx}")
            display("result", result)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    if args.save:
        logging.info(f"saving final image to {args.save_path}")
        cv2.imwrite(args.save_path, results[-1])

# if args.save:
#     logging.info(f"saving final image to {args.save_path}")
#     cv2.imwrite(args.save_path, result)
