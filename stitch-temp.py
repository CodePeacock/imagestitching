import logging
import os
from multiprocessing import Pool

import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def list_image_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]


def stitch_and_save_batch(args):
    image_paths, output_folder, batch_index = args
    images = [cv2.imread(image_path) for image_path in image_paths]
    stitcher = cv2.Stitcher.create()  # Use the default OpenCV Stitcher

    success, panorama = stitcher.stitch(images)

    if success == cv2.Stitcher_OK:
        output_path = os.path.join(output_folder, f"panorama_batch_{batch_index}.jpg")
        cv2.imwrite(output_path, panorama)
        logging.info(f"Batch {batch_index} stitched and saved successfully.")
    else:
        logging.warning(
            f"Batch {batch_index} stitching failed with status code: {success}"
        )


def main():
    input_folder = "frames_output"
    output_folder = "panorama_output_work"
    batch_size = 50

    os.makedirs(output_folder, exist_ok=True)

    image_paths = list_image_files(input_folder)
    num_images = len(image_paths)
    num_batches = num_images // batch_size + 1

    # Use multiprocessing.Pool for parallel processing
    num_processors = min(8, os.cpu_count())
    pool = Pool(num_processors)

    for batch_index in range(num_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, num_images)
        batch_image_paths = image_paths[start:end]

        # Perform stitching using the pool
        pool.apply_async(
            stitch_and_save_batch, [(batch_image_paths, output_folder, batch_index)]
        )

    pool.close()
    pool.join()

    # Combine the batched panoramas into a final panorama
    batch_files = [
        os.path.join(output_folder, f"panorama_batch_{i}.jpg")
        for i in range(num_batches)
    ]

    # Use all available threads for final stitching
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(True)  # Enable OpenCL for GPU acceleration
    final_stitcher = cv2.Stitcher.create()
    success, final_panorama = final_stitcher.stitch(
        [cv2.imread(f) for f in batch_files]
    )

    if success == cv2.Stitcher_OK:
        final_output_path = os.path.join(output_folder, "final_panorama.jpg")
        cv2.imwrite(final_output_path, final_panorama)
        logging.info("Final panorama saved successfully.")
    else:
        logging.warning("Final stitching failed with status code:", success)


if __name__ == "__main__":
    main()
