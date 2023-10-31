import os

import cv2
from skimage.metrics import structural_similarity as compare_ssim


def is_frame_clear(
    frame,
    reference_frame,
    ssim_threshold,
    motion_threshold,
    color_hist_threshold,
    bg_diff_threshold,
    edge_threshold,
):
    """
    Determines if a frame is clear based on specified criteria.

    Args:
        frame: The current frame to be evaluated.
        reference_frame: The reference frame to compare against.
        ssim_threshold: The threshold for structural similarity index (SSIM).
        motion_threshold: The threshold for motion difference.
        color_hist_threshold: The threshold for color histogram difference.
        bg_diff_threshold: The threshold for background subtraction difference.
        edge_threshold: The threshold for edge detection.

    Returns:
        True if the frame is clear based on the specified criteria, False otherwise.
    """

    # Calculate SSIM between the current frame and the reference frame
    ssim = compare_ssim(
        cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
    )

    # Calculate absolute difference between the current frame and the reference frame
    motion = cv2.absdiff(reference_frame, frame)

    # Calculate color histogram difference
    reference_hist = cv2.calcHist(
        [cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)],
        [0],
        None,
        [256],
        [0, 256],
    )
    current_hist = cv2.calcHist(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]
    )
    hist_diff = cv2.compareHist(reference_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA)

    # Calculate background subtraction difference
    bg_diff = cv2.absdiff(reference_frame, frame)

    # Calculate edges in the current frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 50, 150)

    # Check if the frame is clear based on specified criteria
    return bool(
        (
            ssim > ssim_threshold
            or (motion > motion_threshold).any()
            or hist_diff > color_hist_threshold
            or (bg_diff > bg_diff_threshold).any()
            or (edges > edge_threshold).any()
        )
    )


def extract_clear_frames(
    video_path,
    output_folder,
    ssim_threshold,
    motion_threshold,
    color_hist_threshold,
    bg_diff_threshold,
    edge_threshold,
):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    reference_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Set the reference frame initially
        if reference_frame is None:
            reference_frame = frame.copy()

        # Check if the frame is clear based on criteria
        if is_frame_clear(
            frame,
            reference_frame,
            ssim_threshold,
            motion_threshold,
            color_hist_threshold,
            bg_diff_threshold,
            edge_threshold,
        ):
            output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)

    cap.release()


if __name__ == "__main__":
    video_path = "video.mp4"
    output_folder = "frames_output"
    ssim_threshold = 0.9
    motion_threshold = 1000
    color_hist_threshold = 0.2
    bg_diff_threshold = 50
    edge_threshold = 50

    extract_clear_frames(
        video_path,
        output_folder,
        ssim_threshold,
        motion_threshold,
        color_hist_threshold,
        bg_diff_threshold,
        edge_threshold,
    )
