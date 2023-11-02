/**
 * Copyright (c) 2023 Mayur Sinalkar
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */



 extern crate opencv;

use opencv::{
    core, highgui, imgcodecs, imgproc, prelude::*, videoio,
};

fn extract_clear_frames(
    video_path: &str,
    output_folder: &str,
    ssim_threshold: f64,
    ref_frame_interval: i32,
) -> Result<(), opencv::Error> {
    // Create the output folder
    std::fs::create_dir_all(output_folder)?;

    // Open the video file
    let mut cap = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        println!("Error: Could not open video file.");
        return Ok(());
    }

    let mut frame_count = 0;
    let mut reference_frame = Mat::default()?;
    let mut ref_frame_count = 0;

    loop {
        let mut frame = Mat::default()?;
        cap.read(&mut frame)?;

        if frame.empty()? {
            break;
        }

        frame_count += 1;

        if ref_frame_count == 0 {
            reference_frame = Mat::clone(&frame)?;
            ref_frame_count = ref_frame_interval;
        }

        // Method 1: Calculate SSIM between the current frame and the reference frame
        let gray_reference_frame = Mat::default()?;
        let gray_frame = Mat::default()?;
        imgproc::cvt_color(
            &reference_frame,
            &mut gray_reference_frame,
            imgproc::COLOR_BGR2GRAY,
            0,
        )?;
        imgproc::cvt_color(&frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;
        let ssim = imgproc::compare_ssim(&gray_reference_frame, &gray_frame, None)?;

        // Method 2: Detect motion using frame difference
        let mut motion = Mat::default()?;
        core::absdiff(&reference_frame, &frame, &mut motion)?;
        let motion_threshold = 1000.0; // Adjust this threshold as needed
        let has_motion = core::count_non_zero(&motion)? > motion_threshold;

        // Method 3: Keyframe selection (based on frame count)
        let is_keyframe = frame_count % 20 == 0; // Adjust the keyframe interval as needed

        // Method 4: Color histogram analysis (compare histograms)
        let mut reference_hist = Mat::default()?;
        let mut current_hist = Mat::default()?;
        imgproc::calc_hist(
            &[&gray_reference_frame],
            &mut [0],
            core::no_array()?,
            &mut reference_hist,
            1,
            &[256],
            &[0.0, 256.0],
        )?;
        imgproc::calc_hist(
            &[&gray_frame],
            &mut [0],
            core::no_array()?,
            &mut current_hist,
            1,
            &[256],
            &[0.0, 256.0],
        )?;
        let hist_diff = imgproc::compare_hist(
            &reference_hist,
            &current_hist,
            imgproc::HISTCMP_BHATTACHARYYA,
        )?;
        let color_hist_threshold = 0.2; // Adjust this threshold as needed

        // Method 5: Background subtraction (simple difference)
        let mut bg_diff = Mat::default()?;
        core::absdiff(&reference_frame, &frame, &mut bg_diff)?;
        let bg_diff_threshold = 50.0; // Adjust this threshold as needed
        let has_bg_diff = core::count_non_zero(&bg_diff)? > bg_diff_threshold;

        // Method 6: Edge detection
        let mut gray_frame = Mat::default()?;
        imgproc::cvt_color(&frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;
        let mut edges = Mat::default()?;
        imgproc::canny(&gray_frame, &mut edges, 50.0, 150.0, 3, false)?;
        let has_edges = core::count_non_zero(&edges)? > 0;

        // Save the frame if it meets any of the criteria
        if ssim > ssim_threshold
            || has_motion
            || is_keyframe
            || hist_diff > color_hist_threshold
            || has_bg_diff
            || has_edges
        {
            let output_path = format!("{}/frame_{:04}.jpg", output_folder, frame_count);
            imgcodecs::imwrite(&output_path, &frame, &core::Vector::<i32>::new())?;
        }

        ref_frame_count -= 1;
    }

    Ok(())
}

fn main() {
    let video_path = "path/to/your/video.mp4";
    let output_folder = "output_frames";
    let ssim_threshold = 0.9;
    let ref_frame_interval = 10;

    match extract_clear_frames(video_path, output_folder, ssim_threshold, ref_frame_interval) {
        Ok(_) => println!("Frames extracted successfully!"),
        Err(err) => eprintln!("Error: {:?}", err),
    }
}
