use image::{io::Reader as ImageReader, DynamicImage, ImageError};
use minifb::{Key, Window, WindowOptions};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

const WIDTH: usize = 800;
const HEIGHT: usize = 600;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = "D:/RUST_PROJECT/Rust_training/images_vache";
    let output_dir = "D:/Bureau/tout/Licence/Image darriver";
    let mut window = Window::new(
        "Image Viewer - Press ESC to exit, S to save, Left/Right to navigate",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )?;

    let mut image_paths: Vec<PathBuf> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();

    let mut current_index = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_down(Key::Right) {
            current_index = (current_index + 1) % image_paths.len();
        }
        if window.is_key_down(Key::Left) {
            current_index = if current_index == 0 {
                image_paths.len() - 1
            } else {
                current_index - 1
            };
        }
        if let Some(path) = image_paths.get(current_index) {
            let img = ImageReader::open(path)?.decode()?;
            let img = img.resize_exact(WIDTH as u32, HEIGHT as u32, image::imageops::FilterType::Nearest);
            let buffer = convert_image_to_u32_buffer(&img)?;

            window.update_with_buffer(&buffer, WIDTH, HEIGHT)?;
        }
        if window.is_key_down(Key::S) {
            let path = &image_paths[current_index];
            let file_name = path.file_name().unwrap().to_str().unwrap();
            let dest_path = format!("{}/{}", output_dir, file_name);
            fs::copy(path, dest_path)?;
        }
    }

    Ok(())
}

fn convert_image_to_u32_buffer(img: &DynamicImage) -> Result<Vec<u32>, ImageError> {
    let rgb_image = img.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    let raw_u8_buffer = rgb_image.into_raw();
    let buffer_u32 = raw_u8_buffer.chunks(3).map(|chunk| {
        let (r, g, b) = (chunk[0], chunk[1], chunk[2]);
        ((255 << 24) | (r as u32) << 16 | (g as u32) << 8 | (b as u32)) as u32
    }).collect();
    Ok(buffer_u32)
}
