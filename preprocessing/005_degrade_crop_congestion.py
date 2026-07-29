### Use "general_env"

import numpy as np
from PIL import Image
from skimage.measure import block_reduce
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

def load_image(file_path):
    """Load an image from file and convert it to a numpy array."""
    with Image.open(file_path) as img:
        return np.array(img, dtype = np.int32)

def save_image(array, file_path):
    """Save a numpy array as an image file."""
    array = array.astype(np.int32)
    img = Image.fromarray(array)
    img.save(file_path, format = "TIFF")

def sum_reduce(image_array, block_size):
    """Reduce the image array by summing blocks of the given size."""
    return block_reduce(image_array, block_size=block_size, func=np.max)

def process_image(file_path, output_dir, block_size):
    """Process a single image: load, reduce, and save."""
    try:
        image_array = load_image(file_path)
        reduced_array = sum_reduce(image_array, block_size)
        output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.png', '.tiff'))
        save_image(reduced_array, output_file_path)
        return f"Processed {file_path}"
    except Exception as e:
        return f"Error processing {file_path}: {e}"

def process_images(input_dir, output_dir, block_size=(13, 13), num_workers=30):
    """Process all images in the input directory using multiple cores."""
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # List all image files in the input directory
    image_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)
                   if filename.endswith(".png")]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = [executor.submit(process_image, file_path, output_dir, block_size) for file_path in image_files]
        
        # Print results as they complete
        for future in results:
            result = future.result()



if __name__ == '__main__':
    freeze_support()

    input_directory = "C:/Users/LSY/Graduation Paper/004_crop_congestion_label/"
    output_directory = "C:/Users/LSY/st_service_area/000_500m_degraded_congestion/"
    process_images(input_directory, output_directory)

