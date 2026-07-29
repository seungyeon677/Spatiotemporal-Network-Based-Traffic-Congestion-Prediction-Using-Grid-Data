import glob
from PIL import Image
import numpy as np
import rasterio
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support


def get_img(image_path, type):
    with Image.open(image_path) as img:
        return np.array(img, dtype = type)


def checking_shape(image_path, base_array):
    img_array = get_img(image_path, np.int32)

    if img_array.shape != base_array.shape:
        print(f"Image {image_path} does not have matching shape")
        return None
    else:
        return img_array

def process_image(image_path, base_array):
    return checking_shape(image_path, base_array)

def sum_array(img_list):
    return np.sum(img_list, axis = 0)


def save_tif(output_dir, array, row, col):
    array = array.reshape(1, row, col)
    with rasterio.open(output_dir, 'w', dtype = np.int32,
                       height = array.shape[1], width = array.shape[2],
                       count = 1, driver = 'GTiff') as dst:
        dst.write(array, indexes = [1])


def process_pooling(base_array, image_files):
    with ProcessPoolExecutor() as executor:
        print("start processing image")
        futures = {executor.submit(checking_shape, image_path, base_array): image_path for image_path in tqdm(image_files)}
        print("finish processing image")

        img_list = []
        for future in as_completed(futures):
            img_list.append(future.result())

    return sum_array(img_list)


"""-------------------------------------------------------------------------------------------"""


if __name__ == "__main__":
    freeze_support()

    files = glob.glob('C:/Users/jseong/real-time_Crash/005_degrade_crop_congestion/*.tiff')
    output_dir = 'C:/Users/jseong/real-time_Crash/006_degrade_congestion_sum.tiff'

    base_array = get_img(files[0], np.int32)
    row, col = base_array.shape
    batch_size = 1000

    result = []
    for i in range(0, len(files), batch_size): # Using Batch to ignore Memory Error
        batch_files = files[i: i+batch_size]
        result_array = process_pooling(base_array, batch_files)
        result.append(result_array)
        print(f'batch_{i // batch_size + 1} completed.')


    final_result = sum_array(result)
    save_tif(output_dir, final_result, row, col)
