import geopandas as gpd
from PIL import Image
import glob
import numpy as np
from tqdm.auto import tqdm

fishnet = gpd.read_file('C:/Users/jseong/real-time_Crash/010_fishnet_connectivity.shp')
id = fishnet['pixel_id'].tolist()

files = glob.glob('C:/Users/jseong/real-time_Crash/005_degrade_crop_congestion/*.tiff')

image = Image.open(files[0])
image_array = np.array(image)

img_reshape = image_array.reshape(278 * 409, 1)

img_mask = img_reshape[id]

for file in tqdm(files[1:]):
    image = Image.open(file)
    image_array = np.array(image)

    img_reshape = image_array.reshape(278 * 409, 1)

    img_ = img_reshape[id]
    img_mask = np.concatenate((img_mask, img_), axis = 1)


img_mask.shape
img_mask_ = Image.fromarray(img_mask)
img_mask_.save('C:/Users/jseong/real-time_Crash/011_fishnet_connect_10min_congestion.png')
