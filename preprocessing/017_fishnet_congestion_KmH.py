from PIL import Image
import numpy as np
import pandas as pd

cmd = 'C:/Users/jseong/real-time_Crash/'

congestion = Image.open(cmd + '011_fishnet_connect_10min_congestion.png')
cg_array = np.array(congestion, dtype = np.float16)
road_length = pd.read_csv(cmd + '016_fishnet_road_length_km.csv')
road_length_km = np.array(road_length['length(km)'].tolist()).reshape(8308, 1)

cg_array2 = ((cg_array / 4) * road_length_km) * 6

img_cg = Image.fromarray(cg_array2)
img_cg.save(cmd + '017_fishnet_congestion_KmH.tiff')