from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

cmd = 'C:/Users/LSY/Graduation Paper/'

congestion = Image.open(cmd + '017_fishnet_congestion_KmH.tiff')
cg_array = np.array(congestion)
cg_array = cg_array.reshape(8308, 4320, 1)

# # Nov 4: President Election, Nov 11: Veterans Day, Nov 27: Thanksgiving Day
# cg_array = np.concatenate((cg_array[:, :432], cg_array[:, 576:1440], cg_array[:, 1584:3744], cg_array[:, 3888:]), axis = 1)
# cg_array = cg_array.reshape(8308, 3888, 1)



cg_scaled = (cg_array - cg_array.mean()) / cg_array.std()
print(cg_array.mean(), cg_array.std())  # 0.029127512 0.2054107
cg_scaled = np.array(cg_scaled).reshape(8308, 4320, 1)
cg_array = cg_array.reshape(8308, 4320, 1)

crash = Image.open(cmd + 'crash+construction/105_crash_count.tiff')
cr_array = np.array(crash).reshape(8308, 4320, 1)

construction = Image.open(cmd + 'crash+construction/005_construction_01.png')
cs_array = np.array(construction).reshape(8308, 4320, 1)

precipitation = Image.open(cmd + 'precipitation/103_precipitation_max.png')
pr_array = np.array(precipitation).reshape(8308, 4320, 1)

road = pd.read_csv(cmd + '013_fishnet_road_property.csv')
road_array = np.array(road[['motorway', 'primary', 'trunk']])

svi = Image.open(cmd + '015_fishnet_SVI_density.tiff')
svi_array = np.array(svi)

# Normalization
con = []
for i in tqdm(range(4320)):
    con.append(np.concatenate([
        cg_scaled[:, i],
        cr_array[:, i],
        cs_array[:, i], 
        pr_array[:, i],
        road_array,
        svi_array], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_all1.npy', feature)

cg = cg_scaled.reshape(4320, 8308, 1)
np.save(cmd + '018_feature_scaled_cg_only.npy', cg)

con = []
for i in tqdm(range(4312)):
    con.append(
        cg_scaled[:, i:i+6].reshape(8308, 6)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_scaled_cg6_only.npy', feature)


# Non Normalization

cg = cg_array.reshape(4320, 8308, 1)
np.save(cmd + '018_feature_cg_only.npy', cg)

cg = cg_array.reshape(3888, 8308, 1)
np.save(cmd + '018_feature_cg_only_noholiday.npy', cg)

cr = cr_array.reshape(4320, 8308, 1)
np.save(cmd + '018_feature_cr_only.npy', cr)

cs = cs_array.reshape(4320, 8308, 1)
np.save(cmd + '018_feature_cs_only.npy', cs)

pr = pr_array.reshape(4320, 8308, 1)
np.save(cmd + '018_feature_pr_only.npy', pr)



con = []
for i in tqdm(range(4320)):
    con.append(np.concatenate([
        cg_array[:, i],
        road_array,
        svi_array], axis = 1)
    )
    
feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg.npy', feature)

con = []
for i in tqdm(range(4320)):
    con.append(np.concatenate([
        cg_array[:, i],
        cr_array[:, i],
        road_array,
        svi_array], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg+cr.npy', feature)

con = []
for i in tqdm(range(4320)):
    con.append(np.concatenate([
        cg_array[:, i],
        cr_array[:, i],
        cs_array[:, i], 
        road_array,
        svi_array], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg+cr+cs.npy', feature)

con = []
for i in tqdm(range(4320)):
    con.append(np.concatenate([
        cg_array[:, i],
        cr_array[:, i],
        cs_array[:, i], 
        pr_array[:, i],
        road_array,
        svi_array], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg+cr+cs+pr.npy', feature)

con = []
for i in tqdm(range(4312)):
    con.append(np.concatenate([
        cg_array[:, i:i+6].reshape(8308, 6),
        cr_array[:, i],
        cs_array[:, i], 
        pr_array[:, i],
        road_array,
        svi_array], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg+cr+cs+pr_6.npy', feature)


con = []
for i in tqdm(range(4312)):
    con.append(
        cg_array[:, i:i+6].reshape(8308, 6)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg6_only.npy', feature)


con = []
for i in tqdm(range(3880)):
    con.append(
        cg_array[:, i:i+6].reshape(8308, 6)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg6_only_noholiday.npy', feature)


con = []
for i in tqdm(range(4312)):
    con.append(np.concatenate([
        cg_array[:, i:i+6].reshape(8308, 6),
        road_array], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg6+road.npy', feature)


con = []
for i in tqdm(range(4312)):
    con.append(np.concatenate([
        cg_array[:, i:i+6].reshape(8308, 6),
        svi_array], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg6+svi.npy', feature)


con = []
for i in tqdm(range(4312)):
    con.append(np.concatenate([
        cg_array[:, i:i+6].reshape(8308, 6),
        road_array,
        svi_array], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg6+road+svi.npy', feature)


con = []
for i in tqdm(range(4312)):
    con.append(np.concatenate([
        cg_array[:, i:i+6].reshape(8308, 6),
        cr_array[:, i]], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg6+cr.npy', feature)


con = []
for i in tqdm(range(4312)):
    con.append(np.concatenate([
        cg_array[:, i:i+6].reshape(8308, 6),
        cr_array[:, i],
        cs_array[:, i]], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg6+cr+cs.npy', feature)


con = []
for i in tqdm(range(4312)):
    con.append(np.concatenate([
        cg_array[:, i:i+6].reshape(8308, 6),
        cr_array[:, i],
        cs_array[:, i], 
        pr_array[:, i]], axis = 1)
    )

feature = np.array(con)
feature.shape
np.save(cmd + '018_feature_cg6+cr+cs+pr.npy', feature)


# Create target data
target = []
for i in tqdm(range(4312)):
    target.append(cg_scaled[:, i+6])

target10 = np.array(target)
target10.shape

np.save(cmd + '018_target10.npy', target10)

target = []
for i in tqdm(range(4312)):
    target.append(cg_scaled[:, i+7])

target20 = np.array(target)
target20.shape

np.save(cmd + '018_target20.npy', target20)

target = []
for i in tqdm(range(4312)):
    target.append(cg_scaled[:, i+8])

target30 = np.array(target)
target30.shape

np.save(cmd + '018_target30.npy', target30)


target = []
for i in tqdm(range(4312)):
    target.append(cg_array[:, i+6])

target10 = np.array(target)
target10.shape

np.save(cmd + '018_target10_nnor.npy', target10)

target = []
for i in tqdm(range(4312)):
    target.append(cg_array[:, i+7])

target20 = np.array(target)
target20.shape

np.save(cmd + '018_target20_nnor.npy', target20)

target = []
for i in tqdm(range(4312)):
    target.append(cg_array[:, i+8])

target30 = np.array(target)
target30.shape

np.save(cmd + '018_target30_nnor.npy', target30)




target = []
for i in tqdm(range(3880)):
    target.append(cg_array[:, i+6])

target10 = np.array(target)
target10.shape

np.save(cmd + '018_target10_nnor_noholiday.npy', target10)

target = []
for i in tqdm(range(3880)):
    target.append(cg_array[:, i+7])

target20 = np.array(target)
target20.shape

np.save(cmd + '018_target20_nnor_noholiday.npy', target20)

target = []
for i in tqdm(range(3880)):
    target.append(cg_array[:, i+8])

target30 = np.array(target)
target30.shape

np.save(cmd + '018_target30_nnor_noholiday.npy', target30)