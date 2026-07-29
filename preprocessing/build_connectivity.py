import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

# Load the fishnet and road GeoDataFrames
fishnet = gpd.read_file("C:/Users/jseong/real-time_Crash/009_degrade_crop_fishnet.shp")
roads = gpd.read_file("C:/Users/jseong/real-time_Crash/000_studyArea2/filtered_edges.shp").reset_index()

# Ensure the CRS is the same for both GeoDataFrames
roads = roads.to_crs(fishnet.crs)

# Add a unique pixel ID to each cell in the fishnet
fishnet['pixel_id'] = range(len(fishnet))

# Spatial join to find intersections between the roads and fishnet cells
intersections = gpd.sjoin(fishnet, roads, how='inner', predicate='intersects')

def sort_pixels_by_road(road, pixels, reverse=False):
    road_coords = list(road.coords)
    if reverse:
        road_coords = road_coords[::-1]  # 역방향 정렬을 위해 리스트 반전
    
    pixels["nearest_idx"] = pixels.geometry.apply(
        lambda p: min(range(len(road_coords)), key=lambda i: p.distance(Point(road_coords[i])))
    )
    
    return pixels.sort_values(by="nearest_idx")



connectivity = []
for idx, road in roads.iterrows():
    road_pixels = intersections[intersections["index"] == idx]
    sorted_pixels = sort_pixels_by_road(road.geometry, road_pixels)
    pixels = sorted_pixels['pixel_id'].tolist()

    # Create (source, target) pairs
    for i in range(len(pixels) - 1):
        source = pixels[i]
        target = pixels[i + 1]
        connectivity.append((source, target))

    if road['oneway'] == 0:
        road_pixels = intersections[intersections["index"] == idx]
        sorted_pixels = sort_pixels_by_road(road.geometry, road_pixels, reverse=True)
        pixels = sorted_pixels['pixel_id'].tolist()

        # Create (source, target) pairs
        for i in range(len(pixels) - 1):
            source = pixels[i]
            target = pixels[i + 1]
            connectivity.append((source, target))



# Create DataFrame to store the connectivity
df_connectivity = pd.DataFrame(connectivity, columns=['source', 'target'])

# Drop duplicates if necessary
df_connectivity = df_connectivity.drop_duplicates()

# Save to a CSV file if needed
df_connectivity.to_csv("C:/Users/jseong/real-time_Crash/010_fishnet_connectivity.csv", index=False)

connect_fishnet = fishnet[(fishnet['pixel_id'].isin(df_connectivity['source'].tolist())) | (fishnet['pixel_id'].isin(df_connectivity['target'].tolist()))]
connect_fishnet.to_file('C:/Users/jseong/real-time_Crash/010_fishnet_connectivity.shp')
