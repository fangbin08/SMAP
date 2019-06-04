import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math

# # Load in variables
# f = h5py.File("parameters.hdf5", "r")
# varname_list = list(f.keys())
#
# for x in range(len(varname_list)):
#     var_obj = f[varname_list[x]][()]
#     exec(varname_list[x] + '= var_obj')
# f.close()

#########################################################################################################
# (Function 1) Define a function for generating EASE grid projection lat/lon tables from CATDS provided data

def ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col):
    global lat_ease_file, lat_ease_world, lon_ease_file, lon_ease_world
    lat_ease_file = open(path_ease_lat, 'r')
    lat_ease_world = np.fromfile(lat_ease_file, dtype=np.dtype('f8'))
    lat_ease_world = np.reshape(lat_ease_world, (num_row, num_col))
    lat_ease_world = lat_ease_world[:, 1].reshape(-1, 1)
    lat_ease_file.close()

    lon_ease_file = open(path_ease_lon, 'r')
    lon_ease_world = np.fromfile(lon_ease_file, dtype=np.dtype('f8'))
    lon_ease_world = np.reshape(lon_ease_world, (num_row, num_col))
    lon_ease_world = lon_ease_world[1, :].reshape(-1, 1)
    lon_ease_file.close()

    del lat_ease_file, lon_ease_file, path_ease_lat, path_ease_lon, num_row, num_col

    return lat_ease_world, lon_ease_world


#########################################################################################

# (Function 2) Convert latitude and longitude to the corresponding row and col in the
# EASE grid VERSION 2 used at CATDS since processor version 2.7, January 2014

def geo2easeGridV2(latitude, longitude, interdist, num_row, num_col):
    # Constant
    a = 6378137  # equatorial radius
    f = 1 / 298.257223563  # flattening
    b = 6356752.314  # polar radius b=a(1-f)
    e = 0.0818191908426  # eccentricity sqrt(2f-f^2)
    c = interdist  # interdistance pixel
    nl = num_row  # Number of lines
    nc = num_col  # Number of columns
    s0 = (nl - 1) / 2
    r0 = (nc - 1) / 2
    phi0 = 0
    lambda0 = 0  # map reference longitude
    phi1 = 30  # latitude true scale
    k0 = np.cos(np.deg2rad(phi1)) / np.sqrt(1 - (e ** 2 * np.sin(np.deg2rad(phi1)) ** 2))
    q = (1 - e ** 2) * ((np.sin(np.deg2rad(latitude)) / (1 - e ** 2 * np.sin(np.deg2rad(latitude)) ** 2)) -
                        (1 / (2 * e)) * np.log(
                (1 - e * np.sin(np.deg2rad(latitude))) / (1 + e * np.sin(np.deg2rad(latitude)))))
    x = a * k0 * (longitude - lambda0) * np.pi / 180
    y = a * q / (2 * k0)
    # as Brodzik et al
    column = np.round(r0 + (x / c))
    row = np.round(s0 - (y / c))

    del a, f, b, e, c, nl, nc, s0, r0, phi0, lambda0, phi1, k0, q, x, y

    return row, column

#########################################################################################

# (Function 3) Find and map the corresponding index numbers for the high spatial resolution
# row/col tables from the low spatial resolution row/col tables. The output is 1-dimensional
# array containing index numbers.

def find_easeind_hifrlo(lat_hires, lon_hires, interdist_lowres, num_row_lowres, num_col_lowres,
                        row_ind_lowres, col_ind_lowres):

    lon_meshgrid, lat_meshgrid = np.meshgrid(lon_hires, lat_hires)
    lat_meshgrid_array = lat_meshgrid.flatten()
    lon_meshgrid_array = lon_meshgrid.flatten()

    [row_ind_toresp, col_ind_toresp] = \
        geo2easeGridV2(lat_meshgrid_array, lon_meshgrid_array, interdist_lowres,
                       num_row_lowres, num_col_lowres)

    row_ind_toresp = np.reshape(row_ind_toresp, np.shape(lon_meshgrid))
    col_ind_toresp = np.reshape(col_ind_toresp, np.shape(lon_meshgrid))

    row_ind_toresp = row_ind_toresp[:, 1]
    col_ind_toresp = col_ind_toresp[1, :].reshape(-1, 1)

    # find the corresponding row/col indices for high resolution from low resolution lat/lon tables
    row_ind_diff = np.setdiff1d(row_ind_toresp, row_ind_lowres)
    if row_ind_diff.size != 0:
        for x in range(len(row_ind_diff)):
            row_dist = np.absolute(row_ind_diff[x] - row_ind_lowres)
            row_ind_toresp[np.where(row_ind_toresp == row_ind_diff[x])] = row_ind_lowres[np.argmin(row_dist)]
    else:
        pass

    row_ind_unique, row_ind_dest = np.unique(row_ind_toresp, return_inverse=True)
    row_ind_unique = row_ind_unique.reshape(-1, 1)
    row_ind_dest = row_ind_dest.reshape(-1, 1)

    col_ind_diff = np.setdiff1d(col_ind_toresp, col_ind_lowres)
    if col_ind_diff.size != 0:
        for x in range(len(col_ind_diff)):
            col_dist = np.absolute(col_ind_diff[x] - col_ind_lowres)
            col_ind_toresp[np.where(col_ind_toresp == col_ind_diff[x])] = col_ind_lowres[np.argmin(col_dist)]
    else:
        pass

    col_ind_unique, col_ind_dest = np.unique(col_ind_toresp, return_inverse=True)
    col_ind_unique = col_ind_unique.reshape(-1, 1)
    col_ind_dest = col_ind_dest.reshape(-1, 1)

    del lat_meshgrid, lon_meshgrid, lat_meshgrid_array, lon_meshgrid_array, row_ind_toresp, col_ind_toresp, \
        row_ind_unique, col_ind_unique, row_ind_diff, col_ind_diff

    return row_ind_dest, col_ind_dest

#########################################################################################
# Specify file paths
# Path of EASE projection lat/lon tables
# Use the data downloaded from ftp://sidads.colorado.edu/pub/tools/easegrid2/
path_ease_coord_table = '/Volumes/My Passport/SMAP_Project/Datasets/geolocation'
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of MODIS LST folder
path_modis_lst = '/Volumes/My Passport/SMAP_Project/Datasets/MODIS/MYD11A1/Reprojected'

# World extent corner coordinates
lat_geo_world_max = 90
lat_geo_world_min = -90
lon_geo_world_max = 180
lon_geo_world_min = -180

# CONUS extent corner coordinates
lat_geo_conus_max = 53
lat_geo_conus_min = 25
lon_geo_conus_max = -67
lon_geo_conus_min = -125

# Cellsize
cellsize_400m = 0.004
cellsize_1km = 0.01
cellsize_5km = 0.05
cellsize_9km = 0.09
cellsize_12_5km = 0.125
cellsize_25km = 0.25
cellsize_36km = 0.36

# Global extent of EASE grid projection (row/col)
size_ease_world_400m = np.array([36540, 86760])
size_ease_world_1km = np.array([14616, 34704])
size_ease_world_9km = np.array([1624, 3856])
size_ease_world_12_5km = np.array([1168, 2776])
size_ease_world_25km = np.array([584, 1388])
size_ease_world_36km = np.array([406, 964])

# Interdistance of EASE grid projection grids
interdist_ease_400m = 400.4041601296
interdist_ease_1km = 1000.89502334956
interdist_ease_9km = 9009.093602916
interdist_ease_12_5km = 12512.63000405
interdist_ease_25km = 25067.525
interdist_ease_36km = 36036.374411664



#######################################################################################
#######################################################################################
# 1. Generate lat/lon tables of EASE Grid projection in the world
os.chdir(path_ease_coord_table)

# EASE 1 km
path_ease_lat = 'EASE2_M01km.lats.34704x14616x1.double'
path_ease_lon = 'EASE2_M01km.lons.34704x14616x1.double'
num_row = size_ease_world_1km[0]
num_col = size_ease_world_1km[1]
[lat_ease_world_1km, lon_ease_world_1km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)

# EASE 9 km
path_ease_lat = 'EASE2_M09km.lats.3856x1624x1.double'
path_ease_lon = 'EASE2_M09km.lons.3856x1624x1.double'
num_row = size_ease_world_9km[0]
num_col = size_ease_world_9km[1]
[lat_ease_world_9km, lon_ease_world_9km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)

# EASE 12.5 km
path_ease_lat = 'EASE2_M12.5km.lats.2776x1168x1.double'
path_ease_lon = 'EASE2_M12.5km.lons.2776x1168x1.double'
num_row = size_ease_world_12_5km[0]
num_col = size_ease_world_12_5km[1]
[lat_ease_world_12_5km, lon_ease_world_12_5km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)

# EASE 25 km
path_ease_lat = 'EASE2_M25km.lats.1388x584x1.double'
path_ease_lon = 'EASE2_M25km.lons.1388x584x1.double'
num_row = size_ease_world_25km[0]
num_col = size_ease_world_25km[1]
[lat_ease_world_25km, lon_ease_world_25km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)

# EASE 36 km
path_ease_lat = 'EASE2_M36km.lats.964x406x1.double'
path_ease_lon = 'EASE2_M36km.lons.964x406x1.double'
num_row = size_ease_world_36km[0]
num_col = size_ease_world_36km[1]
[lat_ease_world_36km, lon_ease_world_36km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)


# Save variables
var_name = ["cellsize_400m", "cellsize_1km", "cellsize_5km", "cellsize_9km",
            "cellsize_12_5km", "cellsize_25km", "cellsize_36km", "interdist_ease_400m",
            "interdist_ease_1km", "interdist_ease_9km", "interdist_ease_12_5km",
            "interdist_ease_25km", "interdist_ease_36km", "lat_geo_conus_max",
            "lat_geo_conus_min", "lon_geo_conus_max", "lon_geo_conus_min",
            "lat_geo_world_max", "lat_geo_world_min", "lon_geo_world_max",
            "lon_geo_world_min", "lat_ease_world_1km", "lon_ease_world_1km",
            "lat_ease_world_9km", "lon_ease_world_9km", "lat_ease_world_12_5km",
            "lon_ease_world_12_5km", "lat_ease_world_25km", "lon_ease_world_25km",
            "lat_ease_world_36km", "lon_ease_world_36km", "size_ease_world_400m",
            "size_ease_world_1km", "size_ease_world_9km", "size_ease_world_12_5km",
            "size_ease_world_25km", "size_ease_world_36km"]

with h5py.File("parameters.hdf5", "w") as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()

print("Section 1 is completed")


#######################################################################################
# 2. Subset the lat/lon tables of EASE Grid in CONUS region

# EASE 1 km
lat_ease_conus_1km = lat_ease_world_1km[np.where((lat_ease_world_1km <= lat_geo_conus_max) &
                                                 (lat_ease_world_1km >= lat_geo_conus_min))].reshape(-1, 1)
lon_ease_conus_1km = lon_ease_world_1km[np.where((lon_ease_world_1km <= lon_geo_conus_max) &
                                                 (lon_ease_world_1km >= lon_geo_conus_min))].reshape(-1, 1)
row_ease_conus_1km_ind = np.where((lat_ease_world_1km <= lat_geo_conus_max) &
                                  (lat_ease_world_1km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_ease_conus_1km_ind = np.where((lon_ease_world_1km <= lon_geo_conus_max) &
                                  (lon_ease_world_1km >= lon_geo_conus_min))[0].reshape(-1, 1)

# EASE 9 km
lat_ease_conus_9km = lat_ease_world_9km[np.where((lat_ease_world_9km <= lat_geo_conus_max) &
                                                 (lat_ease_world_9km >= lat_geo_conus_min))].reshape(-1, 1)
lon_ease_conus_9km = lon_ease_world_9km[np.where((lon_ease_world_9km <= lon_geo_conus_max) &
                                                 (lon_ease_world_9km >= lon_geo_conus_min))].reshape(-1, 1)
row_ease_conus_9km_ind = np.where((lat_ease_world_9km <= lat_geo_conus_max) &
                                  (lat_ease_world_9km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_ease_conus_9km_ind = np.where((lon_ease_world_9km <= lon_geo_conus_max) &
                                  (lon_ease_world_9km >= lon_geo_conus_min))[0].reshape(-1, 1)

# EASE 12.5 km
lat_ease_conus_12_5km = lat_ease_world_12_5km[np.where((lat_ease_world_12_5km <= lat_geo_conus_max) &
                                                       (lat_ease_world_12_5km >= lat_geo_conus_min))].reshape(-1, 1)
lon_ease_conus_12_5km = lon_ease_world_12_5km[np.where((lon_ease_world_12_5km <= lon_geo_conus_max) &
                                                       (lon_ease_world_12_5km >= lon_geo_conus_min))].reshape(-1, 1)
row_ease_conus_12_5km_ind = np.where((lat_ease_world_12_5km <= lat_geo_conus_max) &
                                     (lat_ease_world_12_5km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_ease_conus_12_5km_ind = np.where((lon_ease_world_12_5km <= lon_geo_conus_max) &
                                     (lon_ease_world_12_5km >= lon_geo_conus_min))[0].reshape(-1, 1)

# EASE 25 km
lat_ease_conus_25km = lat_ease_world_25km[np.where((lat_ease_world_25km <= lat_geo_conus_max) &
                                                   (lat_ease_world_25km >= lat_geo_conus_min))].reshape(-1, 1)
lon_ease_conus_25km = lon_ease_world_25km[np.where((lon_ease_world_25km <= lon_geo_conus_max) &
                                                   (lon_ease_world_25km >= lon_geo_conus_min))].reshape(-1, 1)
row_ease_conus_25km_ind = np.where((lat_ease_world_25km <= lat_geo_conus_max) &
                                   (lat_ease_world_25km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_ease_conus_25km_ind = np.where((lon_ease_world_25km <= lon_geo_conus_max) &
                                   (lon_ease_world_25km >= lon_geo_conus_min))[0].reshape(-1, 1)

# EASE 36 km
lat_ease_conus_36km = lat_ease_world_36km[np.where((lat_ease_world_36km <= lat_geo_conus_max) &
                                                   (lat_ease_world_36km >= lat_geo_conus_min))].reshape(-1, 1)
lon_ease_conus_36km = lon_ease_world_36km[np.where((lon_ease_world_36km <= lon_geo_conus_max) &
                                                   (lon_ease_world_36km >= lon_geo_conus_min))].reshape(-1, 1)
row_ease_conus_36km_ind = np.where((lat_ease_world_36km <= lat_geo_conus_max) &
                                   (lat_ease_world_36km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_ease_conus_36km_ind = np.where((lon_ease_world_36km <= lon_geo_conus_max) &
                                   (lon_ease_world_36km >= lon_geo_conus_min))[0].reshape(-1, 1)

#######################################################################################
# 3. Generate lat/lon tables of Geographic projection in world/CONUS

# World extent (1 km)
lat_geo_world_1km = np.linspace(lat_geo_world_max - cellsize_1km/2, lat_geo_world_min + cellsize_1km/2,
                                num=int((lat_geo_world_max - lat_geo_world_min)/cellsize_1km)).reshape(-1, 1)
lat_geo_world_1km = np.round(lat_geo_world_1km, decimals=3)
lon_geo_world_1km = np.linspace(lon_geo_world_min + cellsize_1km/2, lon_geo_world_max - cellsize_1km/2,
                                num=int((lon_geo_world_max - lon_geo_world_min)/cellsize_1km)).reshape(-1, 1)
lon_geo_world_1km = np.round(lon_geo_world_1km, decimals=3)

# World extent (5 km)
lat_geo_world_5km = np.linspace(lat_geo_world_max - cellsize_5km/2, lat_geo_world_min + cellsize_5km/2,
                                num=int((lat_geo_world_max - lat_geo_world_min)/cellsize_5km)).reshape(-1, 1)
lat_geo_world_5km = np.round(lat_geo_world_5km, decimals=3)
lon_geo_world_5km = np.linspace(lon_geo_world_min + cellsize_5km/2, lon_geo_world_max - cellsize_5km/2,
                                num=int((lon_geo_world_max - lon_geo_world_min)/cellsize_5km)).reshape(-1, 1)
lon_geo_world_5km = np.round(lon_geo_world_5km, decimals=3)

# World extent (12.5 km)
lat_geo_world_12_5km = np.linspace(lat_geo_world_max - cellsize_12_5km/2, lat_geo_world_min + cellsize_12_5km/2,
                                num=int((lat_geo_world_max - lat_geo_world_min)/cellsize_12_5km)).reshape(-1, 1)
lat_geo_world_12_5km = np.round(lat_geo_world_12_5km, decimals=4)
lon_geo_world_12_5km = np.linspace(lon_geo_world_min + cellsize_12_5km/2, lon_geo_world_max - cellsize_12_5km/2,
                                num=int((lon_geo_world_max - lon_geo_world_min)/cellsize_12_5km)).reshape(-1, 1)
lon_geo_world_12_5km = np.round(lon_geo_world_12_5km, decimals=4)

# World extent (25 km)
lat_geo_world_25km = np.linspace(lat_geo_world_max - cellsize_25km/2, lat_geo_world_min + cellsize_25km/2,
                                num=int((lat_geo_world_max - lat_geo_world_min)/cellsize_25km)).reshape(-1, 1)
lat_geo_world_25km = np.round(lat_geo_world_25km, decimals=3)
lon_geo_world_25km = np.linspace(lon_geo_world_min + cellsize_25km/2, lon_geo_world_max - cellsize_25km/2,
                                num=int((lon_geo_world_max - lon_geo_world_min)/cellsize_25km)).reshape(-1, 1)
lon_geo_world_25km = np.round(lon_geo_world_25km, decimals=3)


# CONUS extent (1 km)
lat_geo_conus_1km = lat_geo_world_1km[np.where((lat_geo_world_1km <= lat_geo_conus_max) &
                                               (lat_geo_world_1km >= lat_geo_conus_min))].reshape(-1, 1)
lon_geo_conus_1km = lon_geo_world_1km[np.where((lon_geo_world_1km <= lon_geo_conus_max) &
                                               (lon_geo_world_1km >= lon_geo_conus_min))].reshape(-1, 1)
row_geo_conus_1km_ind = np.where((lat_geo_world_1km <= lat_geo_conus_max) &
                                 (lat_geo_world_1km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_geo_conus_1km_ind = np.where((lon_geo_world_1km <= lon_geo_conus_max) &
                                 (lon_geo_world_1km >= lon_geo_conus_min))[0].reshape(-1, 1)

# CONUS extent (5 km)
lat_geo_conus_5km = lat_geo_world_5km[np.where((lat_geo_world_5km <= lat_geo_conus_max) &
                                               (lat_geo_world_5km >= lat_geo_conus_min))].reshape(-1, 1)
lon_geo_conus_5km = lon_geo_world_5km[np.where((lon_geo_world_5km <= lon_geo_conus_max) &
                                               (lon_geo_world_5km >= lon_geo_conus_min))].reshape(-1, 1)
row_geo_conus_5km_ind = np.where((lat_geo_world_5km <= lat_geo_conus_max) &
                                 (lat_geo_world_5km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_geo_conus_5km_ind = np.where((lon_geo_world_5km <= lon_geo_conus_max) &
                                 (lon_geo_world_5km >= lon_geo_conus_min))[0].reshape(-1, 1)

# CONUS extent (12.5 km)
lat_geo_conus_12_5km = lat_geo_world_12_5km[np.where((lat_geo_world_12_5km <= lat_geo_conus_max) &
                                                     (lat_geo_world_12_5km >= lat_geo_conus_min))].reshape(-1, 1)
lon_geo_conus_12_5km = lon_geo_world_12_5km[np.where((lon_geo_world_12_5km <= lon_geo_conus_max) &
                                                     (lon_geo_world_12_5km >= lon_geo_conus_min))].reshape(-1, 1)
row_geo_conus_12_5km_ind = np.where((lat_geo_world_12_5km <= lat_geo_conus_max) &
                                    (lat_geo_world_12_5km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_geo_conus_12_5km_ind = np.where((lon_geo_world_12_5km <= lon_geo_conus_max) &
                                    (lon_geo_world_12_5km >= lon_geo_conus_min))[0].reshape(-1, 1)

# CONUS extent (25 km)
lat_geo_conus_25km = lat_geo_world_25km[np.where((lat_geo_world_25km <= lat_geo_conus_max) &
                                                 (lat_geo_world_25km >= lat_geo_conus_min))].reshape(-1, 1)
lon_geo_conus_25km = lon_geo_world_25km[np.where((lon_geo_world_25km <= lon_geo_conus_max) &
                                                 (lon_geo_world_25km >= lon_geo_conus_min))].reshape(-1, 1)
row_geo_conus_25km_ind = np.where((lat_geo_world_25km <= lat_geo_conus_max) &
                                  (lat_geo_world_25km >= lat_geo_conus_min))[0].reshape(-1, 1)
col_geo_conus_25km_ind = np.where((lon_geo_world_25km <= lon_geo_conus_max) &
                                  (lon_geo_world_25km >= lon_geo_conus_min))[0].reshape(-1, 1)


# Save new generated variables from section 2 & 3 to parameters.hdf5
var_name_2_3 = ["lat_ease_conus_1km", "lon_ease_conus_1km", "lat_ease_conus_9km", "lon_ease_conus_9km",
                "lat_ease_conus_12_5km", "lon_ease_conus_12_5km", "lat_ease_conus_25km", "lon_ease_conus_25km",
                "lat_ease_conus_36km", "lon_ease_conus_36km", "lat_geo_world_1km", "lon_geo_world_1km",
                "lat_geo_world_5km", "lon_geo_world_5km", "lat_geo_world_12_5km", "lon_geo_world_12_5km",
                "lat_geo_world_25km", "lon_geo_world_25km", "lat_geo_conus_1km", "lon_geo_conus_1km",
                "lat_geo_conus_5km", "lon_geo_conus_5km", "lat_geo_conus_12_5km", "lon_geo_conus_12_5km",
                "lat_geo_conus_25km", "lon_geo_conus_25km", "row_ease_conus_1km_ind", "col_ease_conus_1km_ind",
                "row_ease_conus_9km_ind", "col_ease_conus_9km_ind", "row_ease_conus_12_5km_ind",
                "col_ease_conus_12_5km_ind", "row_ease_conus_25km_ind", "col_ease_conus_25km_ind",
                "row_ease_conus_36km_ind", "col_ease_conus_36km_ind", "row_geo_conus_1km_ind",
                "col_geo_conus_1km_ind", "row_geo_conus_5km_ind", "col_geo_conus_5km_ind",
                "row_geo_conus_12_5km_ind", "col_geo_conus_12_5km_ind", "row_geo_conus_25km_ind",
                "col_geo_conus_25km_ind"]

with h5py.File("parameters.hdf5", "a") as f:
    for x in var_name_2_3:
        f.create_dataset(x, data=eval(x))
f.close()

print("Section 2/3 is completed")


#####################################################################################################
# 4. Find the corresponding index numbers for the high spatial resolution row/col tables
# from the low spatial resolution row/col tables.

# For 1 km from 9 km
[row_ease_conus_1km_from_9km_ind, col_ease_conus_1km_from_9km_ind] = \
    find_easeind_hifrlo(lat_ease_conus_1km, lon_ease_conus_1km, interdist_ease_9km, size_ease_world_9km[0],
                        size_ease_world_9km[1], row_ease_conus_9km_ind, col_ease_conus_9km_ind)

# For 1 km from 12.5 km
[row_ease_conus_1km_from_12_5km_ind, col_ease_conus_1km_from_12_5km_ind] = \
    find_easeind_hifrlo(lat_ease_conus_1km, lon_ease_conus_1km, interdist_ease_12_5km, size_ease_world_12_5km[0],
                        size_ease_world_12_5km[1], row_ease_conus_12_5km_ind, col_ease_conus_12_5km_ind)

# For 1 km from 25 km
[row_ease_conus_1km_from_25km_ind, col_ease_conus_1km_from_25km_ind] = \
    find_easeind_hifrlo(lat_ease_conus_1km, lon_ease_conus_1km, interdist_ease_25km, size_ease_world_25km[0],
                        size_ease_world_25km[1], row_ease_conus_25km_ind, col_ease_conus_25km_ind)

# For 1 km from 36 km
[row_ease_conus_1km_from_36km_ind, col_ease_conus_1km_from_36km_ind] = \
    find_easeind_hifrlo(lat_ease_conus_1km, lon_ease_conus_1km, interdist_ease_36km, size_ease_world_36km[0],
                        size_ease_world_36km[1], row_ease_conus_36km_ind, col_ease_conus_36km_ind)



