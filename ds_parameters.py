import os
import numpy as np
import h5py

# Specify file paths
# Path of EASE projection lat/lon tables
path_ease_coord_table = '/Volumes/My Passport/SMAP_Project/Datasets/geolocation'
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of backup
path_backup = '/Users/binfang/Documents/SMAP_CONUS/codes_py/backup'
# Path of MODIS LST folder
path_modis_lst = '/Volumes/My Passport/SMAP_Project/Datasets/MODIS/MYD11A1/Reprojected'

#########################################################################################################
# (Function 1) Define a function for generating EASE grid projection lat/lon tables from CATDS provided data

def ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col):
    global lat_ease_file, lat_world_ease, lon_ease_file, lon_world_ease
    lat_ease_file = open(path_ease_lat, 'r')
    lat_world_ease = np.fromfile(lat_ease_file, dtype=np.dtype('f8'))
    lat_world_ease = np.reshape(lat_world_ease, (num_row, num_col))
    lat_world_ease = np.array(lat_world_ease[:, 1])
    lat_ease_file.close()

    lon_ease_file = open(path_ease_lon, 'r')
    lon_world_ease = np.fromfile(lon_ease_file, dtype=np.dtype('f8'))
    lon_world_ease = np.reshape(lon_world_ease, (num_row, num_col))
    lon_world_ease = np.array(lon_world_ease[1, :])
    lon_ease_file.close()

    del lat_ease_file, lon_ease_file, path_ease_lat, path_ease_lon, num_row, num_col

    return lat_world_ease, lon_world_ease


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
# (Function 3) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind

#########################################################################################
# (Function 4) Generate lat/lon tables of Geographic projection in world/CONUS

def geo_coord_gen(lat_geo_extent_max, lat_geo_extent_min, lon_geo_extent_max, lon_geo_extent_min, cellsize):
    lat_geo_output = np.linspace(lat_geo_extent_max - cellsize / 2, lat_geo_extent_min + cellsize / 2,
                                    num=int((lat_geo_extent_max - lat_geo_extent_min) / cellsize))
    lat_geo_output = np.round(lat_geo_output, decimals=3)
    lon_geo_output = np.linspace(lon_geo_extent_min + cellsize / 2, lon_geo_extent_max - cellsize / 2,
                                    num=int((lon_geo_extent_max - lon_geo_extent_min) / cellsize))
    lon_geo_output = np.round(lon_geo_output, decimals=3)

    return lat_geo_output, lon_geo_output

#########################################################################################
# (Function 5) Find and map the corresponding index numbers for the high spatial resolution
# row/col tables from the low spatial resolution row/col tables. The output is 1-dimensional
# array containing index numbers.

def find_easeind_hifrlo(lat_hires, lon_hires, interdist_lowres, num_row_lowres, num_col_lowres, row_lowres_ind, col_lowres_ind):

    lon_meshgrid, lat_meshgrid = np.meshgrid(lon_hires, lat_hires)
    lat_meshgrid_array = lat_meshgrid.flatten()
    lon_meshgrid_array = lon_meshgrid.flatten()

    [row_ind_toresp, col_ind_toresp] = \
        geo2easeGridV2(lat_meshgrid_array, lon_meshgrid_array, interdist_lowres,
                       num_row_lowres, num_col_lowres)

    row_ind_toresp = np.reshape(row_ind_toresp, np.shape(lon_meshgrid))
    col_ind_toresp = np.reshape(col_ind_toresp, np.shape(lon_meshgrid))

    row_ind_toresp = np.array(row_ind_toresp[:, 1])
    col_ind_toresp = np.array(col_ind_toresp[1, :])

    # Assign the empty to-be-resampled grids with index numbers of corresponding nearest destination grids
    row_ind_diff = np.setdiff1d(row_ind_toresp, row_lowres_ind)
    if row_ind_diff.size != 0:
        for x in range(len(row_ind_diff)):
            row_dist = np.absolute(row_ind_diff[x] - row_lowres_ind)
            row_ind_toresp[np.where(row_ind_toresp == row_ind_diff[x])] = row_lowres_ind[np.argmin(row_dist)]
    else:
        pass

    row_unique, row_dest = np.unique(row_ind_toresp, return_inverse=True)
    row_ind_unique = row_unique
    row_ind_dest = row_dest

    col_ind_diff = np.setdiff1d(col_ind_toresp, col_lowres_ind)
    if col_ind_diff.size != 0:
        for x in range(len(col_ind_diff)):
            col_dist = np.absolute(col_ind_diff[x] - col_lowres_ind)
            col_ind_toresp[np.where(col_ind_toresp == col_ind_diff[x])] = col_lowres_ind[np.argmin(col_dist)]
    else:
        pass

    col_unique, col_dest = np.unique(col_ind_toresp, return_inverse=True)
    col_ind_unique = col_unique
    col_ind_dest = col_dest

    del lat_meshgrid, lon_meshgrid, lat_meshgrid_array, lon_meshgrid_array, row_ind_toresp, col_ind_toresp, \
        row_ind_unique, col_ind_unique, row_ind_diff, col_ind_diff, row_unique, row_dest, col_unique, col_dest

    return row_ind_dest, col_ind_dest

#########################################################################################
# (Function 6) Find and map the corresponding index numbers for the low spatial resolution
# row/col tables from the high spatial resolution row/col tables. The output is 1-dimensional
# nested list array containing index numbers.

def find_easeind_lofrhi(lat_hires, lon_hires, interdist_lowres, num_row_lowres, num_col_lowres, row_lowres_ind, col_lowres_ind):

    lon_meshgrid, lat_meshgrid = np.meshgrid(lon_hires, lat_hires)
    lat_meshgrid_array = lat_meshgrid.flatten()
    lon_meshgrid_array = lon_meshgrid.flatten()

    [row_ind_toresp, col_ind_toresp] = \
        geo2easeGridV2(lat_meshgrid_array, lon_meshgrid_array, interdist_lowres,
                       num_row_lowres, num_col_lowres)

    row_ind_toresp = np.reshape(row_ind_toresp, np.shape(lon_meshgrid))
    col_ind_toresp = np.reshape(col_ind_toresp, np.shape(lon_meshgrid))

    row_ind_toresp = np.array(row_ind_toresp[:, 1])
    col_ind_toresp = np.array(col_ind_toresp[1, :])

    # Assign the low resolution grids with corresponding high resolution grids index numbers
    row_ease_dest_init = []
    for x in range(len(row_lowres_ind)):
        row_ind = np.where(row_ind_toresp == row_lowres_ind[x])
        row_ind = np.array(row_ind).ravel()
        row_ease_dest_init.append(row_ind)

    row_ease_dest_ind = np.asarray(row_ease_dest_init)

    col_ease_dest_init = []
    for x in range(len(col_lowres_ind)):
        col_ind = np.where(col_ind_toresp == col_lowres_ind[x])
        col_ind = np.array(col_ind).ravel()
        col_ease_dest_init.append(col_ind)

    col_ease_dest_ind = np.asarray(col_ease_dest_init)

    # Assign the empty to-be-resampled grids with index numbers of corresponding nearest destination grids
    for x in range(len(row_ease_dest_ind)):
        if len(row_ease_dest_ind[x]) == 0 and x != 0 and x != len(row_ease_dest_ind)-1:
            # Exclude the first and last elements
            row_ease_dest_ind[x] = np.array([row_ease_dest_ind[x - 1], row_ease_dest_ind[x + 1]]).ravel()
        else:
            pass

    for x in range(len(col_ease_dest_ind)):
        if len(col_ease_dest_ind[x]) == 0 and x != 0 and x != len(col_ease_dest_ind)-1:
            # Exclude the first and last elements
            col_ease_dest_ind[x] = np.array([col_ease_dest_ind[x - 1], col_ease_dest_ind[x + 1]]).ravel()
        else:
            pass

    return row_ease_dest_ind, col_ease_dest_ind

#########################################################################################
# (Function 7) Find and map the corresponding index numbers for the low spatial resolution
# row/col tables from the high spatial resolution row/col tables. The output is 1-dimensional
# nested list array containing index numbers. (For 33 km SMAP grid extension only)

def find_easeind_lofrhi_ext33km(lat_hires, lon_hires, interdist_lowres,
                                num_row_lowres, num_col_lowres, row_lowres_ind, col_lowres_ind, ext_grid):

    lon_meshgrid, lat_meshgrid = np.meshgrid(lon_hires, lat_hires)
    lat_meshgrid_array = lat_meshgrid.flatten()
    lon_meshgrid_array = lon_meshgrid.flatten()

    [row_ind_toresp, col_ind_toresp] = \
        geo2easeGridV2(lat_meshgrid_array, lon_meshgrid_array, interdist_lowres,
                       num_row_lowres, num_col_lowres)

    row_ind_toresp = np.reshape(row_ind_toresp, np.shape(lon_meshgrid))
    col_ind_toresp = np.reshape(col_ind_toresp, np.shape(lon_meshgrid))

    row_ind_toresp = row_ind_toresp[:, 1].reshape(1, -1)
    col_ind_toresp = col_ind_toresp[1, :].reshape(1, -1)

    # Assign the low resolution grids with corresponding high resolution grids index numbers
    row_ease_dest_init = []
    for x in range(len(row_lowres_ind[0])):
        row_ind = np.where(row_ind_toresp == row_lowres_ind[0, x])
        row_ind = row_ind[1].ravel()
        row_ease_dest_init.append(row_ind)

    row_ease_dest_ind = np.asarray(row_ease_dest_init)

    col_ease_dest_init = []
    for x in range(len(col_lowres_ind[0])):
        col_ind = np.where(col_ind_toresp == col_lowres_ind[0, x])
        col_ind = col_ind[1].ravel()
        col_ease_dest_init.append(col_ind)

    col_ease_dest_ind = np.asarray(col_ease_dest_init)

    # Assign the empty to-be-resampled grids with index numbers of corresponding nearest destination grids
    for x in range(len(row_ease_dest_ind)):
        if len(row_ease_dest_ind[x]) == 0 and x != 0 and x != len(row_ease_dest_ind):
            # Exclude the first and last elements
            row_ease_dest_ind[x] = np.array([row_ease_dest_ind[x - 1], row_ease_dest_ind[x + 1]]).ravel()
        else:
            pass

    for x in range(len(col_ease_dest_ind)):
        if len(col_ease_dest_ind[x]) == 0 and x != 0 and x != len(col_ease_dest_ind):
            # Exclude the first and last elements
            col_ease_dest_ind[x] = np.array([col_ease_dest_ind[x - 1], col_ease_dest_ind[x + 1]]).ravel()
        else:
            pass

    # Generate the index tables extended to 33 km domain for the 9 km indexing table at 1-km grids: (33-9)/2=12.
    # So each side (left/right and up/down sides for col/row index tables extend 12 1-km pixels.)
    # ext_grid = 12

    row_ease_dest_ind_new_init = []
    for x in range(len(row_ease_dest_ind)):
        row_newmin = np.amin(row_ease_dest_ind[x]) - ext_grid
        row_newmax = np.amax(row_ease_dest_ind[x]) + ext_grid
        row_newarray = np.arange(row_newmin, row_newmax+1)
        row_newarray = row_newarray[np.where((row_newarray >= 0) & (row_newarray <= len(lat_hires[0])))].ravel()
        row_ease_dest_ind_new_init.append(row_newarray)

    row_ease_dest_ind_new = np.asarray(row_ease_dest_ind_new_init)

    col_ease_dest_ind_new_init = []
    for x in range(len(col_ease_dest_ind)):
        col_newmin = np.amin(col_ease_dest_ind[x]) - ext_grid
        col_newmax = np.amax(col_ease_dest_ind[x]) + ext_grid
        col_newarray = np.arange(col_newmin, col_newmax+1)
        col_newarray = col_newarray[np.where((col_newarray >= 0) & (col_newarray <= len(lon_hires[0])))].ravel()
        col_ease_dest_ind_new_init.append(col_newarray)

    col_ease_dest_ind_new = np.asarray(col_ease_dest_ind_new_init)

    row_ease_dest_ind_new = row_ease_dest_ind_new.reshape(1, -1)
    col_ease_dest_ind_new = col_ease_dest_ind_new.reshape(1, -1)

    return row_ease_dest_ind_new, col_ease_dest_ind_new


#########################################################################################
#########################################################################################
# 0. Input variables

# # Load in variables
# os.chdir(path_workspace)
# f = h5py.File("ds_parameters.hdf5", "r")
# varname_list = list(f.keys())
#
# for x in range(len(varname_list)):
#     var_obj = f[varname_list[x]][()]
#     exec(varname_list[x] + '= var_obj')
# f.close()

# World extent corner coordinates
lat_world_max = 90
lat_world_min = -90
lon_world_max = 180
lon_world_min = -180

# CONUS extent corner coordinates
lat_conus_max = 53
lat_conus_min = 25
lon_conus_max = -67
lon_conus_min = -125

# Cellsize
cellsize_400m = 0.004
cellsize_1km = 0.01
cellsize_5km = 0.05
cellsize_9km = 0.09
cellsize_12_5km = 0.125
cellsize_25km = 0.25
cellsize_36km = 0.36

# Global extent of EASE grid projection (row/col)
size_world_ease_400m = np.array([36540, 86760])
size_world_ease_1km = np.array([14616, 34704])
size_world_ease_9km = np.array([1624, 3856])
size_world_ease_12_5km = np.array([1168, 2776])
size_world_ease_25km = np.array([584, 1388])
size_world_ease_36km = np.array([406, 964])

# Interdistance of EASE grid projection grids
interdist_ease_400m = 400.4041601296
interdist_ease_1km = 1000.89502334956
interdist_ease_9km = 9009.093602916
interdist_ease_12_5km = 12512.63000405
interdist_ease_25km = 25067.525
interdist_ease_36km = 36036.374411664


#######################################################################################
# 1. Generate lat/lon tables of EASE Grid projection in the world
# Use data downloaded from ftp://sidads.colorado.edu/pub/tools/easegrid2/

os.chdir(path_ease_coord_table)

# EASE 1 km
path_ease_lat = 'EASE2_M01km.lats.34704x14616x1.double'
path_ease_lon = 'EASE2_M01km.lons.34704x14616x1.double'
num_row = size_world_ease_1km[0]
num_col = size_world_ease_1km[1]
[lat_world_ease_1km, lon_world_ease_1km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)

# EASE 9 km
path_ease_lat = 'EASE2_M09km.lats.3856x1624x1.double'
path_ease_lon = 'EASE2_M09km.lons.3856x1624x1.double'
num_row = size_world_ease_9km[0]
num_col = size_world_ease_9km[1]
[lat_world_ease_9km, lon_world_ease_9km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)

# EASE 12.5 km
path_ease_lat = 'EASE2_M12.5km.lats.2776x1168x1.double'
path_ease_lon = 'EASE2_M12.5km.lons.2776x1168x1.double'
num_row = size_world_ease_12_5km[0]
num_col = size_world_ease_12_5km[1]
[lat_world_ease_12_5km, lon_world_ease_12_5km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)

# EASE 25 km
path_ease_lat = 'EASE2_M25km.lats.1388x584x1.double'
path_ease_lon = 'EASE2_M25km.lons.1388x584x1.double'
num_row = size_world_ease_25km[0]
num_col = size_world_ease_25km[1]
[lat_world_ease_25km, lon_world_ease_25km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)

# EASE 36 km
path_ease_lat = 'EASE2_M36km.lats.964x406x1.double'
path_ease_lon = 'EASE2_M36km.lons.964x406x1.double'
num_row = size_world_ease_36km[0]
num_col = size_world_ease_36km[1]
[lat_world_ease_36km, lon_world_ease_36km] = \
    ease_coord_gen(path_ease_lat, path_ease_lon, num_row, num_col)


# Save variables
os.chdir(path_workspace)
var_name = ["cellsize_400m", "cellsize_1km", "cellsize_5km", "cellsize_9km",
            "cellsize_12_5km", "cellsize_25km", "cellsize_36km", "interdist_ease_400m",
            "interdist_ease_1km", "interdist_ease_9km", "interdist_ease_12_5km",
            "interdist_ease_25km", "interdist_ease_36km", "lat_conus_max",
            "lat_conus_min", "lon_conus_max", "lon_conus_min",
            "lat_world_max", "lat_world_min", "lon_world_max",
            "lon_world_min", "lat_world_ease_1km", "lon_world_ease_1km",
            "lat_world_ease_9km", "lon_world_ease_9km", "lat_world_ease_12_5km",
            "lon_world_ease_12_5km", "lat_world_ease_25km", "lon_world_ease_25km",
            "lat_world_ease_36km", "lon_world_ease_36km", "size_world_ease_400m",
            "size_world_ease_1km", "size_world_ease_9km", "size_world_ease_12_5km",
            "size_world_ease_25km", "size_world_ease_36km"]

with h5py.File("ds_parameters.hdf5", "w") as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()

print("Section 1 is completed")


#######################################################################################
# 2. Subset the lat/lon tables of EASE Grid at different spatial resolution in CONUS/world regions

[lat_conus_ease_1km, row_conus_ease_1km_ind, lon_conus_ease_1km, col_conus_ease_1km_ind] = coordtable_subset\
    (lat_world_ease_1km, lon_world_ease_1km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)
[lat_conus_ease_9km, row_conus_ease_9km_ind, lon_conus_ease_9km, col_conus_ease_9km_ind] = coordtable_subset\
    (lat_world_ease_9km, lon_world_ease_9km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)
[lat_conus_ease_12_5km, row_conus_ease_12_5km_ind, lon_conus_ease_12_5km, col_conus_ease_12_5km_ind] = coordtable_subset\
    (lat_world_ease_12_5km, lon_world_ease_12_5km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)
[lat_conus_ease_25km, row_conus_ease_25km_ind, lon_conus_ease_25km, col_conus_ease_25km_ind] = coordtable_subset\
    (lat_world_ease_25km, lon_world_ease_25km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)
[lat_conus_ease_36km, row_conus_ease_36km_ind, lon_conus_ease_36km, col_conus_ease_36km_ind] = coordtable_subset\
    (lat_world_ease_36km, lon_world_ease_36km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)

row_world_ease_1km_ind = np.arange(len(lat_world_ease_1km))
col_world_ease_1km_ind = np.arange(len(lon_world_ease_1km))
row_world_ease_9km_ind = np.arange(len(lat_world_ease_9km))
col_world_ease_9km_ind = np.arange(len(lon_world_ease_9km))
row_world_ease_25km_ind = np.arange(len(lat_world_ease_25km))
col_world_ease_25km_ind = np.arange(len(lon_world_ease_25km))
row_world_ease_36km_ind = np.arange(len(lat_world_ease_36km))
col_world_ease_36km_ind = np.arange(len(lon_world_ease_36km))


#######################################################################################
# 3. Generate lat/lon tables of Geographic projection in world/CONUS

# World extent
[lat_world_geo_1km, lon_world_geo_1km] = geo_coord_gen\
    (lat_world_max, lat_world_min, lon_world_max, lon_world_min, cellsize_1km)
[lat_world_geo_5km, lon_world_geo_5km] = geo_coord_gen\
    (lat_world_max, lat_world_min, lon_world_max, lon_world_min, cellsize_5km)
[lat_world_geo_12_5km, lon_world_geo_12_5km] = geo_coord_gen\
    (lat_world_max, lat_world_min, lon_world_max, lon_world_min, cellsize_12_5km)
[lat_world_geo_25km, lon_world_geo_25km] = geo_coord_gen\
    (lat_world_max, lat_world_min, lon_world_max, lon_world_min, cellsize_25km)

# CONUS extent
[lat_conus_geo_1km, row_conus_geo_1km_ind, lon_conus_geo_1km, col_conus_geo_1km_ind] = coordtable_subset\
    (lat_world_geo_1km, lon_world_geo_1km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)
[lat_conus_geo_5km, row_conus_geo_5km_ind, lon_conus_geo_5km, col_conus_geo_5km_ind] = coordtable_subset\
    (lat_world_geo_5km, lon_world_geo_5km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)
[lat_conus_geo_12_5km, row_conus_geo_12_5km_ind, lon_conus_geo_12_5km, col_conus_geo_12_5km_ind] = coordtable_subset\
    (lat_world_geo_12_5km, lon_world_geo_12_5km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)
[lat_conus_geo_25km, row_conus_geo_25km_ind, lon_conus_geo_25km, col_conus_geo_25km_ind] = coordtable_subset\
    (lat_world_geo_25km, lon_world_geo_25km, lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)


# Save new generated variables from section 2 & 3 to parameters.hdf5
var_name_2_3 = ["lat_conus_ease_1km", "lon_conus_ease_1km", "lat_conus_ease_9km", "lon_conus_ease_9km",
                "lat_conus_ease_12_5km", "lon_conus_ease_12_5km", "lat_conus_ease_25km", "lon_conus_ease_25km",
                "lat_conus_ease_36km", "lon_conus_ease_36km", "lat_world_geo_1km", "lon_world_geo_1km",
                "lat_world_geo_5km", "lon_world_geo_5km", "lat_world_geo_12_5km", "lon_world_geo_12_5km",
                "lat_world_geo_25km", "lon_world_geo_25km", "lat_conus_geo_1km", "lon_conus_geo_1km",
                "lat_conus_geo_5km", "lon_conus_geo_5km", "lat_conus_geo_12_5km", "lon_conus_geo_12_5km",
                "lat_conus_geo_25km", "lon_conus_geo_25km", "row_conus_ease_1km_ind", "col_conus_ease_1km_ind",
                "row_conus_ease_9km_ind", "col_conus_ease_9km_ind", "row_conus_ease_12_5km_ind",
                "col_conus_ease_12_5km_ind", "row_conus_ease_25km_ind", "col_conus_ease_25km_ind",
                "row_conus_ease_36km_ind", "col_conus_ease_36km_ind", "row_conus_geo_1km_ind",
                "col_conus_geo_1km_ind", "row_conus_geo_5km_ind", "col_conus_geo_5km_ind",
                "row_conus_geo_12_5km_ind", "col_conus_geo_12_5km_ind", "row_conus_geo_25km_ind",
                "col_conus_geo_25km_ind", "row_world_ease_1km_ind", "col_world_ease_1km_ind", "row_world_ease_9km_ind",
                "col_world_ease_9km_ind", "row_world_ease_25km_ind", "col_world_ease_25km_ind", "row_world_ease_36km_ind",
                "col_world_ease_36km_ind"]

with h5py.File("ds_parameters.hdf5", "a") as f:
    for x in var_name_2_3:
        f.create_dataset(x, data=eval(x))
f.close()

print("Section 2/3 are completed")


#####################################################################################################
# 4. Find the corresponding index numbers for the high spatial resolution row/col tables
# from the low spatial resolution row/col tables (Downscale).

# CONUS
# For 1 km from 9 km
[row_conus_ease_1km_from_9km_ind, col_conus_ease_1km_from_9km_ind] = \
    find_easeind_hifrlo(lat_conus_ease_1km, lon_conus_ease_1km, interdist_ease_9km, size_world_ease_9km[0],
                        size_world_ease_9km[1], row_conus_ease_9km_ind, col_conus_ease_9km_ind)

# For 1 km from 12.5 km
[row_conus_ease_1km_from_12_5km_ind, col_conus_ease_1km_from_12_5km_ind] = \
    find_easeind_hifrlo(lat_conus_ease_1km, lon_conus_ease_1km, interdist_ease_12_5km, size_world_ease_12_5km[0],
                        size_world_ease_12_5km[1], row_conus_ease_12_5km_ind, col_conus_ease_12_5km_ind)

# For 1 km from 25 km
[row_conus_ease_1km_from_25km_ind, col_conus_ease_1km_from_25km_ind] = \
    find_easeind_hifrlo(lat_conus_ease_1km, lon_conus_ease_1km, interdist_ease_25km, size_world_ease_25km[0],
                        size_world_ease_25km[1], row_conus_ease_25km_ind, col_conus_ease_25km_ind)

# For 1 km from 36 km
[row_conus_ease_1km_from_36km_ind, col_conus_ease_1km_from_36km_ind] = \
    find_easeind_hifrlo(lat_conus_ease_1km, lon_conus_ease_1km, interdist_ease_36km, size_world_ease_36km[0],
                        size_world_ease_36km[1], row_conus_ease_36km_ind, col_conus_ease_36km_ind)

# World
# For 1 km from 9 km
[row_world_ease_1km_from_9km_ind, col_world_ease_1km_from_9km_ind] = \
    find_easeind_hifrlo(lat_world_ease_1km, lon_world_ease_1km, interdist_ease_9km, size_world_ease_9km[0],
                        size_world_ease_9km[1], row_world_ease_9km_ind, col_world_ease_9km_ind)

# For 1 km from 25 km
[row_world_ease_1km_from_25km_ind, col_world_ease_1km_from_25km_ind] = \
    find_easeind_hifrlo(lat_world_ease_1km, lon_world_ease_1km, interdist_ease_25km, size_world_ease_25km[0],
                        size_world_ease_25km[1], row_world_ease_25km_ind, col_world_ease_25km_ind)

# For 1 km from 36 km
[row_world_ease_1km_from_36km_ind, col_world_ease_1km_from_36km_ind] = \
    find_easeind_hifrlo(lat_world_ease_1km, lon_world_ease_1km, interdist_ease_36km, size_world_ease_36km[0],
                        size_world_ease_36km[1], row_world_ease_36km_ind, col_world_ease_36km_ind)


#####################################################################################################
# 5. Find the corresponding index numbers for the low spatial resolution row/col tables
# from the high spatial resolution row/col tables (Upscale)
# High resolution: Geographic projection
# Low resolution: EASE grid projection

# CONUS
[row_conus_ease_1km_from_geo_1km_ind, col_conus_ease_1km_from_geo_1km_ind] = \
    find_easeind_lofrhi(lat_conus_geo_1km, lon_conus_geo_1km, interdist_ease_1km,
                        size_world_ease_1km[0], size_world_ease_1km[1], row_conus_ease_1km_ind, col_conus_ease_1km_ind)

[row_conus_ease_12_5km_from_geo_5km_ind, col_conus_ease_12_5km_from_geo_5km_ind] = \
    find_easeind_lofrhi(lat_conus_geo_5km, lon_conus_geo_5km, interdist_ease_12_5km,
                        size_world_ease_12_5km[0], size_world_ease_12_5km[1], row_conus_ease_12_5km_ind, col_conus_ease_12_5km_ind)

[row_conus_ease_12_5km_from_geo_12_5km_ind, col_conus_ease_12_5km_from_geo_12_5km_ind] = \
    find_easeind_lofrhi(lat_conus_geo_12_5km, lon_conus_geo_12_5km, interdist_ease_12_5km,
                        size_world_ease_12_5km[0], size_world_ease_12_5km[1], row_conus_ease_12_5km_ind, col_conus_ease_12_5km_ind)

# World
[row_world_ease_1km_from_geo_1km_ind, col_world_ease_1km_from_geo_1km_ind] = \
    find_easeind_lofrhi(lat_world_geo_1km, lon_world_geo_1km, interdist_ease_1km,
                        size_world_ease_1km[0], size_world_ease_1km[1], row_world_ease_1km_ind, col_world_ease_1km_ind)

[row_world_ease_25km_from_geo_5km_ind, col_world_ease_25km_from_geo_5km_ind] = \
    find_easeind_lofrhi(lat_world_geo_5km, lon_world_geo_5km, interdist_ease_25km,
                        size_world_ease_25km[0], size_world_ease_25km[1], row_world_ease_25km_ind, col_world_ease_25km_ind)


# Save new generated variables from section 4 & 5 to parameters.hdf5
var_name_4_5 = ["row_conus_ease_1km_from_9km_ind", "col_conus_ease_1km_from_9km_ind", "row_conus_ease_1km_from_12_5km_ind",
                "col_conus_ease_1km_from_12_5km_ind", "row_conus_ease_1km_from_25km_ind", "col_conus_ease_1km_from_25km_ind",
                "row_conus_ease_1km_from_36km_ind", "col_conus_ease_1km_from_36km_ind"]
var_name_4_5_vlen = ["row_conus_ease_1km_from_geo_1km_ind", "col_conus_ease_1km_from_geo_1km_ind",
                     "row_conus_ease_12_5km_from_geo_5km_ind", "col_conus_ease_12_5km_from_geo_5km_ind",
                     "row_conus_ease_12_5km_from_geo_12_5km_ind", "col_conus_ease_12_5km_from_geo_12_5km_ind",
                     "row_conus_ease_25km_from_geo_5km_ind", "col_conus_ease_25km_from_geo_5km_ind",
                     "row_world_ease_1km_from_geo_1km_ind", "col_world_ease_1km_from_geo_1km_ind",
                     "row_world_ease_25km_from_geo_5km_ind", "col_world_ease_25km_from_geo_5km_ind"]

with h5py.File("ds_parameters.hdf5", "a") as f:
    for x in var_name_4_5:
        f.create_dataset(x, data=eval(x))
f.close()

# Store variable-length type variables to the parameter file
dt = h5py.special_dtype(vlen=np.int64)
with h5py.File("ds_parameters.hdf5", "a") as f:
    for x in var_name_4_5_vlen:
        f.create_dataset(x, data=eval(x), dtype=dt)
f.close()

print("Section 4/5 is completed")


#####################################################################################################
# 6. Find the corresponding index numbers for the low spatial resolution row/col tables
# from the high spatial resolution row/col tables (Upscale, 33-km extension)

ext_grid = 12
[row_conus_ease_9km_from_1km_ext33km_ind, col_conus_ease_9km_from_1km_ext33km_ind] = find_easeind_lofrhi_ext33km\
    (lat_conus_ease_1km, lon_conus_ease_1km, interdist_ease_9km, size_world_ease_9km[0], size_world_ease_9km[1],
     row_conus_ease_9km_ind, col_conus_ease_9km_ind, ext_grid)

# Save new generated variables from section 6 to parameters.hdf5
var_name_6_vlen = ["row_conus_ease_9km_from_1km_ext33km_ind", "col_conus_ease_9km_from_1km_ext33km_ind"]

# Store variable-length type variables to the parameter file
dt = h5py.special_dtype(vlen=np.int64)
with h5py.File("ds_parameters.hdf5", "a") as f:
    for x in var_name_6_vlen:
        f.create_dataset(x, data=eval(x), dtype=dt)
f.close()

print("Section 6 is completed")
