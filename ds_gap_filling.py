import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from netCDF4 import Dataset
import calendar
import datetime
import glob
import gdal
from sklearn import linear_model
regr = linear_model.LinearRegression()
from sklearn.metrics import mean_squared_error, r2_score
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


##############################################################################################################
# (Function 1) Generate lat/lon tables of Geographic projection in world/CONUS

def geo_coord_gen(lat_geo_extent_max, lat_geo_extent_min, lon_geo_extent_max, lon_geo_extent_min, cellsize):
    lat_geo_output = np.linspace(lat_geo_extent_max - cellsize / 2, lat_geo_extent_min + cellsize / 2,
                                    num=int((lat_geo_extent_max - lat_geo_extent_min) / cellsize))
    lat_geo_output = np.round(lat_geo_output, decimals=3)
    lon_geo_output = np.linspace(lon_geo_extent_min + cellsize / 2, lon_geo_extent_max - cellsize / 2,
                                    num=int((lon_geo_extent_max - lon_geo_extent_min) / cellsize))
    lon_geo_output = np.round(lon_geo_output, decimals=3)

    return lat_geo_output, lon_geo_output

##############################################################################################################
# (Function 2) Define a function to output coefficient and intercept of linear regression fit

def reg_proc(x_arr, y_arr):
    x_arr = np.atleast_1d(x_arr)
    y_arr = np.atleast_1d(y_arr)
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_arr = x_arr[mask].reshape(-1, 1)
    y_arr = y_arr[mask].reshape(-1, 1)
    if len(x_arr) > 8 and len(x_arr) == len(y_arr):
        regr.fit(x_arr, y_arr)
        y_pred = regr.predict(x_arr)
        coef = regr.coef_.item()
        intc = regr.intercept_.item()
        r2 = r2_score(y_arr, y_pred)
        rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
    else:
        coef = np.nan
        intc = np.nan
        r2 = np.nan
        rmse = np.nan

    return coef, intc, r2, rmse

########################################################################################################################
# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of source GLDAS data
path_gldas = '/Volumes/MyPassport/SMAP_Project/Datasets/GLDAS'
# Path of source LTDR NDVI data
path_ltdr = '/Volumes/MyPassport/SMAP_Project/Datasets/LTDR/Ver5'
# Path of processed data
path_procdata = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'

# Load in variables
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['row_world_ease_1km_from_25km_ind', 'col_world_ease_1km_from_25km_ind',
                    'row_world_ease_1km_from_geo_1km_ind', 'col_world_ease_1km_from_geo_1km_ind',
                    'row_world_ease_25km_from_geo_5km_ind', 'col_world_ease_25km_from_geo_5km_ind',
                    'row_world_ease_25km_from_geo_25km_ind', 'col_world_ease_25km_from_geo_25km_ind',
                    'row_world_ease_9km_from_1km_ext33km_ind', 'col_world_ease_9km_from_1km_ext33km_ind',
                    'lat_world_ease_25km', 'lon_world_ease_25km', 'lat_world_ease_9km', 'lon_world_ease_9km',
                    'lat_world_ease_1km', 'lon_world_ease_1km', 'lat_world_geo_1km', 'lon_world_geo_1km',
                    'lat_world_geo_5km', 'lon_world_geo_5km', 'lat_world_geo_25km', 'lon_world_geo_25km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()

# Generate land/water mask provided by GLDAS/NASA
# lmask_file = open(path_lmask + '/FLDAS_GLOBAL_watermask_MOD44W.nc', 'r')
rootgrp = Dataset(path_lmask + '/FLDAS_GLOBAL_watermask_MOD44W.nc', mode='r')
lmask_read = rootgrp.variables['LANDMASK'][:]
lmask_read = np.ma.getdata(np.squeeze((lmask_read)))
lmask_read = np.flipud(lmask_read)
rootgrp.close()
lmask_sh = np.empty((300, lmask_read.shape[1]), dtype='float32')
lmask_sh[:] = 0
lmask_10km = np.concatenate((lmask_read, lmask_sh), axis=0)


# World extent corner coordinates
lat_world_max = 90
lat_world_min = -90
lon_world_max = 180
lon_world_min = -180

cellsize_10km = 0.1
cellsize_1km = 0.01
size_world_ease_1km = np.array([14616, 34704])
interdist_ease_1km = 1000.89502334956

# # Generate 1 km lat/lon tables and corresponding row/col indices in the world lat/lon table
# [lat_world_geo_10km, lon_world_geo_10km] = geo_coord_gen\
#     (lat_world_max, lat_world_min, lon_world_max, lon_world_min, cellsize_10km)
#
# row_world_geo_10km_ind = []
# for x in range(len(lat_world_ease_1km)):
#     row_dist = np.absolute(lat_world_ease_1km[x] - lat_world_geo_10km)
#     row_ind = np.argmin(row_dist)
#     row_world_geo_10km_ind.append(row_ind)
#     del(row_ind, row_dist)
# row_world_geo_10km_ind = np.array(row_world_geo_10km_ind)
#
# col_world_geo_10km_ind = []
# for x in range(len(lon_world_ease_1km)):
#     col_dist = np.absolute(lon_world_ease_1km[x] - lon_world_geo_10km)
#     col_ind = np.argmin(col_dist)
#     col_world_geo_10km_ind.append(col_ind)
#     del(col_ind, col_dist)
# col_world_geo_10km_ind = np.array(col_world_geo_10km_ind)
#
# # Convert the match table files to 1-d linear
# col_world_geo_10km_mesh_ind, row_world_geo_10km_mesh_ind = np.meshgrid(col_world_geo_10km_ind, row_world_geo_10km_ind)
# col_world_geo_10km_mesh_ind = col_world_geo_10km_mesh_ind.reshape(1, -1)
# row_world_geo_10km_mesh_ind = row_world_geo_10km_mesh_ind.reshape(1, -1)
# coord_world_geo_10km_mesh_ind = \
#     np.ravel_multi_index(np.array([row_world_geo_10km_mesh_ind[0], col_world_geo_10km_mesh_ind[0]]), lmask_10km.shape)
#
# # Disaggregate the landmask from 10 km to 1 km of EASE grid projection
# lmask_init = np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km)], dtype='float32')
# lmask_init = lmask_init.reshape(1, -1)
# lmask_init[:] = 0
#
# lmask_10km_1dim = lmask_10km.reshape(1, -1)
# lmask_10km_1dim_ind = np.where(lmask_10km_1dim == 1)[1]
#
# # Find out the land pixels on the 1 km EASE projection map
# coord_world_1km_ind = np.where(np.in1d(coord_world_geo_10km_mesh_ind, lmask_10km_1dim_ind))[0]
# lmask_1km = np.copy(lmask_init)
# lmask_1km[0, coord_world_1km_ind] = 1
# lmask_1km = lmask_1km.reshape(len(lat_world_ease_1km), len(lon_world_ease_1km))
#
# # Save variable
# os.chdir(path_workspace)
# var_name = ['coord_world_1km_ind']
# with h5py.File('coord_world_1km_ind.hdf5', 'w') as f:
#     f.create_dataset('coord_world_1km_ind', data=coord_world_1km_ind)
# f.close()

# Load in variables
os.chdir(path_workspace)
f = h5py.File("coord_world_1km_ind.hdf5", "r")
varname_list = ['coord_world_1km_ind']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()


# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2012-01-01'
end_date = '2020-12-31'
year = 2019 - 2012 + 1

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
date_seq_doy = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(date_str.strftime('%Y%m%d'))
    date_seq_doy.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))

# Count how many days for a specific year
yearname = np.linspace(2012, 2020, 39, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    f_date = datetime.date(yearname[idt], monthnum[0], 1)
    l_date = datetime.date(yearname[idt], monthnum[-1], 31)
    delta_1y = l_date - f_date
    daysofyear.append(delta_1y.days + 1)
    # print(delta_1y.days + 1)

daysofyear = np.asarray(daysofyear)


########################################################################################################################