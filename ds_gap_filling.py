import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from netCDF4 import Dataset
import calendar
import glob
import itertools
import rasterio
import gdal
import pandas as pd
import datetime
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


########################################################################################################################
# (Function 2) Define a function to output coefficient and intercept of linear regression fit

def multi_lrm(var_x, var_y):
    nan_ind = np.where((np.isnan(var_x)) | (np.isnan(var_y)))
    var_x[nan_ind[0], nan_ind[1]] = np.nan
    var_y[nan_ind[0], nan_ind[1]] = np.nan
    slope = np.nansum((var_x - np.nanmean(var_x, axis=0)) * (var_y - np.nanmean(var_y, axis=0)), axis=0) / \
            np.nansum((var_x - np.nanmean(var_x, axis=0)) ** 2, axis=0)
    intercept = np.nanmean(var_y, axis=0) - np.nanmean(var_x, axis=0) * slope
    var_y_pred = slope * var_x + intercept
    r2 = 1 - np.nansum((var_y - var_y_pred) ** 2, axis=0) / np.nansum((var_y - np.nanmean(var_y, axis=0)) ** 2, axis=0)
    rmse = np.sqrt(np.nanmean((var_y - var_y_pred) ** 2, axis=0))

    return slope, intercept, r2, rmse

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
path_procdata = '/Volumes/Elements/processed_data'
# Path of model data
path_model = '/Volumes/Elements/processed_data'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of AMSR2 data
path_amsr2 = '/Volumes/MyPassport/SMAP_Project/Datasets/AMSR2'
# Path of 1 km MODIS LST
path_modis_1km = '/Volumes/My Book/MODIS/Model_Input/MYD11A1/'
path_modis_input = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/HDF_Data'
path_modis_lrm_output = '/Users/binfang/Downloads/Processing/SMAP_Downscale/LRM_output'
path_modis_lrm_output_proc = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data/LRM_output_processed'
path_modis_prediction = '/Users/binfang/Downloads/Processing/SMAP_Downscale/MODIS_LST_pred'
path_modis_lst = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Model_Input/MYD11A1'
path_modis_lst_rev = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Model_Input/MYD11A1_ver2'

# Path of SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'

path_smap_ip = '/Users/binfang/Downloads/Processing/MODIS/Model_Output'
# Path of source output MODIS data
path_smap_op = '/Volumes/Elements/MODIS/Model_Output'

# Load in variables
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_ease_1km', 'lon_world_ease_1km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()
del(var_obj, f, varname_list)

# Generate land/water mask provided by GLDAS/NASA
# rootgrp = Dataset(path_lmask + '/FLDAS_GLOBAL_watermask_MOD44W.nc', mode='r')
# lmask_read = rootgrp.variables['LANDMASK'][:]
# lmask_read = np.ma.getdata(np.squeeze((lmask_read)))
# lmask_read = np.flipud(lmask_read)
# rootgrp.close()
# lmask_sh = np.empty((300, lmask_read.shape[1]), dtype='float32')
# lmask_sh[:] = 0
# lmask_10km = np.concatenate((lmask_read, lmask_sh), axis=0)
# del(lmask_read, lmask_sh, rootgrp)

# World extent corner coordinates
lat_world_max = 90
lat_world_min = -90
lon_world_max = 180
lon_world_min = -180

cellsize_10km = 0.1
cellsize_1km = 0.01
size_world_ease_1km = np.array([14616, 34704])

# Generate 1 km lat/lon tables and corresponding row/col indices in the world lat/lon table
[lat_world_geo_10km, lon_world_geo_10km] = geo_coord_gen\
    (lat_world_max, lat_world_min, lon_world_max, lon_world_min, cellsize_10km)

# #----------------------------------------------------------------------------------------------------------------
# row_world_ease_1km_ind = []
# for x in range(len(lat_world_ease_1km)):
#     row_dist = np.absolute(lat_world_ease_1km[x] - lat_world_geo_10km)
#     row_ind = np.argmin(row_dist)
#     row_world_ease_1km_ind.append(row_ind)
#     del(row_ind, row_dist)
# row_world_ease_1km_ind = np.array(row_world_ease_1km_ind)
#
# col_world_ease_1km_ind = []
# for x in range(len(lon_world_ease_1km)):
#     col_dist = np.absolute(lon_world_ease_1km[x] - lon_world_geo_10km)
#     col_ind = np.argmin(col_dist)
#     col_world_ease_1km_ind.append(col_ind)
#     del(col_ind, col_dist)
# col_world_ease_1km_ind = np.array(col_world_ease_1km_ind)
#
# # Convert the match table files to 1-d linear
# col_world_ease_1km_mesh_ind, row_world_ease_1km_mesh_ind = np.meshgrid(col_world_ease_1km_ind, row_world_ease_1km_ind)
# col_world_ease_1km_mesh_ind = col_world_ease_1km_mesh_ind.reshape(1, -1)
# row_world_ease_1km_mesh_ind = row_world_ease_1km_mesh_ind.reshape(1, -1)
# coord_world_ease_1km_mesh_ind = \
#     np.ravel_multi_index(np.array([row_world_ease_1km_mesh_ind[0], col_world_ease_1km_mesh_ind[0]]), lmask_10km.shape)
#
# # Disaggregate the landmask from 10 km to 1 km of EASE grid projection
# lmask_init = np.empty(size_world_ease_1km, dtype='float32')
# lmask_init = lmask_init.reshape(1, -1)
# lmask_init[:] = 0
#
# lmask_10km_1dim = lmask_10km.reshape(1, -1)
# lmask_10km_1dim_ind = np.where(lmask_10km_1dim == 1)[1]
#
# # Find out the land pixels on the 1 km EASE projection map
# coord_world_1km_ind = np.where(np.in1d(coord_world_ease_1km_mesh_ind, lmask_10km_1dim_ind))[0]
#
# # Group the land pixels on the 1 km EASE projection map by corresponding 10-km pixels
# coord_world_1km_land_ind = coord_world_ease_1km_mesh_ind[coord_world_1km_ind]
#
# lmask_1km = np.copy(lmask_init)
# lmask_1km[0, coord_world_1km_ind] = 1
# lmask_1km = lmask_1km.reshape(size_world_ease_1km)
#
#
# # Find the matched indices for 1 km indices from 10 km AMSR2 pixels
# # coord_world_1km_land_ind_match_old = np.array([np.where(lmask_10km_1dim_ind == coord_world_1km_land_ind[x])[0][0]
# #                                            for x in range(10000)])
# coord_world_1km_land_ind_origin = np.argsort(coord_world_1km_land_ind, kind='mergesort')  # original sorted index
# coord_world_1km_land_ind_sorted = coord_world_1km_land_ind[coord_world_1km_land_ind_origin]  # ascending order sorted array
# coord_world_1km_land_ind_origin_reverse = \
#     np.argsort(coord_world_1km_land_ind_origin, kind='mergesort') # sorted index for reversing to the original
#
# coord_world_1km_land_ind_values, coord_world_1km_land_ind_ind = \
#     np.unique(coord_world_1km_land_ind_sorted, return_index=True, axis=0)
# coord_world_1km_land_ind_ind = np.concatenate((coord_world_1km_land_ind_ind, [len(coord_world_1km_land_ind_sorted)]))
# coord_world_1km_land_ind_values = np.arange(len(coord_world_1km_land_ind_values))
# coord_world_1km_land_ind_match = [np.repeat(coord_world_1km_land_ind_values[x],
#                                             coord_world_1km_land_ind_ind[x+1]-coord_world_1km_land_ind_ind[x])
#                                   for x in range(len(coord_world_1km_land_ind_values))]
# coord_world_1km_land_ind_match = np.concatenate(coord_world_1km_land_ind_match)
#
# coord_world_1km_land_ind_match = \
#     coord_world_1km_land_ind_match[coord_world_1km_land_ind_origin_reverse] # the original array
#
# # Save variable
# var_name = ['coord_world_1km_ind', 'lmask_10km_1dim_ind', 'coord_world_1km_land_ind', 'coord_world_1km_land_ind_match']
# with h5py.File(path_model + '/gap_filling/coord_world_1km_ind.hdf5', 'w') as f:
#     for x in var_name:
#         f.create_dataset(x, data=eval(x))
# f.close()
# #----------------------------------------------------------------------------------------------------------------

# # Load in variables
# f = h5py.File(path_model + '/gap_filling/coord_world_1km_ind.hdf5', "r")
# varname_list = list(f.keys())
# varname_list = varname_list[2:]
# for x in range(len(varname_list)):
#     var_obj = f[varname_list[x]][()]
#     exec(varname_list[x] + '= var_obj')
#     del(var_obj)
# f.close()
# del(f, varname_list)
#
# # Divide into 100 blocks
# coord_world_1km_group_divide_ind = \
#     np.arange(0, len(coord_world_1km_land_ind_match), len(coord_world_1km_land_ind_match) // 100)[1:]
# coord_world_1km_group_divide = np.split(coord_world_1km_land_ind_match, coord_world_1km_group_divide_ind)
# coord_world_1km_group_divide[-2] = np.concatenate((coord_world_1km_group_divide[-2], coord_world_1km_group_divide[-1]))
# del(coord_world_1km_group_divide[-1])
#
# del(coord_world_1km_land_ind_match)

# #----------------------------------------------------------------------------------------------------------------
# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2015-01-01'
end_date = '2020-12-31'
year = 2019 - 2015 + 1

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
yearname = np.linspace(2015, 2020, 6, dtype='int')
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

# Find the indices of each month in the list of days
nlpyear = 1999 # non-leap year
lpyear = 2000 # leap year
daysofmonth_nlp = np.array([calendar.monthrange(nlpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_nlp = [np.arange(daysofmonth_nlp[0:x].sum(), daysofmonth_nlp[0:x+1].sum()) for x in range(0, len(monthnum))]
daysofmonth_lp = np.array([calendar.monthrange(lpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_lp = [np.arange(daysofmonth_lp[0:x].sum(), daysofmonth_lp[0:x+1].sum()) for x in range(0, len(monthnum))]
ind_iflpr = np.array([int(calendar.isleap(yearname[x])) for x in range(len(yearname))]) # Find out leap years
# Generate a sequence of the days of months for all years
daysofmonth_seq = np.array([np.tile(daysofmonth_nlp[x], len(yearname)) for x in range(0, len(monthnum))])
daysofmonth_seq[1, :] = daysofmonth_seq[1, :] + ind_iflpr # Add leap days to February

# print(datetime.datetime.now().strftime("%H:%M:%S"))

########################################################################################################################
# 1.1 Process the AMSR2 Tb data

# date_seq_amsr2 = np.array(date_seq[182:])
amsr2_files = sorted(glob.glob(path_amsr2 + '/Tb_10km/*.h5'))
amsr2_files_doy = [amsr2_files[x].split('/')[-1].split('_')[1] for x in range(len(amsr2_files))]
amsr2_files_month = [amsr2_files_doy[x][0:6] for x in range(len(amsr2_files_doy))]
amsr2_files_month_unique = np.unique(amsr2_files_month)
amsr2_doy_split = np.split(amsr2_files_doy, np.unique(amsr2_files_month, return_index=True)[1][1:])
amsr2_files_doy_split = np.split(amsr2_files, np.unique(amsr2_files_month, return_index=True)[1][1:])

month_ind = daysofmonth_seq.reshape((-1, 1), order='F').ravel()[3:]
month_split_ind = list(np.cumsum(daysofmonth_seq.reshape((-1, 1), order='F').ravel()[3:]))

matsize_amsr2_10km = [len(lat_world_geo_10km), len(lon_world_geo_10km)]
matsize_modis_1km = [len(lat_world_ease_1km), len(lon_world_ease_1km)]
amsr2_10km_empty = np.empty(matsize_amsr2_10km, dtype='float32')
amsr2_10km_empty[:] = np.nan

for imo in range(len(amsr2_files_month_unique)): # Start from April, 2015

    amsr2_doy_split_unique = np.unique(amsr2_doy_split[imo])
    amsr2_doy_split_ind = [int(amsr2_doy_split_unique[x][-2:]) - 1 for x in range(len(amsr2_doy_split_unique))]
    amsr2_doy_split_full_ind = np.arange(month_ind[imo])
    amsr2_doy_file_ind = [np.where(amsr2_doy_split_full_ind[x] == amsr2_doy_split_ind)[0] for x in range(len(amsr2_doy_split_full_ind))]

    tbv_1month_a = []
    tbv_1month_d = []
    for idt in range(len(amsr2_doy_split_full_ind)):
        if idt in amsr2_doy_split_ind:
            # Ascending overpass
            rootgrp_a = Dataset(amsr2_files_doy_split[imo][amsr2_doy_file_ind[idt][0]*2], mode='r')
            tbv_a = rootgrp_a.variables['Brightness Temperature (V)'][:]
            tbv_a = np.array(tbv_a)
            tbv_a[np.where(tbv_a > 65531)] = 0
            tbv_a = tbv_a * 0.01
            tbv_a = np.concatenate((tbv_a[:, 1800:], tbv_a[:, 0:1800]), axis=1)
            tbv_a = tbv_a * lmask_10km
            tbv_a[tbv_a == 0] = np.nan

            # Descending overpass
            rootgrp_d = Dataset(amsr2_files_doy_split[imo][amsr2_doy_file_ind[idt][0]*2+1], mode='r')
            tbv_d = rootgrp_d.variables['Brightness Temperature (V)'][:]
            tbv_d = np.array(tbv_d)
            tbv_d[np.where(tbv_d > 65531)] = 0
            tbv_d = tbv_d * 0.01
            tbv_d = np.concatenate((tbv_d[:, 1800:], tbv_d[:, 0:1800]), axis=1)
            tbv_d = tbv_d * lmask_10km
            tbv_d[tbv_d == 0] = np.nan

            print(amsr2_files_doy_split[imo][amsr2_doy_file_ind[idt][0]*2])

        else:
            tbv_a = amsr2_10km_empty
            tbv_d = amsr2_10km_empty

        tbv_1month_a.append(tbv_a)
        tbv_1month_d.append(tbv_d)
        del(tbv_a, tbv_d)

    tbv_1month_a = np.array(tbv_1month_a)
    # tbv_1month_a = np.transpose(tbv_1month_a, (1, 2, 0))
    tbv_1month_d = np.array(tbv_1month_d)
    # tbv_1month_d = np.transpose(tbv_1month_d, (1, 2, 0))

    # Save AMSR2 data
    # os.chdir(path_model)
    var_name = ['amsr2_tb_a_' + str(amsr2_files_month_unique[imo]), 'amsr2_tb_d_' + str(amsr2_files_month_unique[imo]),
                'lat_world_geo_10km', 'lon_world_geo_10km']
    data_name = ['tbv_1month_a', 'tbv_1month_d', 'lat_world_geo_10km', 'lon_world_geo_10km']

    with h5py.File(path_amsr2 + '/Tb_10km_monthly/amsr2_tb_' + str(amsr2_files_month_unique[imo]) + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]), compression='lzf')
    f.close()

    del(tbv_1month_a, tbv_1month_d, amsr2_doy_split_unique, amsr2_doy_split_ind, amsr2_doy_split_full_ind,
        amsr2_doy_file_ind, var_name, data_name)


# 1.2 Extract only land pixels from AMSR2
amsr2_hdf_files_all = sorted(glob.glob(path_amsr2 + '/Tb_10km_monthly/*.hdf5'))

for imo in range(len(amsr2_hdf_files_all)):
    amsr2_hdf_file = h5py.File(amsr2_hdf_files_all[imo], "r")
    amsr2_file_name = os.path.basename(amsr2_hdf_files_all[imo]).split('_')[-1][0:6]
    amsr2_varname_list = list(amsr2_hdf_file.keys())
    # 2 layers (day/night)
    amsr2_tbd = amsr2_hdf_file[amsr2_varname_list[0]][()]
    amsr2_tbn = amsr2_hdf_file[amsr2_varname_list[1]][()]

    matsize_amsr2 = amsr2_tbd.shape
    amsr2_tbd = np.reshape(amsr2_tbd, (matsize_amsr2[0], (matsize_amsr2[1]*matsize_amsr2[2])))
    amsr2_tbn = np.reshape(amsr2_tbn, (matsize_amsr2[0], (matsize_amsr2[1]*matsize_amsr2[2])))

    amsr2_tbd = amsr2_tbd[:, lmask_10km_1dim_ind]
    amsr2_tbn = amsr2_tbn[:, lmask_10km_1dim_ind]

    # Save to hdf file
    var_name = ['amsr2_tbd_' + amsr2_file_name, 'amsr2_tbn_' + amsr2_file_name]
    data_name = ['amsr2_tbd', 'amsr2_tbn']

    with h5py.File(path_amsr2 + '/Tb_10km_monthly_land/amsr2_tb_' + amsr2_file_name + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]), compression='lzf')
    f.close()

    print(amsr2_file_name)
    del(amsr2_hdf_file, amsr2_file_name, amsr2_varname_list, matsize_amsr2, amsr2_tbd, amsr2_tbn, var_name, data_name)


########################################################################################################################
# 2.  Extract the land pixels of the 1 km MODIS LST and save to hdf files.
date_seq = date_seq[90:]
date_seq_doy = date_seq_doy[90:]
month_ind = daysofmonth_seq.reshape((-1, 1), order='F').ravel()[3:]
month_split = np.array(np.linspace(0, 72, 7), dtype='int') - 3
month_split = month_split[1:]
month_split_ind = np.split(month_ind, month_split)[:-1]
date_seq_month = [date_seq[x][0:6] for x in range(len(date_seq))]
date_seq_month_unique = np.unique(date_seq_month)
date_seq_month_ind = np.split(date_seq_month_unique, month_split)[:-1]


for iyr in range(len(yearname)):
    os.chdir(path_modis_1km + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    month_split_ind_1year = np.cumsum(month_split_ind[iyr])
    tif_files_split = np.split(tif_files, month_split_ind_1year)[:-1]

    for imo in range(len(month_split_ind_1year)):

        src_tf_land_1month_day = []
        src_tf_land_1month_night = []
        for idt in range(len(tif_files_split[imo])):
            src_tf = rasterio.open(tif_files_split[imo][idt]).read()
            src_tf = np.reshape(src_tf, (2, matsize_modis_1km[0]*matsize_modis_1km[1]))
            src_tf_land = src_tf[:, coord_world_1km_ind]
            src_tf_land_1month_day.append(src_tf_land[0, :])
            src_tf_land_1month_night.append(src_tf_land[1, :])
            del(src_tf, src_tf_land)
            print(tif_files_split[imo][idt])
        src_tf_land_1month_day = np.array(src_tf_land_1month_day, dtype='float32')
        src_tf_land_1month_night = np.array(src_tf_land_1month_night, dtype='float32')

        # Save monthly MODIS LST land pixels to hdf file
        var_name = ['modis_lst_day_' + str(date_seq_month_ind[iyr][imo]),
                    'modis_lst_night_' + str(date_seq_month_ind[iyr][imo])]
        data_name = ['src_tf_land_1month_day', 'src_tf_land_1month_night']

        with h5py.File(path_modis_input + '/modis_lst_' + str(date_seq_month_ind[iyr][imo]) + '.hdf5', 'w') as f:
            for idv in range(len(var_name)):
                f.create_dataset(var_name[idv], data=eval(data_name[idv]), compression='lzf')
        f.close()

        del(src_tf_land_1month_day, src_tf_land_1month_night, var_name, data_name)

    del(tif_files, month_split_ind_1year, tif_files_split)


########################################################################################################################
# 3. Use AMSR2 Tb and MODIS LST data to build linear regression model

modis_files_all = sorted(glob.glob(path_modis_input + '/*hdf5*'))
modis_files_month = np.array([int(os.path.basename(modis_files_all[x]).split('_')[2][4:6]) for x in range(len(modis_files_all))])
modis_files_month_ind = [np.where(modis_files_month == x)[0].tolist() for x in range(1, 13)]
# MODIS and AMSR2 HDF files have the same file name specification
amsr2_files_all = sorted(glob.glob(path_amsr2 + '/Tb_10km_monthly_land/*hdf5*'))
len_modis = len(coord_world_1km_group_divide[0]) * 99 + len(coord_world_1km_group_divide[-1])
coord_world_1km_group_divide_ind_rev = \
    np.concatenate(([0], coord_world_1km_group_divide_ind))
coord_world_1km_group_divide_ind_rev[-1] = len_modis

for imo in range(len(monthname)):

    # Read AMSR2 Tb data
    amsr2_tbd_all = []
    amsr2_tbn_all = []
    for ife in range(len(modis_files_month_ind[imo])):
        amsr2_hdf_file = h5py.File(amsr2_files_all[modis_files_month_ind[imo][ife]], "r")
        amsr2_varname_list = list(amsr2_hdf_file.keys())
        # 2 layers (day/night)
        # Ascending (1:30 PM) -> day
        amsr2_tbd = amsr2_hdf_file[amsr2_varname_list[0]][()]
        # Descending (1:30 AM) -> night
        amsr2_tbn = amsr2_hdf_file[amsr2_varname_list[1]][()]
        amsr2_hdf_file.close()
        amsr2_tbd_all.append(amsr2_tbd)
        amsr2_tbn_all.append(amsr2_tbn)
        print(os.path.basename(amsr2_files_all[modis_files_month_ind[imo][ife]]))
        del(amsr2_tbd, amsr2_tbn)

    amsr2_tbd_all = np.concatenate(amsr2_tbd_all, axis=0)
    amsr2_tbn_all = np.concatenate(amsr2_tbn_all, axis=0)

    for idx in range(len(coord_world_1km_group_divide_ind_rev)-1):  # Divide and load in the hdf file by 100 groups

        # print(datetime.datetime.now().strftime("%H:%M:%S"))
        coord_world_1km_group_ind_sub = \
            np.arange(coord_world_1km_group_divide_ind_rev[idx], coord_world_1km_group_divide_ind_rev[idx+1])

        # Read MODIS LST data
        modis_lstd_all = []
        modis_lstn_all = []
        for ife in range(len(modis_files_month_ind[imo])):
            modis_hdf_file = h5py.File(modis_files_all[modis_files_month_ind[imo][ife]], "r")
            modis_varname_list = list(modis_hdf_file.keys())
            # 2 layers (day/night)
            modis_lstd = modis_hdf_file[modis_varname_list[0]][:, coord_world_1km_group_ind_sub]
            modis_lstn = modis_hdf_file[modis_varname_list[1]][:, coord_world_1km_group_ind_sub]
            modis_hdf_file.close()
            modis_lstd_all.append(modis_lstd)
            modis_lstn_all.append(modis_lstn)
            print(os.path.basename(modis_files_all[modis_files_month_ind[imo][ife]]) + ' / ' + str(idx+1))
            del(modis_lstd, modis_lstn)

        modis_lstd_all = np.concatenate(modis_lstd_all, axis=0)
        modis_lstn_all = np.concatenate(modis_lstn_all, axis=0)


        # Extrct the AMSR2 land pixels
        amsr2_tbd_lr = amsr2_tbd_all[:, coord_world_1km_group_divide[idx]]
        amsr2_tbn_lr = amsr2_tbn_all[:, coord_world_1km_group_divide[idx]]

        # Build the linear regression between MODIS/AMSR2 data
        lrm_day = multi_lrm(amsr2_tbd_lr, modis_lstd_all)
        lrm_night = multi_lrm(amsr2_tbn_lr, modis_lstn_all)

        # Save monthly linear regression model variables to hdf file
        var_name = ['lrm_day_' + monthname[imo] + '_' + str(idx+1),
                    'lrm_night_' + monthname[imo] + '_' + str(idx+1)]
        data_name = ['lrm_day', 'lrm_night']

        with h5py.File(path_modis_lrm_output + '/lrm_' + monthname[imo] + '_' + str(idx+1) + '.hdf5', 'w') as f:
            for idv in range(len(var_name)):
                f.create_dataset(var_name[idv], data=eval(data_name[idv]), compression='lzf')
        f.close()

        print('lrm_' + monthname[imo] + '_' + str(idx+1))
        del(coord_world_1km_group_ind_sub, modis_lstd_all, modis_lstn_all, amsr2_tbd_lr, amsr2_tbn_lr,
            lrm_day, lrm_night, var_name, data_name)
        # print(datetime.datetime.now().strftime("%H:%M:%S"))

    del(amsr2_tbd_all, amsr2_tbn_all)


########################################################################################################################
# 4. Combine and apply the linear regression model coefficients to generate 1 km LST from AMSR2 data

# Load in variables
f = h5py.File(path_model + '/gap_filling/coord_world_1km_ind.hdf5', "r")
varname_list = list(f.keys())
varname_list = [varname_list[0], varname_list[3]]
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()
del(f, varname_list)


# 4.1 Combine the linear regression model coefficients to one file
lrm_files_all = sorted(glob.glob(path_modis_lrm_output + '/*hdf5*'))
lrm_name = np.array([int(os.path.basename(lrm_files_all[x]).split('.')[0].split('_')[1] +
            os.path.basename(lrm_files_all[x]).split('.')[0].split('_')[2].zfill(3))
            for x in range(len(lrm_files_all))])
lrm_name_ind = np.argsort(lrm_name, kind='mergesort')
lrm_files_all = [lrm_files_all[lrm_name_ind[x]] for x in range(len(lrm_name_ind))]

lrm_files_month = np.array([int(os.path.basename(lrm_files_all[x]).split('_')[1]) for x in range(len(lrm_files_all))])
lrm_files_month_ind = [np.where(lrm_files_month == x)[0].tolist() for x in range(1, 13)]

for imo in range(len(lrm_files_month_ind)):

    lrm_day_coef_all = []
    lrm_day_stats_all = []
    lrm_night_coef_all = []
    lrm_night_stats_all = []
    for ife in range(len(lrm_files_month_ind[imo])):
        lrm_file = h5py.File(lrm_files_all[lrm_files_month_ind[imo][ife]], "r")
        lrm_varname_list = list(lrm_file.keys())
        # 2 layers (day/night)
        lrm_day_coef = lrm_file[lrm_varname_list[0]][:2, :]
        lrm_day_stats = lrm_file[lrm_varname_list[0]][2:, :]
        lrm_night_coef = lrm_file[lrm_varname_list[1]][:2, :]
        lrm_night_stats = lrm_file[lrm_varname_list[1]][2:, :]
        lrm_day_coef_all.append(lrm_day_coef)
        lrm_day_stats_all.append(lrm_day_stats)
        lrm_night_coef_all.append(lrm_night_coef)
        lrm_night_stats_all.append(lrm_night_stats)
        del(lrm_day_coef, lrm_day_stats, lrm_night_coef, lrm_night_stats)
        print(lrm_files_all[lrm_files_month_ind[imo][ife]])

    lrm_day_coef_all = np.concatenate(lrm_day_coef_all, axis=1)
    lrm_day_stats_all = np.concatenate(lrm_day_stats_all, axis=1)
    lrm_night_coef_all = np.concatenate(lrm_night_coef_all, axis=1)
    lrm_night_stats_all = np.concatenate(lrm_night_stats_all, axis=1)

    # Save variables to hdf file
    var_name = ['lrm_day_coef_' + monthname[imo],
                'lrm_day_stats_' + monthname[imo],
                'lrm_night_coef_' + monthname[imo],
                'lrm_night_stats_' + monthname[imo]]
    data_name = ['lrm_day_coef_all', 'lrm_day_stats_all', 'lrm_night_coef_all', 'lrm_night_stats_all']

    with h5py.File(path_modis_lrm_output_proc + '/lrm_' + monthname[imo] + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]), compression='lzf')
    f.close()
    del(lrm_day_coef_all, lrm_day_stats_all, lrm_night_coef_all, lrm_night_stats_all)


########################################################################################################################
# 4.2 Use the linear regression model coefficients to calculate 1 km LST from AMSR2 data

# Load in variables
f = h5py.File(path_model + '/gap_filling/coord_world_1km_ind.hdf5', "r")
varname_list = list(f.keys())
varname_list = [varname_list[0], varname_list[2]]
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()
del(f, varname_list)

lmask_init = np.empty(size_world_ease_1km, dtype='float32')
lmask_init = lmask_init.reshape(1, -1)
lmask_init[:] = 0

# AMSR2 file name indices
amsr2_files_all = sorted(glob.glob(path_amsr2 + '/Tb_10km_monthly_land/*hdf5*'))
amsr2_files_name = np.array([int(os.path.basename(amsr2_files_all[x]).split('_')[-1][0:6]) for x in range(len(amsr2_files_all))])
amsr2_files_month = np.array([int(os.path.basename(amsr2_files_all[x]).split('_')[-1][4:6]) for x in range(len(amsr2_files_all))])
amsr2_files_month_ind = [np.where(amsr2_files_month == x)[0].tolist() for x in range(1, 13)]

lrm_files = sorted(glob.glob(path_modis_lrm_output_proc + '/*hdf5*'))

# MODIS file name indices
modis_files_all = []
for iyr in range(len(yearname)):
    modis_files = sorted(glob.glob(path_modis_lst + '/' + str(yearname[iyr]) + '/*.tif'))
    modis_files_all.append(modis_files)
    del(modis_files)
modis_files_all = list(itertools.chain(*modis_files_all))
modis_files_name = [modis_files_all[x].split('/')[-1].split('_')[-1][0:7] for x in range(len(modis_files_all))]
modis_files_month = np.array([(datetime.datetime(int(modis_files_name[x][0:4]), 1, 1) +
                              datetime.timedelta(int(modis_files_name[x][4:])-1)).month
                             for x in range(len(modis_files_name))])
modis_files_day = np.array([(datetime.datetime(int(modis_files_name[x][0:4]), 1, 1) +
                              datetime.timedelta(int(modis_files_name[x][4:])-1)).day
                             for x in range(len(modis_files_name))])
modis_files_yearmonth = np.array([int(modis_files_name[x][0:4] + str(modis_files_month[x]).zfill(2))
                                 for x in range(len(modis_files_name))])
modis_files_yearmonthday = [modis_files_name[x][0:4] + str(modis_files_month[x]).zfill(2) +
                                         str(modis_files_day[x]).zfill(2) for x in range(len(modis_files_name))]
modis_files_month_ind = [np.where(modis_files_yearmonth == amsr2_files_name[x])[0].tolist() for x in range(len(amsr2_files_name))]

modis_files_ind = []
for imo in range(len(amsr2_files_month_ind)):
    modis_files_ind_1month = [modis_files_month_ind[amsr2_files_month_ind[imo][x]] for x in range(len(amsr2_files_month_ind[imo]))]
    # modis_files_ind_1month = list(itertools.chain(*modis_files_ind_1month))
    modis_files_ind.append(modis_files_ind_1month)
    del(modis_files_ind_1month)

# Georeference information of tiff file
kwargs = rasterio.open(modis_files_all[0]).meta.copy()
kwargs.update(compress='lzw')


# Generate the AMSR2 Tb data derived LST and use to gap-fill MODIS LST
for imo in [9]:#range(len(monthname)):

    # Read linear regression model data
    lrm_hdf_file = h5py.File(lrm_files[imo], "r")
    lrm_varname_list = list(lrm_hdf_file.keys())
    slope_d = lrm_hdf_file[lrm_varname_list[0]][0, :]
    intercept_d = lrm_hdf_file[lrm_varname_list[0]][1, :]
    slope_n = lrm_hdf_file[lrm_varname_list[2]][0, :]
    intercept_n = lrm_hdf_file[lrm_varname_list[2]][1, :]
    lrm_hdf_file.close()

    # Read MODIS LST data and AMSR2 Tb data
    for ife in [1]:#range(len(amsr2_files_month_ind[imo])):

        amsr2_hdf_file = h5py.File(amsr2_files_all[amsr2_files_month_ind[imo][ife]], "r")
        amsr2_varname_list = list(amsr2_hdf_file.keys())

        for idt in [0]:#range(len(modis_files_ind[imo][ife])):

            # MODIS LST
            tif_file = rasterio.open(modis_files_all[modis_files_ind[imo][ife][idt]]).read()
            tif_file_day = int(modis_files_yearmonthday[modis_files_ind[imo][ife][idt]][-2:])
            tif_file_name = os.path.basename(modis_files_all[modis_files_ind[imo][ife][idt]]).split('_')[3][:7]

            modis_lstd = tif_file[0, :, :].reshape(1, -1)
            modis_lstd = modis_lstd[0, coord_world_1km_ind]
            modis_lstn = tif_file[1, :, :].reshape(1, -1)
            modis_lstn = modis_lstn[0, coord_world_1km_ind]

            modis_lstd_nan_ind = np.where(np.isnan(modis_lstd))
            modis_lstn_nan_ind = np.where(np.isnan(modis_lstn))


            # 2 layers (day/night)
            # Ascending (1:30 PM) -> day
            amsr2_tbd = amsr2_hdf_file[amsr2_varname_list[0]][tif_file_day-1, :] # index-1
            amsr2_tbd_exp = amsr2_tbd[coord_world_1km_land_ind_match]
            # Descending (1:30 AM) -> night
            amsr2_tbn = amsr2_hdf_file[amsr2_varname_list[1]][tif_file_day-1, :]
            amsr2_tbn_exp = amsr2_tbn[coord_world_1km_land_ind_match]


            amsr2_lstd_pred = slope_d * amsr2_tbd_exp + intercept_d
            amsr2_lstn_pred = slope_n * amsr2_tbn_exp + intercept_n

            # Apply the gap-filled LST pixels from AMSR2 to MODIS
            modis_lstd[modis_lstd_nan_ind] = amsr2_lstd_pred[modis_lstd_nan_ind]
            modis_lstn[modis_lstn_nan_ind] = amsr2_lstn_pred[modis_lstn_nan_ind]


            modis_lstd_filled = np.copy(lmask_init)
            modis_lstd_filled[0, coord_world_1km_ind] = modis_lstd
            modis_lstd_filled = modis_lstd_filled.reshape(size_world_ease_1km)
            modis_lstd_filled[modis_lstd_filled > 363] = np.nan
            modis_lstd_filled[modis_lstd_filled < 203] = np.nan

            modis_lstn_filled = np.copy(lmask_init)
            modis_lstn_filled[0, coord_world_1km_ind] = modis_lstn
            modis_lstn_filled = modis_lstn_filled.reshape(size_world_ease_1km)
            modis_lstn_filled[modis_lstn_filled > 363] = np.nan
            modis_lstn_filled[modis_lstn_filled < 203] = np.nan

            modis_lst_filled = np.stack((modis_lstd_filled, modis_lstn_filled))
            print(tif_file_name)

            # Save the MODIS LST data to Geotiff files
            input_ds = rasterio.open(path_modis_lst_rev + '/modis_lst_1km_' + tif_file_name + '.tif', 'w', **kwargs)
            for ilr in range(2):
                input_ds.write(modis_lst_filled[ilr, :, :], ilr+1)
            input_ds = None

            del(tif_file, tif_file_day, tif_file_name, modis_lstd, modis_lstn, modis_lstd_nan_ind, modis_lstn_nan_ind,
                amsr2_tbd, amsr2_tbd_exp, amsr2_tbn, amsr2_tbn_exp, amsr2_lstd_pred, amsr2_lstn_pred, modis_lstd_filled,
                modis_lstn_filled, modis_lst_filled, input_ds)

        amsr2_hdf_file.close()
        del(amsr2_hdf_file, amsr2_varname_list)

    del(lrm_hdf_file, lrm_varname_list, slope_d, intercept_d, slope_n, intercept_n)



    # Build output path
    # band_path = os.path.join(path_modis_op, os.path.basename(os.path.splitext(hdf_files)[0]) + "-ctd" + ".tif")
    # Write raster

    band_ds = gdal.Open(modis_files_all[0], gdal.GA_ReadOnly)
    out_ds = gdal.GetDriverByName('MEM').Create('', band_ds.RasterXSize, band_ds.RasterYSize, band_n, #Number of bands
                                  gdal.GDT_Float32)
    out_ds.SetGeoTransform(band_ds.GetGeoTransform())
    out_ds.SetProjection(band_ds.GetProjection())

    # Loop write each band to Geotiff file
    for idb in range(len(subdataset_id)//2):
        out_ds.GetRasterBand(idb+1).WriteArray(band_array[:, :, idb])
        out_ds.GetRasterBand(idb+1).SetNoDataValue(0)
    # out_ds = None  #close dataset to write to disc



# file = h5py.File('/Users/binfang/Downloads/Processing/SMAP_Downscale/LRM_output/lrm_08_20.hdf5', "r")
# var = list(file.keys())
# lrmd = file[var[0]][()]
# # plt.plot(lrmd[2,:])
# plt.hist(lrmd[2, :].ravel())


########################################################################################################################
# 5. Convert the Geo-tiff files

for iyr in range(len(yearname)):
    for idt in range(daysofyear[iyr]):

        filename = 'smap_sm_1km_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '.tif'
        ds_sm_input = gdal.Open(path_smap_ip + '/' + str(yearname[iyr]) + '/' + filename)
        ds_sm = ds_sm_input.ReadAsArray()

        # Create a raster of EASE grid projection at 1 km resolution
        out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_smap_op + '/' + str(yearname[iyr]) + '/' + filename,
                                                           ds_sm_input.RasterXSize, ds_sm_input.RasterYSize,
                                                           ds_sm_input.RasterCount, gdal.GDT_Float32,
                                                           ['COMPRESS=LZW', 'TILED=YES'])
        out_ds_tiff.SetGeoTransform(ds_sm_input.GetGeoTransform())
        out_ds_tiff.SetProjection(ds_sm_input.GetProjection())

        # Write each band to Geotiff file
        for ilr in range(2):
            out_ds_tiff.GetRasterBand(ilr+1).WriteArray(ds_sm[ilr, :, :])
            out_ds_tiff.GetRasterBand(ilr+1).SetNoDataValue(0)
        out_ds_tiff = None  # close dataset to write to disc
        print(filename)
        del(filename, ds_sm_input, ds_sm, out_ds_tiff)