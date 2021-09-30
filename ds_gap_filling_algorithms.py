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
import osr
import pandas as pd
import datetime
from scipy.optimize import curve_fit
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
# (Function 3) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind


#########################################################################################################################
# # (Function 4) Define a function for temporal Fourier analysis

omega = 2*np.pi/365
def tclm_model(input_x, input_y):
    ind_nonnan = np.where(~np.isnan(input_x) & ~np.isnan(input_y))[0]
    input_x_valid = input_x[ind_nonnan]
    input_y_valid = input_y[ind_nonnan]
    y_mean = np.nanmean(input_y_valid)
    def func(input_x_valid, a1, a2, a3, phi1, phi2, phi3):
        return y_mean + a1 * np.cos(np.radians(omega * input_x_valid - phi1)) + \
               a2 * np.cos(np.radians(2*omega * input_x_valid - phi2)) + \
               a3 * np.cos(np.radians(3*omega * input_x_valid - phi3))

    param, covariance = curve_fit(func, input_x_valid, input_y_valid, maxfev=10**9, ftol=1e-4, xtol=1e-4)

    y_pred = y_mean + param[0] * np.cos(np.radians(omega * input_x - param[3])) + \
               param[1] * np.cos(np.radians(2*omega * input_x - param[4])) + \
               param[2] * np.cos(np.radians(3*omega * input_x - param[5]))
    residuals = input_y - y_pred
    ss_res = np.nansum(residuals**2)
    ss_tot = np.nansum((input_y - np.nanmean(input_y))**2)
    rmse = np.sqrt(ss_tot/365)
    r_squared = 1 - (ss_res / ss_tot)

    return y_pred, r_squared, rmse

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
# Path of AMSR2 data
path_amsr2 = '/Volumes/MyPassport/SMAP_Project/Datasets/AMSR2'
# Path of 1 km MODIS LST
path_modis_1km = '/Volumes/My Book/MODIS/Model_Input/MYD11A1/'
path_modis_input = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/HDF_Data'
path_modis_input_hdf = '/Users/binfang/Downloads/Processing/SMAP_Downscale/HDF_Data'
path_modis_input_hdf_proc = '/Users/binfang/Downloads/Processing/SMAP_Downscale/HDF_Data_processed'
path_modis_input_hdf_tfa = '/Users/binfang/Downloads/Processing/SMAP_Downscale/HDF_Data_TFA'
path_modis_input_tif = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Model_Input/MYD11A1'
path_modis_lrm_output = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data/LRM_output'
path_modis_lrm_output_proc = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data/LRM_output'
path_modis_prediction = '/Users/binfang/Downloads/Processing/SMAP_Downscale/MODIS_LST_pred'
path_modis_lst = '/Volumes/WD My Book /MODIS/Model_Input/MYD11A1'
path_modis_lst_rev = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Model_Input/MYD11A1_ver2'
path_amsr2_lst_rev = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Model_Input/MYD11A1_AMSR2'

# Path of SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'

path_smap_ip = '/Users/binfang/Downloads/Processing/MODIS/Model_Output'
# Path of source output MODIS data
path_smap_op = '/Volumes/Elements/MODIS/Model_Output'

# Load in variables
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_ease_1km', 'lon_world_ease_1km', 'lat_conus_ease_1km', 'lat_conus_ease_1km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()
del (var_obj, f, varname_list)

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

cellsize_10km = 0.1
cellsize_1km = 0.01
size_world_ease_1km = np.array([14616, 34704])

# Generate 1 km lat/lon tables and corresponding row/col indices in the world lat/lon table
[lat_world_geo_10km, lon_world_geo_10km] = geo_coord_gen \
    (lat_world_max, lat_world_min, lon_world_max, lon_world_min, cellsize_10km)

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
nlpyear = 1999  # non-leap year
lpyear = 2000  # leap year
daysofmonth_nlp = np.array([calendar.monthrange(nlpyear, x)[1] for x in range(1, len(monthnum) + 1)])
ind_nlp = [np.arange(daysofmonth_nlp[0:x].sum(), daysofmonth_nlp[0:x + 1].sum()) for x in range(0, len(monthnum))]
daysofmonth_lp = np.array([calendar.monthrange(lpyear, x)[1] for x in range(1, len(monthnum) + 1)])
ind_lp = [np.arange(daysofmonth_lp[0:x].sum(), daysofmonth_lp[0:x + 1].sum()) for x in range(0, len(monthnum))]
ind_iflpr = np.array([int(calendar.isleap(yearname[x])) for x in range(len(yearname))])  # Find out leap years
# Generate a sequence of the days of months for all years
daysofmonth_seq = np.array([np.tile(daysofmonth_nlp[x], len(yearname)) for x in range(0, len(monthnum))])
daysofmonth_seq[1, :] = daysofmonth_seq[1, :] + ind_iflpr  # Add leap days to February

# print(datetime.datetime.now().strftime("%H:%M:%S"))

########################################################################################################################
# 0. Subset CONUS region from the world

# Load in the coord table variables
f = h5py.File(path_model + '/gap_filling/coord_world_1km_ind.hdf5', "r")
varname_list = list(f.keys())
varname_list = [varname_list[0], varname_list[3]]
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del (var_obj)
f.close()
del (f, varname_list)

[lat_conus_ease_1km, row_conus_1km_ind, lon_conus_ease_1km, col_conus_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      lat_conus_max, lat_conus_min, lon_conus_max, lon_conus_min)

col_conus_1km_ind, row_conus_1km_ind = np.meshgrid(col_conus_1km_ind, row_conus_1km_ind)
col_conus_1km_ind = col_conus_1km_ind.reshape(1, -1)
row_conus_1km_ind = row_conus_1km_ind.reshape(1, -1)
coord_conus_ease_1km_mesh_ind = \
    np.ravel_multi_index(np.array([row_conus_1km_ind[0], col_conus_1km_ind[0]]),
                         (len(lat_world_ease_1km), len(lon_world_ease_1km)))
# Find the matched 1km land pixels in world coord table
coord_conus_1km = np.where(np.in1d(coord_world_1km_ind, coord_conus_ease_1km_mesh_ind))[0]  # Subset
coord_conus_1km_ind = coord_world_1km_ind[coord_conus_1km]  # Original
# Find the matched land pixels in subset CONUS coord table (indexed by world coord table indices)
coord_conus_1km_match = np.where(np.in1d(coord_conus_ease_1km_mesh_ind, coord_conus_1km_ind))[0]

# Find the matched land pixels in 10 km AMSR2 coord table
coord_conus_amsr2_ind = coord_world_1km_land_ind[coord_conus_1km]  # AMSR2 coord table (original)
coord_conus_amsr2_ind_unique = np.unique(coord_conus_amsr2_ind)
coord_conus_amsr2_ind_unique_2d = \
    np.unravel_index(coord_conus_amsr2_ind_unique, (len(lat_world_geo_10km), len(lon_world_geo_10km)))
coord_conus_amsr2_2d_size = \
    np.array([len(np.unique(coord_conus_amsr2_ind_unique_2d[0])), len(np.unique(coord_conus_amsr2_ind_unique_2d[1]))])

coord_conus_amsr2_ind_match = coord_world_1km_land_ind_match[coord_conus_1km]  # AMSR2 coord table (subset)
coord_conus_amsr2_ind_unique_match = np.unique(coord_conus_amsr2_ind_match)


########################################################################################################################
# 1. Read AMSR2 Tb and MODIS LST data
modis_files_all = sorted(glob.glob(path_modis_input + '/*hdf5*'))
modis_files_month = np.array(
    [int(os.path.basename(modis_files_all[x]).split('_')[2][4:6]) for x in range(len(modis_files_all))])
modis_files_month_ind = [np.where(modis_files_month == x)[0].tolist() for x in range(1, 13)]
modis_files_year = np.array(
    [int(os.path.basename(modis_files_all[x]).split('_')[2][0:4]) for x in range(len(modis_files_all))])
modis_files_year_ind = [np.where(modis_files_year == x)[0].tolist() for x in np.unique(modis_files_year)]

# len_modis = len(coord_world_1km_group_divide[0]) * 99 + len(coord_world_1km_group_divide[-1])
# coord_world_1km_group_divide_ind_rev = \
#     np.concatenate(([0], coord_world_1km_group_divide_ind))
# coord_world_1km_group_divide_ind_rev[-1] = len_modis

# # Divide MODIS LST data into 10 blocks
modis_lst_divide_ind = \
    np.arange(0, len(coord_conus_1km), len(coord_conus_1km) // 10)[1:]
modis_lst_group_divide = np.split(np.arange(len(coord_conus_1km)), modis_lst_divide_ind)
modis_lst_group_divide[-2] = np.concatenate((modis_lst_group_divide[-2], modis_lst_group_divide[-1]))
del (modis_lst_group_divide[-1])

var_name = ['modis_lstd_all_div', 'modis_lstn_all_div']
for iyr in range(2, len(yearname) - 1):
    modis_lstd_all = []
    modis_lstn_all = []
    for imo in range(len(monthname)):
        modis_hdf_file = h5py.File(modis_files_all[modis_files_year_ind[iyr][imo]], "r")
        modis_varname_list = list(modis_hdf_file.keys())
        modis_lstd = modis_hdf_file[modis_varname_list[0]][()]
        modis_lstd = modis_lstd[:, coord_conus_1km]
        modis_lstn = modis_hdf_file[modis_varname_list[1]][()]
        modis_lstn = modis_lstn[:, coord_conus_1km]
        modis_lstd_all.append(modis_lstd)
        modis_lstn_all.append(modis_lstn)
        print(modis_hdf_file)

        del (modis_hdf_file, modis_varname_list, modis_lstd, modis_lstn)

    modis_lstd_all = np.concatenate(modis_lstd_all, axis=0)
    modis_lstn_all = np.concatenate(modis_lstn_all, axis=0)

    for ife in range(len(modis_lst_group_divide)):
        modis_lstd_all_div = modis_lstd_all[:, modis_lst_group_divide[ife]]
        modis_lstn_all_div = modis_lstn_all[:, modis_lst_group_divide[ife]]
        with h5py.File(path_modis_input_hdf + '/modis_lst_conus_' + str(ife) + '_' + str(yearname[iyr]) + '.hdf5',
                       'w') as f:
            for idv in range(len(var_name)):
                f.create_dataset(var_name[idv], data=eval(var_name[idv]), compression='lzf')
        f.close()
        print(ife)
        del (modis_lstd_all_div, modis_lstn_all_div)

    del (modis_lstd_all, modis_lstn_all)


########################################################################################################################
# 2. Read the annual MODIS data and subset CONUS region
modis_hdf_files_all = sorted(glob.glob(path_modis_input_hdf + '/*hdf5*'))

for ife in range(10):
    modis_lstd_month = []
    modis_lstn_month = []
    for iyr in range(len(yearname) - 1):
        modis_hdf_file = h5py.File(modis_hdf_files_all[iyr + ife * 5], "r")
        modis_hdf_varname_list = list(modis_hdf_file.keys())
        # 2 layers (day/night)
        modis_lstd = modis_hdf_file[modis_hdf_varname_list[0]][()]
        modis_lstn = modis_hdf_file[modis_hdf_varname_list[1]][()]
        if iyr == 0 or iyr == 4:
            modis_lstd = np.delete(modis_lstd, 59, 0)
            modis_lstn = np.delete(modis_lstn, 59, 0)

        modis_lstd_month.append(modis_lstd)
        modis_lstn_month.append(modis_lstn)
        print(modis_hdf_files_all[iyr + ife * 5])

    modis_lstd_month = np.nanmean(np.stack(modis_lstd_month, axis=0), axis=0)
    modis_lstn_month = np.nanmean(np.stack(modis_lstn_month, axis=0), axis=0)

    # Save and load the data
    with h5py.File(path_modis_input_hdf_proc + '/modis_lst_conus_' + str(ife) + '.hdf5', 'w') as f:
        f.create_dataset('modis_lstd_month', data=modis_lstd_month)
        f.create_dataset('modis_lstn_month', data=modis_lstn_month)
    f.close()


# modis_lstd_2d = np.copy(lmask_init)
# modis_lstd_2d[0, coord_conus_1km_match] = modis_lstd
# modis_lstd_2d = modis_lstd_2d.reshape((len(lat_conus_ease_1km), len(lon_conus_ease_1km)))
# modis_lstd_2d[modis_lstd_2d == 0] = np.nan
#
# amsr2_tbd_2d = np.copy(lmask_init_10km)
# amsr2_tbd_2d[0, coord_conus_amsr2_ind_unique] = amsr2_tbd_1
# amsr2_tbd_2d = amsr2_tbd_2d.reshape((len(lat_world_geo_10km), len(lon_world_geo_10km)))
# amsr2_tbd_2d[amsr2_tbd_2d == 0] = np.nan


########################################################################################################################
# 3. Read the annual AMSR2 data and subset CONUS region
# MODIS and AMSR2 HDF files have the same file name specification
amsr2_files_all = sorted(glob.glob(path_amsr2 + '/Tb_10km_monthly_land/*hdf5*'))
amsr2_files_all = amsr2_files_all[9:]

# Read AMSR2 Tb data and calculate multiple-year average Tb of one Julian year
amsr2_tbd_all = []
amsr2_tbn_all = []
for imo in range(len(monthname)):
    amsr2_tbd_month = []
    amsr2_tbn_month = []
    daysofmonth = np.arange(daysofmonth_nlp[imo])
    for iyr in range(len(yearname) - 1):
        amsr2_hdf_file = h5py.File(amsr2_files_all[imo + iyr * 12], "r")
        amsr2_varname_list = list(amsr2_hdf_file.keys())
        # 2 layers (day/night)
        amsr2_tbd = amsr2_hdf_file[amsr2_varname_list[0]][daysofmonth,]
        amsr2_tbd = amsr2_tbd[:, coord_conus_amsr2_ind_unique_match]
        amsr2_tbn = amsr2_hdf_file[amsr2_varname_list[1]][daysofmonth,]
        amsr2_tbn = amsr2_tbn[:, coord_conus_amsr2_ind_unique_match]

        amsr2_tbd_month.append(amsr2_tbd)
        amsr2_tbn_month.append(amsr2_tbn)
        del (amsr2_hdf_file, amsr2_varname_list, amsr2_tbd, amsr2_tbn)
        print(os.path.basename(amsr2_files_all[imo + iyr * 12]))

    amsr2_tbd_month = np.nanmean(np.stack(amsr2_tbd_month, axis=0), axis=0)
    amsr2_tbn_month = np.nanmean(np.stack(amsr2_tbn_month, axis=0), axis=0)

    amsr2_tbd_all.append(amsr2_tbd_month)
    amsr2_tbn_all.append(amsr2_tbn_month)
    del (amsr2_tbd_month, amsr2_tbn_month, daysofmonth)

amsr2_tbd_all = np.concatenate(amsr2_tbd_all, axis=0)
amsr2_tbn_all = np.concatenate(amsr2_tbn_all, axis=0)

# Save and load the data
with h5py.File(path_modis_input_hdf_proc + '/amsr2_tb_avg.hdf5', 'w') as f:
    f.create_dataset('amsr2_tbd_all', data=amsr2_tbd_all)
    f.create_dataset('amsr2_tbn_all', data=amsr2_tbn_all)
f.close()

########################################################################################################################
# 4. Generate TFA coefficients for AMSR2 and MODIS data

# 4.1 AMSR2 variables

f = h5py.File(path_modis_input_hdf_proc + '/amsr2_tb_avg.hdf5', "r")
varname_list = list(f.keys())
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del (var_obj)
f.close()
del(f, varname_list)

x = np.arange(365)
y_d = amsr2_tbd_all
y_n = amsr2_tbn_all

model_output_d_all = []
model_output_n_all = []
for ipt in range(y_d.shape[1]):
    model_output_d = tclm_model(x, y_d[:, ipt])
    model_output_n = tclm_model(x, y_n[:, ipt])
    model_output_d_all.append(model_output_d)
    model_output_n_all.append(model_output_n)
    del(model_output_d, model_output_n)
    print(ipt)

y_pred_d = np.stack([model_output_d_all[n][0] for n in range(y_d.shape[1])], axis=1)
r_squared_d = np.stack([model_output_d_all[n][1] for n in range(y_d.shape[1])])
rmse_d = np.stack([model_output_d_all[n][2] for n in range(y_d.shape[1])])
y_pred_n = np.stack([model_output_n_all[n][0] for n in range(y_n.shape[1])], axis=1)
r_squared_n = np.stack([model_output_n_all[n][1] for n in range(y_n.shape[1])])
rmse_n = np.stack([model_output_n_all[n][2] for n in range(y_n.shape[1])])


# Save and load the data
var_name = ['y_pred_d', 'r_squared_d', 'rmse_d', 'y_pred_n', 'r_squared_n', 'rmse_n']
with h5py.File(path_modis_input_hdf_tfa + '/amsr2_tb_tfa.hdf5', 'w') as f:
    for idv in range(len(var_name)):
        f.create_dataset(var_name[idv], data=eval(var_name[idv]), compression='lzf')
f.close()


# plt.plot(x, y_n[:, -50])
# plt.plot(x, y_pred_n[:, -50], 'r-')

# 4.2 MODIS variables

os.chdir(path_modis_input_hdf_proc)
modis_hdf_proc_files = sorted(glob.glob('*modis*'))
x = np.arange(365)
nan_array = np.empty(365)
nan_array[:] = np.nan
nan_fill = (nan_array, np.nan, np.nan)

for ife in range(9, len(modis_hdf_proc_files)):
    f = h5py.File(modis_hdf_proc_files[ife], "r")
    varname_list = list(f.keys())
    for idx in range(len(varname_list)):
        var_obj = f[varname_list[idx]][()]
        exec(varname_list[idx] + '= var_obj')
        del (var_obj)
    f.close()
    del(f, varname_list)

    y_d = modis_lstd_month
    y_n = modis_lstn_month
    del(modis_lstd_month, modis_lstn_month)

    pixel_divide_ind = \
        np.arange(0, y_d.shape[1], y_d.shape[1] // 10)[1:]
    pixel_group_divide = np.split(np.arange(y_d.shape[1]), pixel_divide_ind)
    pixel_group_divide[-2] = np.concatenate((pixel_group_divide[-2], pixel_group_divide[-1]))
    del(pixel_group_divide[-1])

    for igp in range(len(pixel_group_divide)):
        model_output_d_all = []
        model_output_n_all = []
        for ipt in range(len(pixel_group_divide[igp])):
            if len(np.where(~np.isnan(y_d[:, pixel_group_divide[igp][ipt]]))[0]) != 0 and \
                    len(np.where(~np.isnan(y_n[:, pixel_group_divide[igp][ipt]]))[0]) != 0:
                model_output_d = tclm_model(x, y_d[:, pixel_group_divide[igp][ipt]])
                model_output_n = tclm_model(x, y_n[:, pixel_group_divide[igp][ipt]])
            else:
                model_output_d = nan_fill
                model_output_n = nan_fill

            model_output_d_all.append(model_output_d)
            model_output_n_all.append(model_output_n)
            del(model_output_d, model_output_n)
            print(str(ife)+'/'+str(igp)+'/'+str(ipt))

        y_pred_d = np.stack([model_output_d_all[n][0] for n in range(len(model_output_d_all))], axis=1)
        r_squared_d = np.stack([model_output_d_all[n][1] for n in range(len(model_output_d_all))])
        rmse_d = np.stack([model_output_d_all[n][2] for n in range(len(model_output_d_all))])
        y_pred_n = np.stack([model_output_n_all[n][0] for n in range(len(model_output_n_all))], axis=1)
        r_squared_n = np.stack([model_output_n_all[n][1] for n in range(len(model_output_n_all))])
        rmse_n = np.stack([model_output_n_all[n][2] for n in range(len(model_output_n_all))])


        # Save and load the data
        var_name = ['y_pred_d', 'r_squared_d', 'rmse_d', 'y_pred_n', 'r_squared_n', 'rmse_n']
        with h5py.File(path_modis_input_hdf_tfa + '/' + os.path.basename(modis_hdf_proc_files[ife]).split('.')[0] + \
                       '_' + str(igp) + '_tfa.hdf5', 'w') as f:
            for idv in range(len(var_name)):
                f.create_dataset(var_name[idv], data=eval(var_name[idv]), compression='lzf')
        f.close()
        del(model_output_d_all, model_output_n_all, y_pred_d, r_squared_d, rmse_d, y_pred_n, r_squared_n, rmse_n)

    del(y_d, y_n)



# # Reshape to 2d matrix
# lmask_init = np.empty((len(lat_conus_ease_1km), len(lon_conus_ease_1km)), dtype='float32')
# lmask_init = lmask_init.reshape(1, -1)
# lmask_init[:] = 0
#
# lmask_init_10km = np.empty((len(lat_world_geo_10km), len(lon_world_geo_10km)), dtype='float32')
# lmask_init_10km = lmask_init_10km.reshape(1, -1)
# lmask_init_10km[:] = 0



