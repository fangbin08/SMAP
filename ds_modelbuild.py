import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import statsmodels.api as sm
from netCDF4 import Dataset
import calendar
import datetime
import glob
import gdal
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

############################################################################################################################
# (Function 1) Find out the correct corresponding UTC time for each time zone for one specific local time (1:00 a.m., e.g.),
# by using the different hours between local time and UTC time

def timezone_converter(var_name, ind_timezone):
    var_name_utct = var_name - ind_timezone # Calculate the UTC time (it may be out of 24-hour range)
    # Find if the correct file is from today or yesterday/tomorrow
    var_name_utct_day = np.array(np.zeros(len(var_name_utct), ), dtype=int)
    var_name_utct_day[np.where(var_name_utct < 0)] = -1  # From yesterday
    var_name_utct_day[np.where(var_name_utct >= 24)] = 1  # From tomorrow

    # Correct the UTC time to 24-hour range for locating the corresponding hdf file
    var_name_utct_ind = np.copy(var_name_utct)
    var_name_utct_ind[np.where(var_name_utct < 0)] = \
        var_name_utct_ind[np.where(var_name_utct < 0)] + 24  # From yesterday
    var_name_utct_ind[np.where(var_name_utct >= 24)] = \
        var_name_utct_ind[np.where(var_name_utct >= 24)] - 24  # From tomorrow

    return var_name_utct_day, var_name_utct, var_name_utct_ind

########################################################################################################################
# (Function 2) Find out the corresponding file of the closest UTC time from GLDAS data collection for any UTC times

def gldas_filefinder(var_name_utct_ind, timestamp_gldas):
    var_name_gldas_ind = np.copy(var_name_utct_ind)
    for tm in var_name_utct_ind:
        if var_name_utct_ind[tm] == 23: # Assign the UTC time = 23:00 to 0:00
            time_min = np.absolute(var_name_utct_ind[tm]-24 - timestamp_gldas)
            var_name_gldas_ind[tm] = timestamp_gldas[np.argmin(time_min)]
        else:
            time_min = np.absolute(var_name_utct_ind[tm] - timestamp_gldas)
            var_name_gldas_ind[tm] = timestamp_gldas[np.argmin(time_min)]

    return var_name_gldas_ind

##############################################################################################################
# (Function 3) Generate lat/lon tables of Geographic projection in world/CONUS

def geo_coord_gen(lat_geo_extent_max, lat_geo_extent_min, lon_geo_extent_max, lon_geo_extent_min, cellsize):
    lat_geo_output = np.linspace(lat_geo_extent_max - cellsize / 2, lat_geo_extent_min + cellsize / 2,
                                    num=int((lat_geo_extent_max - lat_geo_extent_min) / cellsize))
    lat_geo_output = np.round(lat_geo_output, decimals=3)
    lon_geo_output = np.linspace(lon_geo_extent_min + cellsize / 2, lon_geo_extent_max - cellsize / 2,
                                    num=int((lon_geo_extent_max - lon_geo_extent_min) / cellsize))
    lon_geo_output = np.round(lon_geo_output, decimals=3)

    return lat_geo_output, lon_geo_output

##############################################################################################################
# (Function 4) Define a function to output coefficient and intercept of linear regression fit

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
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of source GLDAS data
path_gldas = '/Volumes/MyPassport/SMAP_Project/Datasets/GLDAS'
# Path of source LTDR NDVI data
path_ltdr = '/Volumes/MyPassport/SMAP_Project/Datasets/LTDR/Ver5'
# Path of processed data
path_procdata = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
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
os.chdir(path_lmask)

lmask_file = open('EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin', 'r')
lmask_ease_25km = np.fromfile(lmask_file, dtype=np.dtype('uint8'))
lmask_ease_25km = np.reshape(lmask_ease_25km, [len(lat_world_ease_25km), len(lon_world_ease_25km)]).astype(float)
lmask_ease_25km[np.where(lmask_ease_25km != 0)] = np.nan
lmask_ease_25km[np.where(lmask_ease_25km == 0)] = 1
# lmask_ease_25km[np.where((lmask_ease_25km == 101) | (lmask_ease_25km == 255))] = 0
lmask_file.close()

# World extent corner coordinates
lat_world_max = 90
lat_world_min = -90
lon_world_max = 180
lon_world_min = -180
# GLDAS extent corner coordinates
lat_gldas_max = 90
lat_gldas_min = -60
lon_gldas_max = 180
lon_gldas_min = -180
n_timezone = 24
step_timezone = 15
ind_timezone_min = -12
ind_timezone_max = 11
lst_am = 1 # 1:00
lst_pm = 13 # 13:00
sm_am = 6 # 6:00
sm_pm = 18 # 18:00
cellsize_25km = 0.25

# Generate GLDAS extent lat/lon tables and corresponding row indices in the world lat table
[lat_gldas_geo_25km, lon_gldas_geo_25km] = geo_coord_gen\
    (lat_gldas_max, lat_gldas_min, lon_gldas_max, lon_gldas_min, cellsize_25km)
row_gldas_geo_25km_ind = np.array(np.where((lat_world_geo_25km <= lat_gldas_max) &
                          (lat_world_geo_25km >= lat_gldas_min))).flatten()

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '1981-01-01'
end_date = '2018-12-31'
year = 2018 - 1981 + 1

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
yearname = np.linspace(1981, 2018, 38, dtype='int')
monthname = np.linspace(1, 12, 12, dtype='int')
monthnum = np.arange(1, 13)
monthnum = [str(i).zfill(2) for i in monthnum]

daysofyear = []
for idt in range(len(yearname)):
    f_date = datetime.date(yearname[idt], monthname[0], 1)
    l_date = datetime.date(yearname[idt], monthname[-1], 31)
    delta_1y = l_date - f_date
    daysofyear.append(delta_1y.days + 1)
    # print(delta_1y.days + 1)

daysofyear = np.asarray(daysofyear)


########################################################################################################################

# 1. Time zone conversions
# Longitudes for 15° apart time zones (TZ)
lon_timezone = np.linspace(lon_world_min, lon_world_max, num=n_timezone+1, dtype=float)
lon_timezone = lon_timezone + step_timezone/2 # The UTC timezone starts from 7.5°W
lon_timezone = lon_timezone[0:-1] # Delete the last element
lon_timezone_center = lon_timezone - step_timezone/2
lon_timezone_center = np.append(lon_timezone_center, 180) # Add one index for latitudes near -180°

# Index of time zones
ind_timezone = np.linspace(ind_timezone_min, ind_timezone_max, num=n_timezone, dtype=int)
ind_timezone_p1 = np.append(ind_timezone, 12) # Add one index for latitudes near -180°

# Find the corresponding UTC time zone for each latitude in the lat table
row_gldas_geo_25km_tz_ind = []
for i in range(lon_gldas_geo_25km.size):
    ind_min = np.absolute(lon_gldas_geo_25km[i] - lon_timezone_center)
    row_gldas_geo_25km_tz_ind.append(ind_timezone_p1[np.argmin(ind_min)])

row_gldas_geo_25km_tz_ind = np.asarray(row_gldas_geo_25km_tz_ind)
row_gldas_geo_25km_tz_ind[np.where(row_gldas_geo_25km_tz_ind == 12)] = -12

# Group the latitudes by time zone indices
row_gldas_geo_25km_tz_ind_group = \
    [np.where(row_gldas_geo_25km_tz_ind == ind_timezone_p1[x]) for x in range(len(ind_timezone_p1)-1)]

# Find out the correct corresponding UTC time (UTCT) for each time zone for specific local times
[lst_am_utct_day, lst_am_utct, lst_am_utct_ind] = timezone_converter(lst_am, ind_timezone)
[lst_pm_utct_day, lst_pm_utct, lst_pm_utct_ind] = timezone_converter(lst_pm, ind_timezone)
[sm_am_utct_day, sm_am_utct, sm_am_utct_ind] = timezone_converter(sm_am, ind_timezone)
[sm_pm_utct_day, sm_pm_utct, sm_pm_utct_ind] = timezone_converter(sm_pm, ind_timezone)

# GLDAS V2 data has 3-hourly time interval. So 8 files for each day.
# Find out the corresponding file of the closest UTC time from GLDAS data collection for any UTC times
timestamp_gldas = np.linspace(0, 24, num=8+1, dtype=int)
timestamp_gldas = timestamp_gldas[0:-1]

lst_am_gldas_ind = gldas_filefinder(lst_am_utct_ind, timestamp_gldas)
lst_pm_gldas_ind = gldas_filefinder(lst_pm_utct_ind, timestamp_gldas)
sm_am_gldas_ind = gldas_filefinder(sm_am_utct_ind, timestamp_gldas)
sm_pm_gldas_ind = gldas_filefinder(sm_pm_utct_ind, timestamp_gldas)


########################################################################################################################
# 2. Process GLDAS data

matsize_gldas_geo_1day = [len(lat_gldas_geo_25km), len(lon_gldas_geo_25km)]
gldas_mat_init_geo_1day = np.empty(matsize_gldas_geo_1day, dtype='float32')
gldas_mat_init_geo_1day[:] = np.nan
matsize_world_geo_1day = [len(lat_world_geo_25km), len(lon_world_geo_25km)]
world_mat_init_geo_1day = np.empty(matsize_world_geo_1day, dtype='float32')
world_mat_init_geo_1day[:] = np.nan
matsize_gldas_ease_1day = [len(lat_world_ease_25km), len(lon_world_ease_25km)]
gldas_mat_init_ease_1day = np.empty(matsize_gldas_ease_1day, dtype='float32')
gldas_mat_init_ease_1day[:] = np.nan


for iyr in range(len(daysofyear)):

    # Create initial empty matrices for yearly GLDAS LST/SM final output data
    matsize_gldas = [matsize_gldas_geo_1day[0], matsize_gldas_geo_1day[1], (daysofyear[iyr]+2)*8]
    lst_gldas_geo = np.empty(matsize_gldas, dtype='float32')
    lst_gldas_geo[:] = np.nan
    sm_gldas_geo = np.empty(matsize_gldas, dtype='float32')
    sm_gldas_geo[:] = np.nan

    # The indices for GLDAS data are between daysofyear[0:iyr].sum()*8 : daysofyear[0:iyr+1].sum()*8
    # Add the 8 files from Dec 31, 1980 at the beginning of the yearly index
    # Back to the last day from last year (minus 8) and the first day from next year to the array for each loop (add 8)

    # 2.1 Extract the 3-hour GLDAS LST and SM data
    # extract data from the three years (extract one day data from the before or after year, respectively)

    # The last day of the previous year
    path_gldas_0 = path_gldas + '/' + str(yearname[iyr] - 1) + '/'
    os.chdir(path_gldas_0)
    gldas_files_0 = sorted(glob.glob('*.nc4'))
    gldas_files_0 = gldas_files_0[-8:]
    gldas_files_yearly_0 = [path_gldas_0 + gldas_files_0[x] for x in range(len(gldas_files_0))]
    # The current year
    path_gldas_1 = path_gldas + '/' + str(yearname[iyr]) + '/'
    os.chdir(path_gldas_1)
    gldas_files_1 = sorted(glob.glob('*.nc4'))
    gldas_files_yearly_1 = [path_gldas_1 + gldas_files_1[x] for x in range(len(gldas_files_1))]
    # The first day of the next year
    path_gldas_2 = path_gldas + '/' + str(yearname[iyr] + 1) + '/'
    os.chdir(path_gldas_2)
    gldas_files_2 = sorted(glob.glob('*.nc4'))
    gldas_files_2 = gldas_files_2[:8]
    gldas_files_yearly_2 = [path_gldas_2 + gldas_files_2[x] for x in range(len(gldas_files_2))]

    gldas_files_yearly = gldas_files_yearly_0 + gldas_files_yearly_1 + gldas_files_yearly_2

    for idt in range(len(gldas_files_yearly)):
        rootgrp = Dataset(gldas_files_yearly[idt], mode='r')
        lst_read = rootgrp.variables['AvgSurfT_inst'][:]
        lst_read = np.ma.getdata(np.squeeze((lst_read)))
        lst_read = np.flipud(lst_read)
        lst_read[np.where(lst_read == -9999)] = np.nan

        sm_read = rootgrp.variables['SoilMoi0_10cm_inst'][:]
        sm_read = np.ma.getdata(np.squeeze((sm_read)))
        sm_read = np.flipud(sm_read)
        sm_read[np.where(sm_read == -9999)] = np.nan
        sm_read = sm_read/100

        lst_gldas_geo[:, :, idt] = lst_read
        sm_gldas_geo[:, :, idt] = sm_read

        rootgrp.close()
        del(lst_read, sm_read)
        print(gldas_files_yearly[idt].split('/')[-1])

    del(gldas_files_0, gldas_files_1, gldas_files_2)

    # 2.2 Extract data of the correct UTC time files for different locations in the world and joint
    # Extract the 3-hour GLDAS data from different UTC time zones and rebind the new data
    gldas_ease_mat_init = np.empty\
        ([matsize_gldas_ease_1day[0], matsize_gldas_ease_1day[1], daysofyear[iyr]], dtype='float32')
    gldas_ease_mat_init[:] = np.nan
    lst_gldas_am_ease = np.copy(gldas_ease_mat_init)
    lst_gldas_pm_ease = np.copy(gldas_ease_mat_init)
    sm_gldas_am_ease = np.copy(gldas_ease_mat_init)
    sm_gldas_pm_ease = np.copy(gldas_ease_mat_init)
    del(gldas_ease_mat_init)

    for idc in range(daysofyear[iyr]):

        lst_gldas_geo_3day = np.copy(lst_gldas_geo[:, :, (idc+1)*8-8 : (idc+1)*8+16])
        lst_gldas_utc_am_geo_1day = np.copy(gldas_mat_init_geo_1day)
        lst_gldas_utc_pm_geo_1day = np.copy(gldas_mat_init_geo_1day)
        sm_gldas_geo_3day = np.copy(sm_gldas_geo[:, :, (idc+1)*8-8 : (idc+1)*8+16])
        sm_gldas_utc_am_geo_1day = np.copy(gldas_mat_init_geo_1day)
        sm_gldas_utc_pm_geo_1day = np.copy(gldas_mat_init_geo_1day)

        for itm in range(len(ind_timezone)):
            lst_gldas_utc_am_geo_1day[:, row_gldas_geo_25km_tz_ind_group[itm]] = \
                lst_gldas_geo_3day[:, row_gldas_geo_25km_tz_ind_group[itm],
                (lst_am_utct_day[itm] + 1) * 8 + lst_am_gldas_ind[itm] // 3]
            lst_gldas_utc_pm_geo_1day[:, row_gldas_geo_25km_tz_ind_group[itm]] = \
                lst_gldas_geo_3day[:, row_gldas_geo_25km_tz_ind_group[itm],
                (lst_pm_utct_day[itm] + 1) * 8 + lst_pm_gldas_ind[itm] // 3]
            sm_gldas_utc_am_geo_1day[:, row_gldas_geo_25km_tz_ind_group[itm]] = \
                sm_gldas_geo_3day[:, row_gldas_geo_25km_tz_ind_group[itm],
                (sm_am_utct_day[itm] + 1) * 8 + sm_am_gldas_ind[itm] // 3]
            sm_gldas_utc_pm_geo_1day[:, row_gldas_geo_25km_tz_ind_group[itm]] = \
                sm_gldas_geo_3day[:, row_gldas_geo_25km_tz_ind_group[itm],
                (sm_pm_utct_day[itm] + 1) * 8 + sm_pm_gldas_ind[itm] // 3]


        # 2.3 Add rows to GLDAS data to reach world extent
        lst_gldas_world_am_geo_1day = np.copy(world_mat_init_geo_1day)
        lst_gldas_world_am_geo_1day[row_gldas_geo_25km_ind, :] = lst_gldas_utc_am_geo_1day
        lst_gldas_world_pm_geo_1day = np.copy(world_mat_init_geo_1day)
        lst_gldas_world_pm_geo_1day[row_gldas_geo_25km_ind, :] = lst_gldas_utc_pm_geo_1day
        sm_gldas_world_am_geo_1day = np.copy(world_mat_init_geo_1day)
        sm_gldas_world_am_geo_1day[row_gldas_geo_25km_ind, :] = sm_gldas_utc_am_geo_1day
        sm_gldas_world_pm_geo_1day = np.copy(world_mat_init_geo_1day)
        sm_gldas_world_pm_geo_1day[row_gldas_geo_25km_ind, :] = sm_gldas_utc_pm_geo_1day

        # 2.4 Reproject to 25 km EASE grid projection
        # LST (AM)
        lst_gldas_am_ease_1day = np.array\
            ([np.nanmean(lst_gldas_world_am_geo_1day[row_world_ease_25km_from_geo_25km_ind[x], :], axis=0)
              for x in range(len(lat_world_ease_25km))])
        lst_gldas_am_ease_1day = np.array\
            ([np.nanmean(lst_gldas_am_ease_1day[:, col_world_ease_25km_from_geo_25km_ind[y]], axis=1)
              for y in range(len(lon_world_ease_25km))])
        lst_gldas_am_ease_1day = np.fliplr(np.rot90(lst_gldas_am_ease_1day, 3))
        # LST (PM)
        lst_gldas_pm_ease_1day = np.array\
            ([np.nanmean(lst_gldas_world_pm_geo_1day[row_world_ease_25km_from_geo_25km_ind[x], :], axis=0)
              for x in range(len(lat_world_ease_25km))])
        lst_gldas_pm_ease_1day = np.array\
            ([np.nanmean(lst_gldas_pm_ease_1day[:, col_world_ease_25km_from_geo_25km_ind[y]], axis=1)
              for y in range(len(lon_world_ease_25km))])
        lst_gldas_pm_ease_1day = np.fliplr(np.rot90(lst_gldas_pm_ease_1day, 3))
        # SM (AM)
        sm_gldas_am_ease_1day = np.array\
            ([np.nanmean(sm_gldas_world_am_geo_1day[row_world_ease_25km_from_geo_25km_ind[x], :], axis=0)
              for x in range(len(lat_world_ease_25km))])
        sm_gldas_am_ease_1day = np.array\
            ([np.nanmean(sm_gldas_am_ease_1day[:, col_world_ease_25km_from_geo_25km_ind[y]], axis=1)
              for y in range(len(lon_world_ease_25km))])
        sm_gldas_am_ease_1day = np.fliplr(np.rot90(sm_gldas_am_ease_1day, 3))
        # SM (PM)
        sm_gldas_pm_ease_1day = np.array\
            ([np.nanmean(sm_gldas_world_pm_geo_1day[row_world_ease_25km_from_geo_25km_ind[x], :], axis=0)
              for x in range(len(lat_world_ease_25km))])
        sm_gldas_pm_ease_1day = np.array\
            ([np.nanmean(sm_gldas_pm_ease_1day[:, col_world_ease_25km_from_geo_25km_ind[y]], axis=1)
              for y in range(len(lon_world_ease_25km))])
        sm_gldas_pm_ease_1day = np.fliplr(np.rot90(sm_gldas_pm_ease_1day, 3))


        # Final output
        lst_gldas_am_ease[:, :, idc] = lst_gldas_am_ease_1day
        lst_gldas_pm_ease[:, :, idc] = lst_gldas_pm_ease_1day
        sm_gldas_am_ease[:, :, idc] = sm_gldas_am_ease_1day
        sm_gldas_pm_ease[:, :, idc] = sm_gldas_pm_ease_1day


        del(lst_gldas_geo_3day, lst_gldas_utc_am_geo_1day, lst_gldas_utc_pm_geo_1day,
             sm_gldas_geo_3day, sm_gldas_utc_am_geo_1day, sm_gldas_utc_pm_geo_1day)
        del(lst_gldas_world_am_geo_1day, lst_gldas_world_pm_geo_1day,
            sm_gldas_world_am_geo_1day, sm_gldas_world_pm_geo_1day)

        print(gldas_files_yearly[(idc+1) * 8].split('.')[1])


    # 2.5 Save GLDAS variables by year
    os.chdir(path_procdata)
    var_name = ['lst_gldas_am_ease_' + str(yearname[iyr]), 'lst_gldas_pm_ease_' + str(yearname[iyr]),
                'sm_gldas_am_ease_' + str(yearname[iyr]), 'sm_gldas_pm_ease_' + str(yearname[iyr])]
    data_name = ['lst_gldas_am_ease', 'lst_gldas_pm_ease', 'sm_gldas_am_ease', 'sm_gldas_pm_ease']

    with h5py.File('ds_gldas_ease_' + str(yearname[iyr]) + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]))
    f.close()

    del(lst_gldas_geo, sm_gldas_geo, gldas_files_yearly, matsize_gldas, lst_gldas_am_ease, lst_gldas_pm_ease,
        sm_gldas_am_ease, sm_gldas_pm_ease)



# 2.6 Calculate daily LST difference corresponding to SMAP SM of AM/PM overpass
os.chdir(path_procdata)
gldas_files = sorted(glob.glob('*gldas_ease*'))

for ife in range(len(gldas_files)):

    if ife != len(gldas_files)-1:
        fe_1 = h5py.File(gldas_files[ife], "r") # Read the current yearly GLDAS file
        varname_list_fe1 = list(fe_1.keys())
        fe_2 = h5py.File(gldas_files[ife+1], "r") # Read the next yearly GLDAS file
        varname_list_fe2 = list(fe_2.keys())

        for x in range(0, 4):
            var_obj_1 = fe_1[varname_list_fe1[x]][()] # Extract the current yearly GLDAS LST
            exec(varname_list_fe1[x] + '= var_obj_1')
            var_obj_2 = fe_2[varname_list_fe2[x]][:, :, 0] # Extract the first day of next yearly GLDAS LST
            exec(varname_list_fe2[x] + '= var_obj_2')
            del (var_obj_1, var_obj_2)
        fe_1.close()
        fe_2.close()

        # lst_gldas_am exlude the first day of the current year and concatenate the first day of next year
        lst_gldas_am_rebind = \
            np.concatenate((eval(varname_list_fe1[0])[:, :, 1:], np.expand_dims(eval(varname_list_fe2[0]), axis=2)), axis=2)

    else: # For the last year of 2018
        fe_1 = h5py.File(gldas_files[ife], "r") # Read the current yearly GLDAS file
        varname_list_fe1 = list(fe_1.keys())

        for x in range(0, 4):
            var_obj_1 = fe_1[varname_list_fe1[x]][()] # Extract the current yearly GLDAS LST
            exec(varname_list_fe1[x] + '= var_obj_1')
            del (var_obj_1)
        fe_1.close()

        # lst_gldas_am exlude the first day of the current year and concatenate the first day of next year
        lst_gldas_am_rebind = \
            np.concatenate((eval(varname_list_fe1[0])[:, :, 1:], np.expand_dims(gldas_mat_init_ease_1day, axis=2)), axis=2)


    # LST difference corresponding to SMAP SM of AM overpass (lst_gldas_pm - lst_gldas_am of the same day)
    lst_gldas_am_delta = np.absolute(np.subtract(eval(varname_list_fe1[1]), eval(varname_list_fe1[0])))
    # LST difference corresponding to SMAP SM of PM overpass (lst_gldas_pm - lst_gldas_am of the next day)
    lst_gldas_pm_delta = np.absolute(np.subtract(eval(varname_list_fe1[1]), lst_gldas_am_rebind))

    # Copy GLDAS SM of AM/PM
    sm_gldas_am = np.copy(eval(varname_list_fe1[2]))
    sm_gldas_pm = np.copy(eval(varname_list_fe1[3]))

    # Save GLDAS LST difference by year
    var_name = ['lst_gldas_am_delta_' + str(yearname[ife]), 'lst_gldas_pm_delta_' + str(yearname[ife]),
                'sm_gldas_am_' + str(yearname[ife]), 'sm_gldas_pm_' + str(yearname[ife])]
    data_name = ['lst_gldas_am_delta', 'lst_gldas_pm_delta', 'sm_gldas_am', 'sm_gldas_pm']

    with h5py.File('ds_gldas_' + str(yearname[ife]) + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]))
    f.close()

    print(yearname[ife])

    if ife != len(gldas_files)-1:
        for x in range(0, 4):
            exec('del(' + varname_list_fe1[x] + ')')
            exec('del(' + varname_list_fe2[x] + ')')
        del(lst_gldas_am_delta, lst_gldas_am_rebind, lst_gldas_pm_delta, varname_list_fe1, varname_list_fe2, fe_1, fe_2)
    else:
        for x in range(0, 4):
            exec('del(' + varname_list_fe1[x] + ')')
        del(lst_gldas_am_delta, lst_gldas_am_rebind, lst_gldas_pm_delta, varname_list_fe1, fe_1)

print('Section 2 is completed')


########################################################################################################################
# 3. Read LTDR NDVI data and reproject to 25 km EASE grid projection

matsize_gldas_ease_1day = [len(lat_world_ease_25km), len(lon_world_ease_25km)]
gldas_mat_init_ease_1day = np.empty(matsize_gldas_ease_1day, dtype='float32')
gldas_mat_init_ease_1day[:] = np.nan

for iyr in range(len(daysofyear)):

    os.chdir(path_ltdr + '/' + str(yearname[iyr]))
    ltdr_files = sorted(glob.glob('*.hdf'))

    # Create initial empty matrices for yearly LTDR NDVI final output data
    matsize_ltdr = [matsize_gldas_ease_1day[0], matsize_gldas_ease_1day[1], (daysofyear[iyr])]
    ltdr_ndvi_ease = np.empty(matsize_ltdr, dtype='float32')
    ltdr_ndvi_ease[:] = np.nan

    # 3.1 Extract 5 km LTDR NDVI daily data
    for idt in range(len(ltdr_files)):

        ltdr_data = gdal.Open(ltdr_files[idt]).GetSubDatasets()
        ltdr_ndvi_geo_1day = gdal.Open(ltdr_data[0][0]).ReadAsArray()
        ltdr_ndvi_geo_1day = np.asarray(ltdr_ndvi_geo_1day) * 0.0001
        ltdr_ndvi_geo_1day[np.where(ltdr_ndvi_geo_1day <= 0)] = np.nan

        # 3.2 Reproject to 25 km EASE grid projection
        ltdr_ndvi_ease_1day = np.copy(gldas_mat_init_ease_1day)

        ltdr_ndvi_ease_1day = np.array\
            ([np.nanmean(ltdr_ndvi_geo_1day[row_world_ease_25km_from_geo_5km_ind[x], :], axis=0)
              for x in range(len(lat_world_ease_25km))])
        ltdr_ndvi_ease_1day = np.array \
            ([np.nanmean(ltdr_ndvi_ease_1day[:, col_world_ease_25km_from_geo_5km_ind[y]], axis=1)
              for y in range(len(lon_world_ease_25km))])
        ltdr_ndvi_ease_1day = np.fliplr(np.rot90(ltdr_ndvi_ease_1day, 3))
        ltdr_ndvi_ease_1day = ltdr_ndvi_ease_1day * lmask_ease_25km

        # Find the corresponding day by the file name
        file_id = ltdr_files[idt].split('.')[1]
        file_id = int(file_id[-3:])-1
        ltdr_ndvi_ease[:, :, file_id] = ltdr_ndvi_ease_1day

        print(ltdr_files[idt])
        del(ltdr_data, ltdr_ndvi_geo_1day, ltdr_ndvi_ease_1day, file_id)


    # 3.3 Save LTDR NDVI data by year
    os.chdir(path_procdata)
    var_name = ['ltdr_ndvi_ease_' + str(yearname[iyr])]
    data_name = ['ltdr_ndvi_ease']

    with h5py.File('ds_ltdr_ease_' + str(yearname[iyr]) + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]))
    f.close()

    del(ltdr_ndvi_ease, matsize_ltdr, ltdr_files)

print('Section 3 is completed')


#########################################################################################################################

# 4. Build the SM - delta LST by using linear regression model

# Find the indices of land pixels by the 25-km resolution land-ocean mask
[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind] = np.where(~np.isnan(lmask_ease_25km))

# Find the indices of each month in the list of days between 1981 - 2018
nlpyear = 1999
lpyear = 2000
daysofmonth_nlp = np.array([calendar.monthrange(nlpyear, x)[1] for x in range(1, len(monthname)+1)])
ind_nlp = [np.arange(daysofmonth_nlp[0:x].sum(), daysofmonth_nlp[0:x+1].sum()) for x in range(0, len(monthname))]
daysofmonth_lp = np.array([calendar.monthrange(lpyear, x)[1] for x in range(1, len(monthname)+1)])
ind_lp = [np.arange(daysofmonth_lp[0:x].sum(), daysofmonth_lp[0:x+1].sum()) for x in range(0, len(monthname))]
ind_iflpr = np.array([int(calendar.isleap(yearname[x])) for x in range(len(yearname))]) # Find out leap years
# Generate a sequence of the days of months for all years
daysofmonth_seq = np.array([np.tile(daysofmonth_nlp[x], len(yearname)) for x in range(0, len(monthname))])
daysofmonth_seq[1, :] = daysofmonth_seq[1, :] + ind_iflpr # Add leap days to February
daysofmonth_sum = np.sum(daysofmonth_seq, axis=1)


# 4.1 Extract GLDAS/LTDR data from each month and do regresson modeling
os.chdir(path_procdata)
gldas_model_files = sorted(glob.glob('*ds_gldas_[0-9]*'))
ltdr_model_files = sorted(glob.glob('*ds_ltdr_ease_[0-9]*'))

# Loop all years for each month to build the model
for imo in range(len(monthname)):

    # Initialize empty matrices
    lst_gldas_am_delta_all = np.empty((len(row_lmask_ease_25km_ind), daysofmonth_sum[imo])).astype('float32')
    lst_gldas_am_delta_all[:] = np.nan
    lst_gldas_pm_delta_all = np.empty((len(row_lmask_ease_25km_ind), daysofmonth_sum[imo])).astype('float32')
    lst_gldas_pm_delta_all[:] = np.nan
    sm_gldas_am_all = np.empty((len(row_lmask_ease_25km_ind), daysofmonth_sum[imo])).astype('float32')
    sm_gldas_am_all[:] = np.nan
    sm_gldas_pm_all = np.empty((len(row_lmask_ease_25km_ind), daysofmonth_sum[imo])).astype('float32')
    sm_gldas_pm_all[:] = np.nan
    ltdr_ndvi_all = np.empty((len(row_lmask_ease_25km_ind), daysofmonth_sum[imo])).astype('float32')
    ltdr_ndvi_all[:] = np.nan

    for iyr in range(len(gldas_model_files)):

        # Loop read yearly GLDAS/LTDR data and concatenate (land pixels only)
        fe_gldas = h5py.File(gldas_model_files[iyr], "r")
        varname_list_gldas = list(fe_gldas.keys())

        fe_ndvi = h5py.File(ltdr_model_files[iyr], "r")
        varname_list_ndvi = list(fe_ndvi.keys())

        if calendar.isleap(yearname[iyr]) != True:
            lst_gldas_am_delta = fe_gldas[varname_list_gldas[0]][:, :, ind_nlp[imo]]
            lst_gldas_pm_delta = fe_gldas[varname_list_gldas[1]][:, :, ind_nlp[imo]]
            sm_gldas_am = fe_gldas[varname_list_gldas[2]][:, :, ind_nlp[imo]]
            sm_gldas_pm = fe_gldas[varname_list_gldas[3]][:, :, ind_nlp[imo]]
            ltdr_ndvi = fe_ndvi[varname_list_ndvi[0]][:, :, ind_nlp[imo]]
        else:
            lst_gldas_am_delta = fe_gldas[varname_list_gldas[0]][:, :, ind_lp[imo]]
            lst_gldas_pm_delta = fe_gldas[varname_list_gldas[1]][:, :, ind_lp[imo]]
            sm_gldas_am = fe_gldas[varname_list_gldas[2]][:, :, ind_lp[imo]]
            sm_gldas_pm = fe_gldas[varname_list_gldas[3]][:, :, ind_lp[imo]]
            ltdr_ndvi = fe_ndvi[varname_list_ndvi[0]][:, :, ind_lp[imo]]

        lst_gldas_am_delta = lst_gldas_am_delta[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind, :]
        lst_gldas_pm_delta = lst_gldas_pm_delta[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind, :]
        sm_gldas_am = sm_gldas_am[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind, :]
        sm_gldas_pm = sm_gldas_pm[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind, :]
        ltdr_ndvi = ltdr_ndvi[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind, :]

        # Write the extracted yearly matrices to all-year matrices
        lst_gldas_am_delta_all[:, daysofmonth_seq[imo, 0:iyr].sum() : daysofmonth_seq[imo, 0:iyr+1].sum()] = lst_gldas_am_delta
        lst_gldas_pm_delta_all[:, daysofmonth_seq[imo, 0:iyr].sum() : daysofmonth_seq[imo, 0:iyr+1].sum()] = lst_gldas_pm_delta
        sm_gldas_am_all[:, daysofmonth_seq[imo, 0:iyr].sum() : daysofmonth_seq[imo, 0:iyr+1].sum()] = sm_gldas_am
        sm_gldas_pm_all[:, daysofmonth_seq[imo, 0:iyr].sum() : daysofmonth_seq[imo, 0:iyr+1].sum()] = sm_gldas_pm
        ltdr_ndvi_all[:, daysofmonth_seq[imo, 0:iyr].sum(): daysofmonth_seq[imo, 0:iyr + 1].sum()] = ltdr_ndvi

        print(str(yearname[iyr]) + '-' + monthnum[imo])

        fe_gldas.close()
        fe_ndvi.close()
        del(lst_gldas_am_delta, lst_gldas_pm_delta, sm_gldas_am, sm_gldas_pm, ltdr_ndvi, fe_gldas, fe_ndvi,
            varname_list_gldas, varname_list_ndvi)

    # 4.2 Save the modeling data by month

    var_name = ['lst_gldas_am_delta_all_' + monthnum[imo], 'lst_gldas_pm_delta_all_' + monthnum[imo],
                'sm_gldas_am_all_' + monthnum[imo], 'sm_gldas_pm_all_' + monthnum[imo], 'ltdr_ndvi_all_' + monthnum[imo]]
    data_name = ['lst_gldas_am_delta_all', 'lst_gldas_pm_delta_all', 'sm_gldas_am_all', 'sm_gldas_pm_all', 'ltdr_ndvi_all']

    with h5py.File('ds_model_' + monthnum[imo] + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]))
    f.close()

    del(lst_gldas_am_delta_all, lst_gldas_pm_delta_all, sm_gldas_am_all, sm_gldas_pm_all, ltdr_ndvi_all)
    print(monthnum[imo] + ' is completed')



# 4.3 Build linear regression model between SM and delta LST for each month

os.chdir(path_procdata)
ds_model_files = sorted(glob.glob('*ds_model_[0-9]*'))

# Find the indices of land pixels by the 25-km resolution land-ocean mask
[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind] = np.where(~np.isnan(lmask_ease_25km))

# The 20 numbers are linear regression model coefficients for 10 NDVI classes
model_mat_initial = np.empty([len(row_lmask_ease_25km_ind), 20], dtype='float32')
model_mat_initial[:] = np.nan
regr = linear_model.LinearRegression()
ndvi_class = np.linspace(0, 1, 11)

for imo in range(1, len(monthname)):

    coef_mat_am_output = np.copy(model_mat_initial)
    coef_mat_pm_output = np.copy(model_mat_initial)
    metric_mat_am_output = np.copy(model_mat_initial)
    metric_mat_pm_output = np.copy(model_mat_initial)

    fe_model = h5py.File(ds_model_files[imo], "r")
    varname_list_model = list(fe_model.keys())
    lst_gldas_am_delta = fe_model[varname_list_model[0]][()]
    lst_gldas_pm_delta = fe_model[varname_list_model[1]][()]
    ltdr_ndvi = fe_model[varname_list_model[2]][()]
    sm_gldas_am = fe_model[varname_list_model[3]][()]
    sm_gldas_pm = fe_model[varname_list_model[4]][()]
    fe_model.close()


    for ipx in range(len(ltdr_ndvi)):
        # Find the indices for each NDVI class through each land pixel
        ind_ndvi_px = [np.squeeze(np.array(np.where((ltdr_ndvi[ipx, :] >= ndvi_class[x]) & (ltdr_ndvi[ipx, :] < ndvi_class[x + 1]))))
                            for x in range(len(ndvi_class)-1)]

        # Output Order: coefficient, interceptor, r2, rmse
        # Fit AM data
        regr_mat_am = np.array([reg_proc(lst_gldas_am_delta[ipx, ind_ndvi_px[y]], sm_gldas_am[ipx, ind_ndvi_px[y]])
                                  for y in range(len(ndvi_class)-1)])
        regr_mat_am = np.transpose(regr_mat_am)
        coef_mat_am = regr_mat_am[:2, :].flatten('F')
        metric_mat_am = regr_mat_am[2:, :].flatten('F')
        coef_mat_am_output[ipx, :] = coef_mat_am
        metric_mat_am_output[ipx, :] = metric_mat_am

        # Fit PM data
        regr_mat_pm = np.array([reg_proc(lst_gldas_pm_delta[ipx, ind_ndvi_px[y]], sm_gldas_pm[ipx, ind_ndvi_px[y]])
                                  for y in range(len(ndvi_class)-1)])
        regr_mat_pm = np.transpose(regr_mat_pm)
        coef_mat_pm = regr_mat_pm[:2, :].flatten('F')
        metric_mat_pm = regr_mat_pm[2:, :].flatten('F')
        coef_mat_pm_output[ipx, :] = coef_mat_pm
        metric_mat_pm_output[ipx, :] = metric_mat_pm

        print(monthnum[imo] + '-' + str(ipx))
        del(ind_ndvi_px, regr_mat_am, coef_mat_am, metric_mat_am, regr_mat_pm, coef_mat_pm, metric_mat_pm)


    # 4.4 Save the regression coefficients and correlation metrics by month

    var_name = ['coef_mat_am_' + monthnum[imo], 'metric_mat_am_' + monthnum[imo],
                'coef_mat_pm_' + monthnum[imo], 'metric_mat_pm_' + monthnum[imo]]
    data_name = ['coef_mat_am_output', 'metric_mat_am_output', 'coef_mat_pm_output', 'metric_mat_pm_output']

    with h5py.File('ds_model_coef_' + monthnum[imo] + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]))
    f.close()

    del(coef_mat_am_output, metric_mat_am_output, coef_mat_pm_output, metric_mat_pm_output, fe_model, varname_list_model,
        lst_gldas_am_delta, lst_gldas_pm_delta, sm_gldas_am, sm_gldas_pm, ltdr_ndvi)
    print(monthnum[imo] + ' is completed')

