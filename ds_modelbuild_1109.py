import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import statsmodels.api as sm
from netCDF4 import Dataset
import datetime
import glob

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

#########################################################################################
# (Function 3) Generate lat/lon tables of Geographic projection in world/CONUS

def geo_coord_gen(lat_geo_extent_max, lat_geo_extent_min, lon_geo_extent_max, lon_geo_extent_min, cellsize):
    lat_geo_output = np.linspace(lat_geo_extent_max - cellsize / 2, lat_geo_extent_min + cellsize / 2,
                                    num=int((lat_geo_extent_max - lat_geo_extent_min) / cellsize)).reshape(1, -1)
    lat_geo_output = np.round(lat_geo_output, decimals=3)
    lon_geo_output = np.linspace(lon_geo_extent_min + cellsize / 2, lon_geo_extent_max - cellsize / 2,
                                    num=int((lon_geo_extent_max - lon_geo_extent_min) / cellsize)).reshape(1, -1)
    lon_geo_output = np.round(lon_geo_output, decimals=3)

    return lat_geo_output, lon_geo_output

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of source GLDAS data
path_gldas = '/Volumes/SBac/Dataset/GLDAS'
# Path of processed data
path_procdata = '/Volumes/SBac/files'
# Load in variables
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = list(f.keys())
varname_import = [varname_list[x] for x in [2, 43, 48, 50, 51, 65, 70, 72, 73, 93]]

for x in range(len(varname_import)):
    var_obj = f[varname_import[x]][()]
    exec(varname_import[x] + '= var_obj')
f.close()

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

# Generate GLDAS extent lat/lon tables and corresponding row indices in the world lat table
[lat_gldas_geo_25km, lon_gldas_geo_25km] = geo_coord_gen\
    (lat_gldas_max, lat_gldas_min, lon_gldas_max, lon_gldas_min, cellsize_25km)
row_gldas_geo_25km_ind = np.where((lat_world_geo_25km <= lat_gldas_max) &
                          (lat_world_geo_25km >= lat_gldas_min))[1].reshape(1, -1)

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

days_year = []
for idt in range(len(yearname)):
    f_date = datetime.date(yearname[idt], monthname[0], 1)
    l_date = datetime.date(yearname[idt], monthname[-1], 31)
    delta_1y = l_date - f_date
    days_year.append(delta_1y.days + 1)
    # print(delta_1y.days + 1)

days_year = np.asarray(days_year)

########################################################################################################################

# 1. Time zone conversions
# Longitudes for 15° apart time zones
lon_timezone = np.linspace(lon_world_geo_min, lon_world_geo_max, num=n_timezone+1, dtype=float)
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
    ind_min = np.absolute(lon_gldas_geo_25km[0, i] - lon_timezone_center)
    row_gldas_geo_25km_tz_ind.append(ind_timezone_p1[np.argmin(ind_min)])

row_gldas_geo_25km_tz_ind = np.asarray(row_gldas_geo_25km_tz_ind)
row_gldas_geo_25km_tz_ind[np.where(row_gldas_geo_25km_tz_ind == 12)] = -12

# Group the latitudes by time zone indices
row_gldas_geo_25km_tz_ind_group = \
    [np.where(row_gldas_geo_25km_tz_ind == ind_timezone_p1[x]) for x in range(len(ind_timezone_p1)-1)]

# Find out the correct corresponding UTC time for each time zone for specific local times
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
# 2.1 Extract the 3-hour GLDAS LST and SM data
os.chdir(path_gldas)
gldas_files = sorted(glob.glob('*.nc4'))
gldas_files = gldas_files[0:2920]
matsize_gldas = [lat_gldas_geo_25km.size, lon_gldas_geo_25km.size, len(gldas_files)]
matsize_gldas_1day = [lat_gldas_geo_25km.size, lon_gldas_geo_25km.size]
lst_gldas_geo = np.empty(matsize_gldas, dtype='float32')
lst_gldas_geo[:] = np.nan
sm_gldas_geo = np.empty(matsize_gldas, dtype='float32')
sm_gldas_geo[:] = np.nan

for idt in range(len(gldas_files)):
    rootgrp = Dataset(gldas_files[idt], mode='r')

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
    print(gldas_files[idt])


# 2.2 Save GLDAS variables by year
os.chdir(path_procdata)
var_name = ['lst_gldas_geo', 'sm_gldas_geo']
data_name = ['lst_gldas_geo_sub', 'sm_gldas_geo_sub']
days_year = days_year[0:2]

for idt in range(len(days_year)-1):
    lst_gldas_geo_sub = np.copy(lst_gldas_geo[:, :, days_year[0:idt].sum()*8:days_year[0:idt+1].sum()*8])
    sm_gldas_geo_sub = np.copy(sm_gldas_geo[:, :, days_year[0:idt].sum()*8:days_year[0:idt+1].sum()*8])
    with h5py.File('ds_gldas_geo_' + str(yearname[idt]) + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(data_name[idv]))
    f.close()
    del(lst_gldas_geo_sub, sm_gldas_geo_sub)
    print(yearname[idt])

print('Section 2 is completed')


########################################################################################################################
# 3. Extract data of the correct UTC time files for different locations in the world and joint
# 3.1 Extract the 3-hour GLDAS data from different UTC time zones and rebind the new data
gldas_mat_init_1day = np.empty(matsize_gldas_1day, dtype='float32')
gldas_mat_init_1day[:] = np.nan

# LST
gldas_mat_init = np.empty([matsize_gldas_1day[0], matsize_gldas_1day[1], len(gldas_files)//8], dtype='float32')
gldas_mat_init[:] = np.nan
lst_gldas_utc_am_geo = np.copy(gldas_mat_init)
lst_gldas_utc_pm_geo = np.copy(gldas_mat_init)
del(gldas_mat_init)

for idt in range(1, len(gldas_files)//8 - 1):
    lst_gldas_geo_3day = np.copy(lst_gldas_geo[:, :, (idt*8-8):(idt*8+16)])
    lst_gldas_utc_am_geo_1day = np.copy(gldas_mat_init_1day)
    lst_gldas_utc_pm_geo_1day = np.copy(gldas_mat_init_1day)
    for itm in range(len(ind_timezone)):
        lst_gldas_utc_am_geo_1day[:, row_gldas_geo_25km_tz_ind_group[itm]] = \
        lst_gldas_geo_3day[:, row_gldas_geo_25km_tz_ind_group[itm], (lst_am_utct_day[itm]+1)*8 + lst_am_gldas_ind[itm]//3]
        lst_gldas_utc_pm_geo_1day[:, row_gldas_geo_25km_tz_ind_group[itm]] = \
        lst_gldas_geo_3day[:, row_gldas_geo_25km_tz_ind_group[itm], (lst_pm_utct_day[itm]+1)*8 + lst_pm_gldas_ind[itm]//3]
    lst_gldas_utc_am_geo[:, :, idt] = lst_gldas_utc_am_geo_1day
    lst_gldas_utc_pm_geo[:, :, idt] = lst_gldas_utc_pm_geo_1day
    del(lst_gldas_geo_3day, lst_gldas_utc_am_geo_1day, lst_gldas_utc_pm_geo_1day)
    print(gldas_files[idt*8].split('.')[1] + '_lst')

# Save the LST GLDAS data by year
os.chdir(path_procdata)
var_name = ['lst_gldas_utc_am_geo_sub', 'lst_gldas_utc_pm_geo_sub']
days_year = days_year[0:1]

for idt in range(len(days_year)):
    lst_gldas_utc_am_geo_sub = np.copy(lst_gldas_utc_am_geo[:, :, days_year[0:idt].sum():days_year[0:idt+1].sum()])
    lst_gldas_utc_pm_geo_sub = np.copy(lst_gldas_utc_pm_geo[:, :, days_year[0:idt].sum():days_year[0:idt + 1].sum()])
    with h5py.File('ds_gldas_utc_lst_geo_' + str(yearname[idt]) + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(var_name[idv]))
    f.close()
    del(lst_gldas_utc_am_geo_sub, lst_gldas_utc_pm_geo_sub)
    print(yearname[idt] + '_lst')



# Soil Moisture
gldas_mat_init = np.empty([matsize_gldas_1day[0], matsize_gldas_1day[1], len(gldas_files)//8], dtype='float32')
gldas_mat_init[:] = np.nan
sm_gldas_utc_am_geo = np.copy(gldas_mat_init)
sm_gldas_utc_pm_geo = np.copy(gldas_mat_init)
del(gldas_mat_init)

for idt in range(1, len(gldas_files)//8 - 1):
    sm_gldas_geo_3day = np.copy(sm_gldas_geo[:, :, (idt*8-8):(idt*8+16)])
    sm_gldas_utc_am_geo_1day = np.copy(gldas_mat_init_1day)
    sm_gldas_utc_pm_geo_1day = np.copy(gldas_mat_init_1day)
    for itm in range(len(ind_timezone)):
        sm_gldas_utc_am_geo_1day[:, row_gldas_geo_25km_tz_ind_group[itm]] = \
        sm_gldas_geo_3day[:, row_gldas_geo_25km_tz_ind_group[itm], (sm_am_utct_day[itm]+1)*8 + sm_am_gldas_ind[itm]//3]
        sm_gldas_utc_pm_geo_1day[:, row_gldas_geo_25km_tz_ind_group[itm]] = \
        sm_gldas_geo_3day[:, row_gldas_geo_25km_tz_ind_group[itm], (sm_pm_utct_day[itm]+1)*8 + sm_pm_gldas_ind[itm]//3]
    sm_gldas_utc_am_geo[:, :, idt] = sm_gldas_utc_am_geo_1day
    sm_gldas_utc_pm_geo[:, :, idt] = sm_gldas_utc_pm_geo_1day
    del(sm_gldas_geo_3day, sm_gldas_utc_am_geo_1day, sm_gldas_utc_pm_geo_1day)
    print(gldas_files[idt*8].split('.')[1] + '_sm')

# Save the SM GLDAS data by year
os.chdir(path_procdata)
var_name = ['sm_gldas_utc_am_geo_sub', 'sm_gldas_utc_pm_geo_sub']
days_year = days_year[0:1]

for idt in range(len(days_year)):
    sm_gldas_utc_am_geo_sub = np.copy(sm_gldas_utc_am_geo[:, :, days_year[0:idt].sum():days_year[0:idt+1].sum()])
    sm_gldas_utc_pm_geo_sub = np.copy(sm_gldas_utc_pm_geo[:, :, days_year[0:idt].sum():days_year[0:idt+1].sum()])
    with h5py.File('ds_gldas_utc_sm_geo_' + str(yearname[idt]) + '.hdf5', 'w') as f:
        for idv in range(len(var_name)):
            f.create_dataset(var_name[idv], data=eval(var_name[idv]))
    f.close()
    del(sm_gldas_utc_am_geo_sub, sm_gldas_utc_pm_geo_sub)
    print(yearname[idt] + '_sm')














########################################################################################################################

# * Check the completeness of downloaded files
os.chdir(path_gldas)
files = sorted(glob.glob('*.nc4'))
files_group = []
for idt in range(6938):
    files_group_1day = [files.index(i) for i in files if 'A' + date_seq[idt] in i]
    files_group.append(files_group_1day)
    print(idt)

file_miss = []
for idt in range(len(files_group)):
    if len(files_group[idt]) != 8:
        file_miss.append(date_seq[idt])
        print(date_seq[idt])
        print(len(files_group[idt]))
    else:
        pass