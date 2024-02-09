import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import calendar
import math
import statsmodels.api as sm
import zipfile
import tempfile
import shutil
from netCDF4 import Dataset
import itertools
import datetime
import glob
import cdsapi
import io
from sklearn import linear_model
regr = linear_model.LinearRegression()
from sklearn.metrics import mean_squared_error, r2_score

########################################################################################################################
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
# (Function 2) Find out the corresponding file of the closest UTC time from era data collection for any UTC times

def era_filefinder(var_name_utct_ind, timestamp_era):
    var_name_era_ind = np.copy(var_name_utct_ind)
    for tm in var_name_utct_ind:
        if var_name_utct_ind[tm] == 23: # Assign the UTC time = 23:00 to 0:00
            time_min = np.absolute(var_name_utct_ind[tm]-24 - timestamp_era)
            var_name_era_ind[tm] = timestamp_era[np.argmin(time_min)]
        else:
            time_min = np.absolute(var_name_utct_ind[tm] - timestamp_era)
            var_name_era_ind[tm] = timestamp_era[np.argmin(time_min)]

    return var_name_era_ind

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
# (Function 4) Define a function to output coefficient and intercept of linear regression fit

def regr_proc(x_arr, y_arr):
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

##############################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of processed data
path_procdata = '/Volumes/SBac/files'
path_era5_land = '/Volumes/Elements2/ERA5-Land'
path_era5_land_download = '/Users/binfang/Downloads/era5_land'

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'lat_world_geo_10km', 'lon_world_geo_10km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'row_world_ease_1km_from_10km_ind', 'col_world_ease_1km_from_10km_ind',
                'row_world_ease_1km_from_9km_ind', 'col_world_ease_1km_from_9km_ind',
                'row_world_ease_9km_from_geo_10km_ind', 'col_world_ease_9km_from_geo_10km_ind',
                'row_world_ease_9km_from_1km_ext33km_ind', 'col_world_ease_9km_from_1km_ext33km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


# era extent corner coordinates
lat_era_max = 90
lat_era_min = -90
lon_era_max = 180
lon_era_min = -180
n_timezone = 24
step_timezone = 15
ind_timezone_min = -12
ind_timezone_max = 11
lst_am = 1 # 1:00
lst_pm = 13 # 13:00
sm_am = 6 # 6:00
sm_pm = 18 # 18:00

lat_era = np.round(np.linspace(lat_era_min, lat_era_max, 1801), 2)
lon_era = np.round(np.linspace(lon_era_min, lon_era_max - 0.1, 3600), 2)
# # Generate era extent lat/lon tables and corresponding row indices in the world lat table
# [lat_era_geo_10km, lon_era_geo_10km] = geo_coord_gen\
#     (lat_era_max, lat_era_min, lon_era_max, lon_era_min, cellsize_10km)
# row_era_geo_10km_ind = np.where((lat_world_geo_10km <= lat_era_max) &
#                           (lat_world_geo_10km >= lat_era_min))[1].reshape(1, -1)

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '1981-01-01'
end_date = '2020-12-31'
year = 2020 - 1981 + 1

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
yearname = np.linspace(1981, 2020, 40, dtype='int')
monthname = np.linspace(1, 12, 12, dtype='int')

days_year = []
for idt in range(len(yearname)):
    f_date = datetime.date(yearname[idt], monthname[0], 1)
    l_date = datetime.date(yearname[idt], monthname[-1], 31)
    delta_1y = l_date - f_date
    days_year.append(delta_1y.days + 1)
    # print(delta_1y.days + 1)

days_year = np.asarray(days_year)

# Find the indices of each month in the list of days between 2015 - 2018
nlpyear = 1999 # non-leap year
lpyear = 2000 # leap year
daysofmonth_nlp = np.array([calendar.monthrange(nlpyear, x)[1] for x in range(1, len(monthname)+1)])
ind_nlp = [np.arange(daysofmonth_nlp[0:x].sum(), daysofmonth_nlp[0:x+1].sum()) for x in range(0, len(monthname))]
daysofmonth_lp = np.array([calendar.monthrange(lpyear, x)[1] for x in range(1, len(monthname)+1)])
ind_lp = [np.arange(daysofmonth_lp[0:x].sum(), daysofmonth_lp[0:x+1].sum()) for x in range(0, len(monthname))]
ind_iflpr = np.array([int(calendar.isleap(yearname[x])) for x in range(len(yearname))]) # Find out leap years
# Generate a sequence of the days of months for all years
daysofmonth_seq = np.array([np.tile(daysofmonth_nlp[x], len(yearname)) for x in range(0, len(monthname))])
daysofmonth_seq[1, :] = daysofmonth_seq[1, :] + ind_iflpr # Add leap days to February
# daysofmonth_seq = daysofmonth_seq[:, :-1]
# days_year = days_year[:-1]
daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=0)
days_year_cumsum = np.cumsum(days_year, axis=0)
days_year_cumsum = np.insert(days_year_cumsum, 0, 0)
days_year_cumsum = days_year_cumsum[:-1]
daysofmonth_seq_ind = days_year_cumsum + daysofmonth_seq_cumsum
daysofmonth_seq_ind_linear = np.reshape(daysofmonth_seq_ind, -1, order='F')
daysofmonth_seq_ind_linear = np.insert(daysofmonth_seq_ind_linear, 0, 0)

days_start_all = []
days_end_all = []
for imo in range(len(monthname)):
    days_start = [daysofmonth_seq_ind_linear[x*12+imo] for x in range(len(yearname))]
    days_end = [daysofmonth_seq_ind_linear[x*12+imo+1] for x in range(len(yearname))]
    days_start_all.append(days_start)
    days_end_all.append(days_end)
    del(days_start, days_end)


rootgrp_lmask = Dataset('/Volumes/Elements/Datasets/Lmask/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc', mode='r')
lmask_read = rootgrp_lmask.variables['lsm'][:]
lmask_10km = np.ma.getdata(np.squeeze((lmask_read)))
lmask_10km_shape = lmask_10km.shape
lmask_10km[np.where(lmask_10km != 0)] = 1
lmask_10km[np.where(lmask_10km == 0)] = np.nan
lmask_10km = np.concatenate((lmask_10km[:, 1800:], lmask_10km[:, :1800]), axis=1)
lmask_10km = lmask_10km.reshape(-1, 1).astype(float)
lmask_10km_ind = np.where(~np.isnan(lmask_10km))[0]
antar_ind = 1500 * lmask_10km_shape[1]
lmask_10km_ind = lmask_10km_ind[lmask_10km_ind < antar_ind]

########################################################################################################################
# 1. Time zone conversions
# Longitudes for 15° apart time zones
lon_timezone = np.linspace(lon_era_min, lon_era_max, num=n_timezone+1, dtype=float)
lon_timezone = lon_timezone + step_timezone/2 # The UTC timezone starts from 7.5°W
lon_timezone = lon_timezone[0:-1] # Delete the last element
lon_timezone_center = lon_timezone - step_timezone/2
lon_timezone_center = np.append(lon_timezone_center, 180) # Add one index for latitudes near -180°

# Index of time zones
ind_timezone = np.linspace(ind_timezone_min, ind_timezone_max, num=n_timezone, dtype=int)
ind_timezone_p1 = np.append(ind_timezone, 12) # Add one index for latitudes near -180°

# Find the corresponding UTC time zone for each latitude in the lat table
row_era_geo_10km_tz_ind = []
for i in range(lon_world_geo_10km.size):
    ind_min = np.absolute(lon_world_geo_10km[i] - lon_timezone_center)
    row_era_geo_10km_tz_ind.append(ind_timezone_p1[np.argmin(ind_min)])

row_era_geo_10km_tz_ind = np.asarray(row_era_geo_10km_tz_ind)
row_era_geo_10km_tz_ind[np.where(row_era_geo_10km_tz_ind == 12)] = -12

# Group the latitudes by time zone indices
row_era_geo_10km_tz_ind_group = \
    [np.where(row_era_geo_10km_tz_ind == ind_timezone_p1[x]) for x in range(len(ind_timezone_p1)-1)]

# Find out the correct corresponding UTC time for each time zone for specific local times
[lst_am_utct_day, lst_am_utct, lst_am_utct_ind] = timezone_converter(lst_am, ind_timezone)
[lst_pm_utct_day, lst_pm_utct, lst_pm_utct_ind] = timezone_converter(lst_pm, ind_timezone)
[sm_am_utct_day, sm_am_utct, sm_am_utct_ind] = timezone_converter(sm_am, ind_timezone)
[sm_pm_utct_day, sm_pm_utct, sm_pm_utct_ind] = timezone_converter(sm_pm, ind_timezone)



########################################################################################################################
# 2. Download the ERA5-land data
var_list = [['leaf_area_index_high_vegetation', 'volumetric_soil_water_layer_1'], ['skin_temperature']]
var_name_list = ['lai_sm', 'lst']
year_era = np.arange(1981, 2022)
lon_timezone_add = np.concatenate([[-180], lon_timezone, [180]])
time_list = [[np.concatenate([sm_am_utct_ind, [sm_am_utct_ind[0]]]),
              np.concatenate([sm_pm_utct_ind, [sm_pm_utct_ind[0]]])],
             [np.concatenate([lst_am_utct_ind, [lst_am_utct_ind[0]]]),
              np.concatenate([lst_pm_utct_ind, [lst_pm_utct_ind[0]]])]]
# os.makedirs(path_era5_land_download + '/' + var_name_list[0])
# os.makedirs(path_era5_land_download + '/' + var_name_list[1])

c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api/v2',
                  key='264966:ca416c06-eadd-4d6f-afab-bef556a38336')

for iva in range(len(var_list)): # 2 variable lists
    for iyr in range(len(year_era)):
        for itm in range(len(lon_timezone_add) - 1):  # 24 time zones
            output_file_name = var_name_list[iva] + '_' + str(year_era[iyr]) + '_' + str(itm+1).zfill(2)
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': var_list[iva],
                    'year': str(year_era[iyr]),
                    'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                            '11', '12',],
                    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
                            ],
                    'time': [str(time_list[iva][0][itm]).zfill(2) + ':00',
                             str(time_list[iva][1][itm]).zfill(2) + ':00'],
                    'format': 'netcdf.zip',
                    'area': [90, lon_timezone_add[itm], -90, lon_timezone_add[itm+1]]
                },

                path_era5_land_download + '/' + var_name_list[iva] + '/' + output_file_name + '.zip')
            print(output_file_name)


########################################################################################################################
# 3. Process era data
# 3.1 Extract era5-land LST and SM data
var_name_list = ['lai_sm', 'lst']
year_era = np.arange(1980, 2022)
file_name_all = []
for iva in range(len(var_name_list)):
    file_name = sorted(glob.glob(path_era5_land + '/data/' + var_name_list[iva] + '/*'))
    file_name_all.append(file_name)
    del(file_name)
file_name_all = list((file_name_all[0], file_name_all[0], file_name_all[1]))
era_variables = ['lai_hv', 'swvl1', 'skt']
era_variable_names = ['_lai', '_sm', '_lst']


# Find the indices of columns for re-combining era5 data for adjusting to correct date
sm_am_timespan_ind = [2775, 3525]
sm_am_timespan_factor_ind = [1, 0, 1]
sm_pm_timespan_ind = [975, 3525]
sm_pm_timespan_factor_ind = [2, 1, 2]
lst_am_timespan_ind = [2025, 3525]
lst_am_timespan_factor_ind = [1, 0, 1]
lst_pm_timespan_ind = [225, 3525]
lst_pm_timespan_factor_ind = [2, 1, 2]

var_am_timespan_ind = [sm_am_timespan_ind, sm_am_timespan_ind, lst_am_timespan_ind]
var_pm_timespan_ind = [sm_pm_timespan_ind, sm_pm_timespan_ind, lst_pm_timespan_ind]
var_am_timespan_factor_ind = [sm_am_timespan_factor_ind, sm_am_timespan_factor_ind, lst_am_timespan_factor_ind]
var_pm_timespan_factor_ind = [sm_pm_timespan_factor_ind, sm_pm_timespan_factor_ind, lst_pm_timespan_factor_ind]


for iyr in range(len(year_era)-2):
    for iva in range(len(era_variables)):

        # 3-year data
        file_list = [file_name_all[iva][(iyr+x)*25:(iyr+1+x)*25] for x in range(3)]
        ind_dates = [[-2, -1], np.arange(days_year[iyr]*2), [0, 1]]

        var_read_all_3years = []
        for ils in range(3): # 3 years
            var_read_all = []
            for ife in range(len(file_list[ils])):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(file_list[ils][ife], 'r') as zip_ref:
                        zip_ref.extractall(tmp_dir)
                        extracted_files = zip_ref.namelist()
                        nc_file_path = os.path.join(tmp_dir, extracted_files[0])
                        rootgrp = Dataset(nc_file_path, mode='r')

                        # skip the first column which is duplicate
                        var_read = rootgrp.variables[era_variables[iva]][ind_dates[ils], :, 1:]
                        var_read = np.ma.getdata(np.squeeze((var_read)))
                        var_read[np.where(var_read == -32767.0)] = np.nan
                        var_read_all.append(var_read)

                        print(file_list[ils][ife] + '_' + era_variables[iva])
                        del(var_read)
                        zip_ref.close()
                        rootgrp.close()

            var_world = np.concatenate(var_read_all, axis=2)
            var_read_all_3years.append(var_world)
            del(var_read_all, var_world)

        var_read_all_3years = np.concatenate(var_read_all_3years, axis=0)

        var_am = var_read_all_3years[::2, :, :]
        var_pm = var_read_all_3years[1::2, :, :]

        var_am_group = np.split(var_am, var_am_timespan_ind[iva], axis=2)
        var_am_group_rev = [
            var_am_group[x][var_am_timespan_factor_ind[iva][x]:var_am_timespan_factor_ind[iva][x] + days_year[iyr], :,
            :]
            for x in range(len(var_am_group))]
        var_am_group_rev = np.concatenate(var_am_group_rev, axis=2)

        var_pm_group = np.split(var_pm, var_pm_timespan_ind[iva], axis=2)
        var_pm_group_rev = [
            var_pm_group[x][var_pm_timespan_factor_ind[iva][x]:var_pm_timespan_factor_ind[iva][x] + days_year[iyr], :,
            :]
            for x in range(len(var_pm_group))]
        var_pm_group_rev = np.concatenate(var_pm_group_rev, axis=2)

        var_am_land = var_am_group_rev.reshape(var_am_group_rev.shape[0],
                                               var_am_group_rev.shape[1] * var_am_group_rev.shape[2]).astype(float)
        var_pm_land = var_pm_group_rev.reshape(var_pm_group_rev.shape[0],
                                               var_pm_group_rev.shape[1] * var_pm_group_rev.shape[2]).astype(float)

        var_am_land = var_am_land[:, lmask_10km_ind]
        var_pm_land = var_pm_land[:, lmask_10km_ind]
        del(var_read_all_3years, var_am, var_pm, file_list, var_am_group, var_am_group_rev, var_pm_group, var_pm_group_rev)

        output_file_yearly_name = path_era5_land + '/yearly/yearly_' + str(year_era[iyr+1]) + era_variable_names[iva]
        with h5py.File(output_file_yearly_name + '.hdf5', 'w') as f:
            f.create_dataset('var_am', data=var_am_land, compression='lzf')
            f.create_dataset('var_pm', data=var_pm_land, compression='lzf')
        f.close()
        del(var_am_land, var_pm_land)
        print(output_file_yearly_name + '.hdf5')


########################################################################################################################
# 4.3 Build linear regression model between SM and delta LST for each month

# var_name_list = ['lai_sm', 'lst']
year_era = np.arange(1981, 2021)
file_name = sorted(glob.glob(path_era5_land + '/yearly/*'))
f = h5py.File(file_name[0], 'r')
varname_list = list(f.keys())
num_pixel = f[varname_list[0]][()].shape[1]
lai_class = np.arange(6)
lai_class = np.append(lai_class, 10)
date_seq_month = np.array([int(date_seq[x][4:6]) for x in range(len(date_seq))])
month_ind = [np.where(date_seq_month == monthname[x])[0] for x in range(len(monthname))]
var_divide_ind = np.arange(0, num_pixel, 10000)[1:]
var_divide = np.split(np.arange(num_pixel), var_divide_ind)
del(f, varname_list)

file_lai = [file_name[n*3] for n in range(len(year_era))]
file_lst = [file_name[n*3+1] for n in range(len(year_era))]
file_sm = [file_name[n*3+2] for n in range(len(year_era))]


for igp in range(0, len(var_divide)):

    lai_am_all = []
    lai_pm_all = []
    lst_am_all = []
    lst_pm_all = []
    sm_am_all = []
    sm_pm_all = []
    for ife in range(len(file_lai)):
        # LAI
        f_lai = h5py.File(file_lai[ife], 'r')
        varname_list_lai = list(f_lai.keys())
        lai_am = f_lai[varname_list_lai[0]][:, var_divide[igp]]
        lai_pm = f_lai[varname_list_lai[1]][:, var_divide[igp]]
        lai_am_all.append(lai_am)
        lai_pm_all.append(lai_pm)
        del (lai_am, lai_pm)
        f_lai.close()
        # LST
        f_lst = h5py.File(file_lst[ife], 'r')
        varname_list_lst = list(f_lst.keys())
        lst_am = f_lst[varname_list_lst[0]][:, var_divide[igp]]
        lst_pm = f_lst[varname_list_lst[1]][:, var_divide[igp]]
        lst_am_all.append(lst_am)
        lst_pm_all.append(lst_pm)
        del (lst_am, lst_pm)
        f_lst.close()
        # SM
        f_sm = h5py.File(file_sm[ife], 'r')
        varname_list_sm = list(f_sm.keys())
        sm_am = f_sm[varname_list_sm[0]][:, var_divide[igp]]
        sm_pm = f_sm[varname_list_sm[1]][:, var_divide[igp]]
        sm_am_all.append(sm_am)
        sm_pm_all.append(sm_pm)
        del (sm_am, sm_pm)
        f_sm.close()

    lai_am_all = list(itertools.chain(*lai_am_all))
    lai_am_all = np.array(lai_am_all)
    lai_pm_all = list(itertools.chain(*lai_pm_all))
    lai_pm_all = np.array(lai_pm_all)

    lst_am_all = list(itertools.chain(*lst_am_all))
    lst_am_all = np.array(lst_am_all)
    lst_pm_all = list(itertools.chain(*lst_pm_all))
    lst_pm_all = np.array(lst_pm_all)

    sm_am_all = list(itertools.chain(*sm_am_all))
    sm_am_all = np.array(sm_am_all)
    sm_pm_all = list(itertools.chain(*sm_pm_all))
    sm_pm_all = np.array(sm_pm_all)

    # Make an array and let the LST AM start from the second day.
    empty_array = np.empty((1, lst_am_all.shape[1]))
    empty_array[:] = np.nan
    empty_array.shape
    lst_am_all_rev = np.concatenate((lst_am_all, empty_array), axis=0)
    lst_am_all_rev = lst_am_all_rev[1:, :]

    lai_am_all[(lai_am_all < 0) | (np.isnan(lai_am_all))] = 0
    lai_pm_all[(lai_pm_all < 0) | (np.isnan(lai_pm_all))] = 0
    lst_am_delta_all = np.subtract(lst_pm_all, lst_am_all)
    lst_pm_delta_all = np.subtract(lst_pm_all, lst_am_all_rev)

    # Do regression fitting in each month
    for imo in range(0, len(month_ind)):
        lai_am_all_monthly = lai_am_all[month_ind[imo]]
        lai_pm_all_monthly = lai_pm_all[month_ind[imo]]
        lst_am_delta_all_monthly = lst_am_delta_all[month_ind[imo]]
        lst_pm_delta_all_monthly = lst_pm_delta_all[month_ind[imo]]
        sm_am_all_monthly = sm_am_all[month_ind[imo]]
        sm_pm_all_monthly = sm_pm_all[month_ind[imo]]

        coef_am_all = []
        stat_am_all = []
        coef_pm_all = []
        stat_pm_all = []
        for ipx in range(lai_am_all_monthly.shape[1]):
            # AM
            lai_class_am_ind = [np.squeeze(np.array(np.where((lai_am_all_monthly[:, ipx] >= lai_class[x]) &
                                                             (lai_am_all_monthly[:, ipx] < lai_class[x + 1]))))
                                for x in range(len(lai_class) - 1)]
            regr_am = np.array(
                [regr_proc(lst_am_delta_all_monthly[lai_class_am_ind[y], ipx], sm_am_all_monthly[lai_class_am_ind[y], ipx])
                 for y in range(len(lai_class) - 1)])
            coef_am = regr_am[:, :2]
            stat_am = regr_am[:, 2:]
            coef_am_all.append(coef_am)
            stat_am_all.append(stat_am)

            # PM
            lai_class_pm_ind = [np.squeeze(np.array(np.where((lai_pm_all_monthly[:, ipx] >= lai_class[x]) &
                                                             (lai_pm_all_monthly[:, ipx] < lai_class[x + 1]))))
                                for x in range(len(lai_class) - 1)]
            regr_pm = np.array(
                [regr_proc(lst_pm_delta_all_monthly[lai_class_pm_ind[y], ipx], sm_pm_all_monthly[lai_class_pm_ind[y], ipx])
                 for y in range(len(lai_class) - 1)])
            coef_pm = regr_pm[:, :2]
            stat_pm = regr_pm[:, 2:]
            coef_pm_all.append(coef_pm)
            stat_pm_all.append(stat_pm)

            del (lai_class_am_ind, regr_am, coef_am, stat_am, lai_class_pm_ind, regr_pm, coef_pm, stat_pm)

        coef_am_all = np.array(coef_am_all)
        stat_am_all = np.array(stat_am_all)
        coef_pm_all = np.array(coef_pm_all)
        stat_pm_all = np.array(stat_pm_all)

        coef_shape = coef_am_all.shape
        coef_am_all_1d = np.reshape(coef_am_all, (coef_shape[0], coef_shape[1] * coef_shape[2]))
        stat_am_all_1d = np.reshape(stat_am_all, (coef_shape[0], coef_shape[1] * coef_shape[2]))
        coef_pm_all_1d = np.reshape(coef_pm_all, (coef_shape[0], coef_shape[1] * coef_shape[2]))
        stat_pm_all_1d = np.reshape(stat_pm_all, (coef_shape[0], coef_shape[1] * coef_shape[2]))

        # Write coefficients and statistical variables to file
        df_coef_am = pd.DataFrame(coef_am_all_1d)
        df_coef_pm = pd.DataFrame(coef_pm_all_1d)
        path_output_coef = path_era5_land + '/model_output/coef_' + str(imo + 1).zfill(2) + '_' + str(
            igp + 1).zfill(3) + '.xlsx'
        writer_coef = pd.ExcelWriter(path_output_coef)
        df_coef_am.to_excel(writer_coef, sheet_name='AM')
        df_coef_pm.to_excel(writer_coef, sheet_name='PM')
        writer_coef.save()
        print(path_output_coef)

        df_stat_am = pd.DataFrame(stat_am_all_1d)
        df_stat_pm = pd.DataFrame(stat_pm_all_1d)
        path_output_stat = path_era5_land + '/model_output/stat_' + str(imo + 1).zfill(2) + '_' + str(
            igp + 1).zfill(3) + '.xlsx'
        writer_stat = pd.ExcelWriter(path_output_stat)
        df_stat_am.to_excel(writer_stat, sheet_name='AM')
        df_stat_pm.to_excel(writer_stat, sheet_name='PM')
        writer_stat.save()
        print(path_output_stat)

        del(lai_am_all_monthly, lai_pm_all_monthly, lst_am_delta_all_monthly, lst_pm_delta_all_monthly,
             sm_am_all_monthly, sm_pm_all_monthly, path_output_coef, path_output_stat)

    del(lai_am_all, lai_pm_all, lst_am_all, lst_pm_all, sm_am_all, sm_pm_all, empty_array, lst_am_all_rev)


########################################################################################################################
# 5. Combine regression fit tables and fill the empty cells with the closest values.
file_name = sorted(glob.glob(path_era5_land + '/data/model_output/*'))
file_coef = file_name[:len(file_name)//2]
file_stat = file_name[len(file_name)//2:]
num_group = len(file_coef)//12
file_divide_ind = np.arange(0, len(file_coef), num_group)[1:]
file_coef_divide = np.split(file_coef, file_divide_ind)
file_stat_divide = np.split(file_stat, file_divide_ind)


for imo in range(len(monthname)):
    coef_am_all = []
    coef_pm_all = []
    stat_am_all = []
    stat_pm_all = []
    for ife in range(len(file_coef_divide[0])):
        coef_am = np.array(pd.read_excel(file_coef_divide[imo][ife], index_col=0, sheet_name='AM'))
        coef_pm = np.array(pd.read_excel(file_coef_divide[imo][ife], index_col=0, sheet_name='PM'))
        stat_am = np.array(pd.read_excel(file_stat_divide[imo][ife], index_col=0, sheet_name='AM'))
        stat_pm = np.array(pd.read_excel(file_stat_divide[imo][ife], index_col=0, sheet_name='PM'))
        coef_am_all.append(coef_am)
        coef_pm_all.append(coef_pm)
        stat_am_all.append(stat_am)
        stat_pm_all.append(stat_pm)
        del(coef_am, coef_pm, stat_am, stat_pm)
        print(str(imo+1).zfill(2) + '_' + str(ife+1).zfill(3))

    coef_am_all = np.concatenate(coef_am_all, axis=0)
    coef_pm_all = np.concatenate(coef_pm_all, axis=0)
    stat_am_all = np.concatenate(stat_am_all, axis=0)
    stat_pm_all = np.concatenate(stat_pm_all, axis=0)

    with h5py.File(path_era5_land + '/model/coef_' + str(imo + 1).zfill(2) + '.hdf5', 'w') as f:
        f.create_dataset('coef_am_all', data=coef_am_all, compression='lzf')
        f.create_dataset('coef_pm_all', data=coef_pm_all, compression='lzf')
    f.close()

    with h5py.File(path_era5_land + '/model/stat_' + str(imo + 1).zfill(2) + '.hdf5', 'w') as f:
        f.create_dataset('stat_am_all', data=stat_am_all, compression='lzf')
        f.create_dataset('stat_pm_all', data=stat_pm_all, compression='lzf')
    f.close()

    del(coef_am_all, coef_pm_all, stat_am_all, coef_pm_all)


########################################################################################################################
# 6. Combine regression fit tables and fill the empty cells with the closest values.
mat_init = np.empty((lmask_10km_shape[0]*lmask_10km_shape[1]))
mat_init[:] = np.nan
mat_init = np.tile(mat_init, (12, 1)).T

file_name = sorted(glob.glob(path_era5_land + '/data/model_output/*'))
file_coef = file_name[:len(file_name)//2]
file_stat = file_name[len(file_name)//2:]

# Coefficients
var_obj_all = []
var_obj_fill_all = []
for ife in range(len(file_coef)):
    f_coef = h5py.File(file_coef[ife], 'r')
    varname_list_coef = list(f_coef.keys())

    var_obj_1year = []
    var_obj_fill_1year = []
    for x in range(len(varname_list_coef)): #AM/PM
        var_obj = f_coef[varname_list_coef[x]][()]
        var_obj_fill = np.copy(var_obj)

        #Do the coefficients and intercepts filling using values from the closest class
        for m in range(var_obj.shape[0]):
            ind_nan = np.where(np.isnan(var_obj[m, ::2]))[0]  # Find the nan/nonnan using odd indices
            ind_nonnan = np.where(~np.isnan(var_obj[m, ::2]))[0]
            if len(ind_nan) != 0 and len(ind_nan) != var_obj.shape[1]//2:
                ind_fill = np.array([np.where(
                    np.amin(np.absolute(ind_nan[n] - ind_nonnan)) == np.absolute(ind_nan[n] - ind_nonnan))[0][0]
                                     for n in range(len(ind_nan))])

                for i in range(len(ind_fill)):
                    var_obj_fill[m, ind_nan[i] * 2] = var_obj[m, ind_fill[i] * 2]
                    var_obj_fill[m, ind_nan[i] * 2 + 1] = var_obj[m, ind_fill[i] * 2 + 1]

                del(ind_fill)

            else:
                pass

            del(ind_nan, ind_nonnan)

        var_obj_1year.append(var_obj)
        var_obj_fill_1year.append(var_obj_fill)
        del(var_obj, var_obj_fill)

    var_obj_all.append(var_obj_1year)
    var_obj_fill_all.append(var_obj_fill_1year)
    del(var_obj_1year, var_obj_fill_1year)
    print(ife)


# Statistical metrics
var_obj_stat_all = []
for ife in range(len(file_stat)):
    f_stat = h5py.File(file_stat[ife], 'r')
    varname_list_stat = list(f_stat.keys())

    var_obj_stat_1year = []
    for x in range(len(varname_list_stat)): #AM/PM
        var_obj_stat = f_stat[varname_list_stat[x]][()]
        var_obj_stat_1year.append(var_obj_stat)
        del(var_obj_stat)

    var_obj_stat_all.append(var_obj_stat_1year)
    del(var_obj_stat_1year)
    print(ife)


with h5py.File(path_era5_land + '/model_data/ds_model_coef_geo.hdf5', 'w') as f:
    for ife in range(len(var_obj_all)):

        # Coefficients
        coef_am_mat = np.copy(mat_init)
        coef_am_mat[lmask_10km_ind, :] = var_obj_fill_all[ife][0]
        coef_am_mat = np.reshape(coef_am_mat, [lmask_10km_shape[0], lmask_10km_shape[1], 12])

        coef_pm_mat = np.copy(mat_init)
        coef_pm_mat[lmask_10km_ind, :] = var_obj_fill_all[ife][1]
        coef_pm_mat = np.reshape(coef_pm_mat, [lmask_10km_shape[0], lmask_10km_shape[1], 12])

        f.create_dataset('coef_am_' + str(ife+1).zfill(2), data=coef_am_mat, compression='lzf')
        f.create_dataset('coef_pm_' + str(ife+1).zfill(2), data=coef_pm_mat, compression='lzf')
        del(coef_am_mat, coef_pm_mat)

        # Statistical metrics
        stat_am_mat = np.copy(mat_init)
        stat_am_mat[lmask_10km_ind, :] = var_obj_stat_all[ife][0]
        stat_am_mat = np.reshape(stat_am_mat, [lmask_10km_shape[0], lmask_10km_shape[1], 12])

        stat_pm_mat = np.copy(mat_init)
        stat_pm_mat[lmask_10km_ind, :] = var_obj_stat_all[ife][1]
        stat_pm_mat = np.reshape(stat_pm_mat, [lmask_10km_shape[0], lmask_10km_shape[1], 12])

        f.create_dataset('stat_am_' + str(ife+1).zfill(2), data=stat_am_mat, compression='lzf')
        f.create_dataset('stat_pm_' + str(ife+1).zfill(2), data=stat_pm_mat, compression='lzf')
        del(stat_am_mat, stat_pm_mat)

        print(ife)

f.close()

del(var_obj_all, var_obj_fill_all, var_obj_stat_all)


########################################################################################################################
# 7. Reproject the model data into 9 km EASE-grid projection

f_coef = h5py.File(path_era5_land + '/model_data/ds_model_coef_geo.hdf5', 'r')
varname_list_coef = list(f_coef.keys())
var_obj_shape = f_coef[varname_list_coef[0]][()].shape

var_obj_dis_ease_all = []
for ife in range(len(varname_list_coef)):
    var_obj = f_coef[varname_list_coef[ife]][()]
    var_obj = np.concatenate((var_obj, np.expand_dims(var_obj[:, 0, :], axis=1)), axis=1)
    var_obj_dis_row = np.array([np.nanmean(var_obj[[x, x+1], :, y], axis=0)
                                for y in range(var_obj_shape[2]) for x in range(var_obj_shape[0]-1)])
    var_obj_dis = np.array([np.nanmean(var_obj_dis_row[:, [x, x+1]], axis=1) for x in range(var_obj_shape[1])])
    var_obj_dis = np.transpose(var_obj_dis, (1, 0))
    var_obj_dis = np.reshape(var_obj_dis, (12, var_obj_shape[0]-1, var_obj_shape[1]))
    var_obj_dis = np.transpose(var_obj_dis, (1, 2, 0))

    var_obj_dis_row_ease = np.array([np.nanmean(var_obj_dis[row_world_ease_9km_from_geo_10km_ind[x], :, y], axis=0)
                                     for y in range(var_obj_shape[2]) for x in range(len(lat_world_ease_9km))])
    var_obj_dis_ease = np.array([np.nanmean(var_obj_dis_row_ease[:, col_world_ease_9km_from_geo_10km_ind[x]], axis=1)
                                 for x in range(len(lon_world_ease_9km))])
    var_obj_dis_ease = np.transpose(var_obj_dis_ease, (1, 0))
    var_obj_dis_ease = np.reshape(var_obj_dis_ease, (12, len(lat_world_ease_9km), len(lon_world_ease_9km)))
    var_obj_dis_ease = np.transpose(var_obj_dis_ease, (1, 2, 0))

    var_obj_dis_ease_all.append(var_obj_dis_ease)
    del(var_obj, var_obj_dis_row, var_obj_dis, var_obj_dis_row_ease, var_obj_dis_ease)
    print(ife)


with h5py.File(path_era5_land + '/model_data/ds_model_coef.hdf5', 'w') as f:
    for ife in range(len(var_obj_dis_ease_all)):
        f.create_dataset(varname_list_coef[ife], data=var_obj_dis_ease_all[ife], compression='lzf')
        print(ife)
f.close()







# len_all_lat = np.array([len(row_world_ease_9km_from_geo_10km_ind[x]) for x in range(len(row_world_ease_9km_from_geo_10km_ind))])
# len_all_lat_ind = np.where(len_all_lat == 0)[0]
# len_all_lon = np.array([len(col_world_ease_9km_from_geo_10km_ind[x]) for x in range(len(col_world_ease_9km_from_geo_10km_ind))])
# len_all_lon_ind = np.where(len_all_lon == 0)[0]

f_coef = h5py.File(path_era5_land + '/model_data/ds_model_coef.hdf5', 'r')
varname_list_coef = list(f_coef.keys())

var_obj_all = []
for x in range(12):
    var_obj = f_coef[varname_list_coef[x+24]][:, :, ::2]
    var_obj = np.nanmean(var_obj, axis=2)
    var_obj_all.append(var_obj)
    del(var_obj)
    print(varname_list_coef[x+24])


file_name = sorted(glob.glob(path_era5_land + '/data/model_output/*'))
file_coef = file_name[:len(file_name)//2]
file_stat = file_name[len(file_name)//2:]

var_obj_all = []
for x in range(12):
    f_coef = h5py.File(file_stat[x], 'r')
    varname_list_coef = list(f_coef.keys())
    var_obj = f_coef[varname_list_coef[0]][864817, :]
    var_obj = np.nanmean(var_obj)
    # var_obj = np.nanmean(np.nanmean(var_obj, axis=1), axis=0)
    var_obj_all.append(var_obj)
    del(var_obj)
    print(file_stat[x])

