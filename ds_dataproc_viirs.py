import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import h5py
import calendar
import gzip
import rasterio
from rasterio.transform import Affine
import scipy.io
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################
# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data/nldas'
# Path of source viirs data
path_modis = '/Volumes/SBac/data/MODIS/HDF_Data'
# Path of viirs data for SM downscaling model input
path_modis_ip = '/Volumes/SBac/data/MODIS/Model_Input'
# Path of source output MODIS data
path_modis_op = '/Users/binfang/Downloads/Processing/Model_Output'
# Path of downscaled SM
path_smap_sm_ds = '/Users/binfang/Downloads/Processing/Downscale'
# Path of 9 km SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'
# Path of VIIRS data
path_viirs = '/Volumes/MyPassport/SMAP_Project/Datasets/VIIRS'
# Path of VIIRS data regridded output
path_viirs_output = '/Users/binfang/Downloads/Processing/VIIRS/'

smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_conus_max', 'lat_conus_min', 'lon_conus_max', 'lon_conus_min',
                'lat_conus_geo_400m', 'lon_conus_geo_400m', 'lat_conus_ease_400m', 'lon_conus_ease_400m',
                'lat_conus_ease_9km', 'lon_conus_ease_9km', 'lat_conus_ease_12_5km', 'lon_conus_ease_12_5km',
                'row_conus_ease_400m_ind', 'col_conus_ease_400m_ind', 'row_conus_ease_9km_ind', 'col_conus_ease_9km_ind',
                'row_conus_ease_400m_from_geo_400m_ind', 'col_conus_ease_400m_from_geo_400m_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2015-01-01'
end_date = '2019-12-31'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))

# Count how many days for a specific year
yearname = np.linspace(2015, 2020, 6, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    # if idt == 0:
    #     f_date = datetime.date(yearname[idt], monthnum[3], 1)
    #     l_date = datetime.date(yearname[idt], monthnum[-1], 31)
    #     delta_1y = l_date - f_date
    #     daysofyear.append(delta_1y.days + 1)
    # else:
        f_date = datetime.date(yearname[idt], monthnum[0], 1)
        l_date = datetime.date(yearname[idt], monthnum[-1], 31)
        delta_1y = l_date - f_date
        daysofyear.append(delta_1y.days + 1)

daysofyear = np.asarray(daysofyear)

# Find the indices of each month in the list of days between 2015 - 2018
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

# Generate index for viirs LAI
viirs_lai_days = np.arange(1, 366, 8)
oneyeardays_sq = np.arange(1, 367)
# viirs_lai_days_ind = np.array([viirs_lai_days[np.where(np.absolute(oneyeardays_sq[x] - viirs_lai_days) ==
#                                                 np.amin((np.absolute(oneyeardays_sq[x] - viirs_lai_days))))].item(0)
#                        for x in range(len(oneyeardays_sq))])

# VIIRS data description:
# Dx=3750
# Dy=3750
# dlat/dlon = 0.04 degrees
# Lower left lat / lon
# T028: Lat = 45.002, -134.998
# T029: Lat = 45.002, -119.998
# T030: Lat = 45.002, -104.998
# T031: Lat = 45.002, -89.998
# T032: Lat = 45.002, -74.998
# T052: Lat = 30.002, -134.998
# T053: Lat = 30.002, -119.998
# T054: Lat = 30.002, -104.998
# T055: Lat = 30.002, -89.998
# T056: Lat = 30.002, -74.998
# T077: Lat = 15.002, -119.998
# T078: Lat = 15.002, -104.998
# T079: Lat = 15.002, -89.998
# T080: Lat = 15.002, -74.998

# Position of the viirs tiles:
# [[T028, T029, T030, T031, T032],
# [[T052, T053, T054, T055, T056],
# [[N/A, T077, T078, T079, T080]]

lat_viirs_tile_min = 15.002
lon_viirs_tile_min = -134.998
size_viirs_tile = 3750
cellsize_400m = 0.004

lat_viirs_tile_max = round(lat_viirs_tile_min + (size_viirs_tile*3-1)*cellsize_400m, 3)
lon_viirs_tile_max = round(lon_viirs_tile_min + (size_viirs_tile*5-1)*cellsize_400m, 3)
lat_viirs_tile = np.round(np.linspace(lat_viirs_tile_max, lat_viirs_tile_min, num=size_viirs_tile*3), decimals=3)
lon_viirs_tile = np.round(np.linspace(lon_viirs_tile_min, lon_viirs_tile_max, num=size_viirs_tile*5), decimals=3)

lat_viirs_tile_start_ind = np.where(lat_viirs_tile == lat_conus_geo_400m[0])[0][0]
lat_viirs_tile_end_ind = np.where(lat_viirs_tile == lat_conus_geo_400m[-1])[0][0]
lon_viirs_tile_start_ind = np.where(lon_viirs_tile == lon_conus_geo_400m[0])[0][0]
lon_viirs_tile_end_ind = np.where(lon_viirs_tile == lon_conus_geo_400m[-1])[0][0]


########################################################################################################################
# 1. Process viirs data of 2019
start_day = np.cumsum(daysofmonth_seq[0:3, 4])[-1]
end_day = np.cumsum(daysofmonth_seq[0:9, 4])[-1]
datename_seq_2019 = np.arange(start_day+1, end_day+2) # Add 1 day for Sept. 1st

subfolders = sorted(glob.glob(path_viirs + '/2019/*/'))
subfolders_name = [subfolders[x].split('/')[-2] for x in range(len(subfolders)-1)]

# Extract the day/night viirs LST data and LAI data and merge/regrid to EASE grid V2 projection in CONUS region
viirs_mat_initial = np.empty([size_viirs_tile, size_viirs_tile], dtype='float32')
viirs_mat_initial[:] = np.nan
viirs_mat_fill = np.empty([size_viirs_tile, 1250], dtype='float32')
viirs_mat_fill[:] = np.nan

# 1.1 Process VIIRS LST data
# Find the matched files for each day between March - September for each tile (LST)
file_lstd_all = []
file_lstn_all = []
file_lstd_ind_all = []
file_lstn_ind_all = []
for ifo in range(len(subfolders)-1): # Exclude the LAI data folder
    file_lstd = sorted(glob.glob(subfolders[ifo] + '/*DAY*'))
    file_lstn = sorted(glob.glob(subfolders[ifo] + '/*NIGHT*'))

    file_lstd_datename = np.array([int(file_lstd[x].split('/')[-1].split('.')[0].split('_')[-1][-3:])
                                   for x in range(len(file_lstd))])
    file_lstn_datename = np.array([int(file_lstn[x].split('/')[-1].split('.')[0].split('_')[-1][-3:])
                                   for x in range(len(file_lstn))])
    file_lstd_ind = [np.where(file_lstd_datename == datename_seq_2019[x])[0] for x in range(len(datename_seq_2019))]
    file_lstn_ind = [np.where(file_lstn_datename == datename_seq_2019[x])[0] for x in range(len(datename_seq_2019))]
    file_lstd_all.append(file_lstd)
    file_lstn_all.append(file_lstn)
    file_lstd_ind_all.append(file_lstd_ind)
    file_lstn_ind_all.append(file_lstn_ind)
    del(file_lstd_datename, file_lstn_datename, file_lstd, file_lstn, file_lstd_ind, file_lstn_ind)


# Keyword arguments of the tiff file to write
kwargs = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_400m), 'height': len(lat_conus_ease_400m),
          'count': 2, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(400.358009339824, 0.0, -12060785.03616676, 0.0, -400.358009339824, 5854435.16975144)
          }

# Extract the viirs LST day/night data and write to geotiff files
for idt in range(len(datename_seq_2019)):
    viirs_mat_day_daily = []
    viirs_mat_night_daily = []
    for ifo in range(len(subfolders) - 1):  # Exclude the LAI data folder
        # LST Day
        if len(file_lstd_ind_all[ifo][idt]) != 0:
            with gzip.open(file_lstd_all[ifo][file_lstd_ind_all[ifo][idt][0]], 'rb') as file:
                viirs_mat_day = np.frombuffer(file.read(), dtype='f')
                viirs_mat_day = np.reshape(viirs_mat_day, (size_viirs_tile, size_viirs_tile)).copy()
                viirs_mat_day[viirs_mat_day <= 0] = np.nan
            file.close()
            viirs_mat_day_daily.append(viirs_mat_day)
            print(file_lstd_all[ifo][file_lstd_ind_all[ifo][idt][0]])
            del(viirs_mat_day)
        else:
            viirs_mat_day_daily.append(viirs_mat_initial)

        # LST Night
        if len(file_lstn_ind_all[ifo][idt]) != 0:
            with gzip.open(file_lstn_all[ifo][file_lstn_ind_all[ifo][idt][0]], 'rb') as file:
                viirs_mat_night = np.frombuffer(file.read(), dtype='f')
                viirs_mat_night = np.reshape(viirs_mat_night, (size_viirs_tile, size_viirs_tile)).copy()
                viirs_mat_night[viirs_mat_night <= 0] = np.nan
            file.close()
            viirs_mat_night_daily.append(viirs_mat_night)
            print(file_lstn_all[ifo][file_lstn_ind_all[ifo][idt][0]])
            del(viirs_mat_night)
        else:
            viirs_mat_night_daily.append(viirs_mat_initial)

    viirs_mat_day_daily.insert(10, viirs_mat_initial) # Insert the empty matrix to the first place of the third line
    viirs_mat_night_daily.insert(10, viirs_mat_initial)
    viirs_mat_daily = viirs_mat_day_daily + viirs_mat_night_daily

    # Merge tile column wise first, and row wise (day and night layers)
    viirs_mat_ease_2lyr = []
    for ilr in range(2):
        viirs_mat_daily_line1 = np.concatenate(viirs_mat_daily[15*ilr+0:15*ilr+5], axis=1)
        viirs_mat_daily_line2 = np.concatenate(viirs_mat_daily[15*ilr+5:15*ilr+10], axis=1)
        viirs_mat_daily_line3 = np.concatenate(viirs_mat_daily[15*ilr+10:15*ilr+15], axis=1)
        viirs_mat_merge = \
            np.concatenate([viirs_mat_daily_line1, viirs_mat_daily_line2, viirs_mat_daily_line3], axis=0)

        # Subset the merged viirs data in CONUS area
        viirs_mat_sub = viirs_mat_merge[lat_viirs_tile_start_ind:lat_viirs_tile_end_ind+1,
                            lon_viirs_tile_start_ind:lon_viirs_tile_end_ind+1]

        # Regrid the viirs data in EASE grid projection
        viirs_mat_ease = np.array \
            ([np.nanmean(viirs_mat_sub[row_conus_ease_400m_from_geo_400m_ind[x], :], axis=0)
              for x in range(len(lat_conus_ease_400m))])
        viirs_mat_ease = np.array \
            ([np.nanmean(viirs_mat_ease[:, col_conus_ease_400m_from_geo_400m_ind[y]], axis=1)
              for y in range(len(lon_conus_ease_400m))])
        viirs_mat_ease = np.fliplr(np.rot90(viirs_mat_ease, 3))

        viirs_mat_ease_2lyr.append(viirs_mat_ease)
        del(viirs_mat_daily_line1, viirs_mat_daily_line2, viirs_mat_daily_line3, viirs_mat_merge,
            viirs_mat_sub, viirs_mat_ease)

    viirs_mat_ease_2lyr = np.array(viirs_mat_ease_2lyr)

    # Write to Geotiff file
    with rasterio.open(path_viirs_output + '/Model_Input/LST/2019/' + 'viirs_lst_2019' +
                        str(datename_seq_2019[idt]).zfill(3) + '.tif', 'w', **kwargs) as output_ds:
        output_ds.write(viirs_mat_ease_2lyr)

    del(viirs_mat_ease_2lyr, viirs_mat_day_daily, viirs_mat_night_daily, viirs_mat_daily)



# 1.2 Process LAI data
# Find the matched tile files for each day (LAI)
file_lai = sorted(glob.glob(subfolders[14] + '/*'))
file_lai_name = [file_lai[x].split('.')[0][-12:] for x in range(len(file_lai))]

# Find the location of each file
file_lai_ind_all = []
for idt in range(len(viirs_lai_days)):
    file_lai_tos = ['2019' + str(viirs_lai_days[idt]).zfill(3) + '_' + subfolders_name[y]
                    for y in range(len(subfolders_name))] # All tile names of one day
    file_lai_ind = [file_lai_name.index(i) for i in file_lai_name for x in range(len(file_lai_tos))
                    if file_lai_tos[x] in i]
    file_lai_ind_all.append(file_lai_ind)
    del(file_lai_ind)

# Keyword arguments of the tiff file to write
kwargs = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_400m), 'height': len(lat_conus_ease_400m),
          'count': 1, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(400.358009339824, 0.0, -12060785.03616676, 0.0, -400.358009339824, 5854435.16975144)
          }

# Extract the viirs LAI data and write to geotiff files
for idt in range(len(viirs_lai_days)):
    viirs_mat_lai_daily = []
    for ifo in range(len(subfolders_name)):
        with gzip.open(file_lai[file_lai_ind_all[idt][ifo]], 'rb') as file:
            viirs_mat_lai = np.frombuffer(file.read(), dtype='f')
            viirs_mat_lai = np.reshape(viirs_mat_lai, (size_viirs_tile, size_viirs_tile)).copy()
            viirs_mat_lai = np.flipud(viirs_mat_lai)
            viirs_mat_lai[viirs_mat_lai <= 0] = np.nan
        file.close()
        viirs_mat_lai_daily.append(viirs_mat_lai)
        print(file_lai[file_lai_ind_all[idt][ifo]])
        del(viirs_mat_lai)

    viirs_mat_lai_daily.insert(10, viirs_mat_initial) # Insert the empty matrix to the first place of the third line

    # Fix the drift issue of tile 77 and 78
    image = viirs_mat_lai_daily[11][:, 0:2500]
    viirs_mat_lai_daily[11] = np.concatenate((viirs_mat_fill, image), axis=1)
    # image_new = np.concatenate((image, viirs_mat_lai_daily[12]), axis=1)

    # viirs_mat_lai_daily[11] = viirs_mat_initial
    # Merge tile column wise first, and row wise
    viirs_mat_lai_daily_line1 = np.concatenate(viirs_mat_lai_daily[0:5], axis=1)
    viirs_mat_lai_daily_line2 = np.concatenate(viirs_mat_lai_daily[5:10], axis=1)
    viirs_mat_lai_daily_line3 = np.concatenate(viirs_mat_lai_daily[10:15], axis=1)
    viirs_mat_lai_daily_merge = \
        np.concatenate([viirs_mat_lai_daily_line1, viirs_mat_lai_daily_line2, viirs_mat_lai_daily_line3], axis=0)

    # Subset the merged viirs data in CONUS area
    viirs_mat_sub = viirs_mat_lai_daily_merge[lat_viirs_tile_start_ind:lat_viirs_tile_end_ind+1,
                            lon_viirs_tile_start_ind:lon_viirs_tile_end_ind+1]

    # Regrid the viirs data in EASE grid projection
    viirs_mat_ease = np.array \
        ([np.nanmean(viirs_mat_sub[row_conus_ease_400m_from_geo_400m_ind[x], :], axis=0)
            for x in range(len(lat_conus_ease_400m))])
    viirs_mat_ease = np.array \
        ([np.nanmean(viirs_mat_ease[:, col_conus_ease_400m_from_geo_400m_ind[y]], axis=1)
            for y in range(len(lon_conus_ease_400m))])
    viirs_mat_ease = np.fliplr(np.rot90(viirs_mat_ease, 3))
    viirs_mat_ease = np.expand_dims(viirs_mat_ease, axis=0)

    del(viirs_mat_lai_daily_line1, viirs_mat_lai_daily_line2, viirs_mat_lai_daily_line3, viirs_mat_lai_daily,
        viirs_mat_lai_daily_merge, viirs_mat_sub)

    # Write to Geotiff file
    with rasterio.open(path_viirs_output + '/Model_Input/LAI/2019/' + 'viirs_lai_2019' +
                       str(viirs_lai_days[idt]).zfill(3) + '.tif', 'w', **kwargs) as output_ds:
        output_ds.write(viirs_mat_ease)

    del(viirs_mat_ease)


########################################################################################################################
# 2. Process viirs data of 2019

# 2.1 Process VIIRS LST data
# Keyword arguments of the tiff file to write
kwargs = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_400m), 'height': len(lat_conus_ease_400m),
          'count': 2, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(400.358009339824, 0.0, -12060785.03616676, 0.0, -400.358009339824, 5854435.16975144)
          }

file_lstd = sorted(glob.glob(path_viirs + '/2018/LST_mat/*day*'))
file_lstn = sorted(glob.glob(path_viirs + '/2018/LST_mat/*night*'))
month_2019 = daysofmonth_seq[3:9, 4]
month_2019_cumsum = np.cumsum(month_2019)
month_2019_cumsum = [0] + list(month_2019_cumsum)
datename_seq_2019_bymonth = [datename_seq_2019[month_2019_cumsum[x]:month_2019_cumsum[x+1]] for x in range(len(month_2019_cumsum)-1)]


for imo in range(6):
    file_list_1month = np.arange(imo, 84, 6)
    file_lstd_1month = [file_lstd[file_list_1month[x]] for x in range(len(file_list_1month))]
    file_lstn_1month = [file_lstn[file_list_1month[x]] for x in range(len(file_list_1month))]

    viirs_mat_day_all = []
    viirs_mat_night_all = []
    for ife in range(len(file_lstd_1month)):
        # LST Day
        viirs_mat_day = h5py.File(file_lstd_1month[ife], 'r')
        viirs_mat_day = viirs_mat_day['viirs_lst_day'][()]
        viirs_mat_day_all.append(viirs_mat_day)
        print(os.path.basename(file_lstd_1month[ife]))

        # LST Night
        viirs_mat_night = h5py.File(file_lstn_1month[ife], 'r')
        viirs_mat_night = viirs_mat_night['viirs_lst_night'][()]
        viirs_mat_night_all.append(viirs_mat_night)
        print(os.path.basename(file_lstn_1month[ife]))

        del(viirs_mat_day, viirs_mat_night)


    # Merge tiles from each day and write to geotiff files
    for idt in range(month_2019[imo]):

        viirs_mat_day_1day = []
        viirs_mat_night_1day = []
        for ifo in range(len(viirs_mat_day_all)):
            # Day
            viirs_mat_day_daily_1tile = viirs_mat_day_all[ifo][idt, :, :]
            viirs_mat_day_daily_1tile = np.fliplr(np.rot90(viirs_mat_day_daily_1tile, -1))
            viirs_mat_day_1day.append(viirs_mat_day_daily_1tile)

            # Night
            viirs_mat_night_daily_1tile = viirs_mat_night_all[ifo][idt, :, :]
            viirs_mat_night_daily_1tile = np.fliplr(np.rot90(viirs_mat_night_daily_1tile, -1))
            viirs_mat_night_1day.append(viirs_mat_night_daily_1tile)

            del(viirs_mat_day_daily_1tile, viirs_mat_night_daily_1tile)

        viirs_mat_day_1day.insert(10, viirs_mat_initial)
        viirs_mat_night_1day.insert(10, viirs_mat_initial)
        viirs_mat_daily = viirs_mat_day_1day + viirs_mat_night_1day

        # Merge tile column wise first, and row wise (day and night layers)
        viirs_mat_ease_2lyr = []
        for ilr in range(2):
            viirs_mat_daily_line1 = np.concatenate(viirs_mat_daily[15 * ilr + 0:15 * ilr + 5], axis=1)
            viirs_mat_daily_line2 = np.concatenate(viirs_mat_daily[15 * ilr + 5:15 * ilr + 10], axis=1)
            viirs_mat_daily_line3 = np.concatenate(viirs_mat_daily[15 * ilr + 10:15 * ilr + 15], axis=1)
            viirs_mat_merge = \
                np.concatenate([viirs_mat_daily_line1, viirs_mat_daily_line2, viirs_mat_daily_line3], axis=0)

            # Subset the merged viirs data in CONUS area
            viirs_mat_sub = viirs_mat_merge[lat_viirs_tile_start_ind:lat_viirs_tile_end_ind + 1,
                            lon_viirs_tile_start_ind:lon_viirs_tile_end_ind + 1]

            # Regrid the viirs data in EASE grid projection
            viirs_mat_ease = np.array \
                ([np.nanmean(viirs_mat_sub[row_conus_ease_400m_from_geo_400m_ind[x], :], axis=0)
                  for x in range(len(lat_conus_ease_400m))])
            viirs_mat_ease = np.array \
                ([np.nanmean(viirs_mat_ease[:, col_conus_ease_400m_from_geo_400m_ind[y]], axis=1)
                  for y in range(len(lon_conus_ease_400m))])
            viirs_mat_ease = np.fliplr(np.rot90(viirs_mat_ease, 3))

            viirs_mat_ease_2lyr.append(viirs_mat_ease)
            del(viirs_mat_daily_line1, viirs_mat_daily_line2, viirs_mat_daily_line3, viirs_mat_merge,
                 viirs_mat_sub, viirs_mat_ease)

        viirs_mat_ease_2lyr = np.array(viirs_mat_ease_2lyr).astype('float32')


        # Write to Geotiff file
        with rasterio.open(path_viirs_output + '/Model_Input/LST/2018/' + 'viirs_lst_2018' +
                           str(datename_seq_2019_bymonth[imo][idt]).zfill(3) + '.tif', 'w', **kwargs) as output_ds:
            output_ds.write(viirs_mat_ease_2lyr)

        print('viirs_lst_2018' + str(datename_seq_2019_bymonth[imo][idt]).zfill(3))
        del(viirs_mat_ease_2lyr, viirs_mat_day_1day, viirs_mat_night_1day, viirs_mat_daily)


    del(file_list_1month, file_lstd_1month, file_lstn_1month, viirs_mat_day_all, viirs_mat_night_all)


# 2.2 Process LAI data

# Keyword arguments of the tiff file to write
kwargs = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_400m), 'height': len(lat_conus_ease_400m),
          'count': 1, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(400.358009339824, 0.0, -12060785.03616676, 0.0, -400.358009339824, 5854435.16975144)
          }

file_lai = sorted(glob.glob(path_viirs + '/2018/LAI_mat_mosaic/*'))

for ife in range(len(file_lai)):
    # LST Day
    viirs_mat_lai = scipy.io.loadmat(file_lai[ife])
    viirs_mat_lai = viirs_mat_lai['viirs_lai_400m'][()]
    # Fill the missing columns
    viirs_mat_lai[:, 1248] = viirs_mat_lai[:, 1247]
    viirs_mat_lai[:, 1249] = viirs_mat_lai[:, 1250]
    viirs_mat_lai[:, 4999] = np.nanmean(viirs_mat_lai[:, 4998:5001], axis=1)
    viirs_mat_lai[:, 8749] = np.nanmean(viirs_mat_lai[:, 8748:8751], axis=1)
    viirs_mat_lai[:, 12499] = np.nanmean(viirs_mat_lai[:, 12498:12501], axis=1)
    viirs_mat_lai[2000, :] = np.nanmean(viirs_mat_lai[1999:2002, :], axis=0)

    # Regrid the viirs data in EASE grid projection
    viirs_mat_ease = np.array \
        ([np.nanmean(viirs_mat_lai[row_conus_ease_400m_from_geo_400m_ind[x], :], axis=0)
            for x in range(len(lat_conus_ease_400m))])
    viirs_mat_ease = np.array \
        ([np.nanmean(viirs_mat_ease[:, col_conus_ease_400m_from_geo_400m_ind[y]], axis=1)
            for y in range(len(lon_conus_ease_400m))])
    viirs_mat_ease = np.fliplr(np.rot90(viirs_mat_ease, 3))
    viirs_mat_ease = np.expand_dims(viirs_mat_ease, axis=0)
    viirs_mat_ease = viirs_mat_ease.astype('float32')

    # Write to Geotiff file
    with rasterio.open(path_viirs_output + '/Model_Input/LAI/2018/' + 'viirs_' +
                       os.path.basename(file_lai[ife]).split('.')[0] + '.tif', 'w', **kwargs) as output_ds:
        output_ds.write(viirs_mat_ease)

    del(viirs_mat_lai, viirs_mat_ease)

    print(os.path.basename(file_lai[ife]).split('.')[0])


# col_mean = np.nanmean(viirs_mat_lai, axis=0)
# row_mean = np.nanmean(viirs_mat_lai, axis=1)
# np.where(np.isnan(col_mean))
# np.where(np.isnan(row_mean))

