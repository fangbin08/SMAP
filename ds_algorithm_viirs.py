import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import h5py
import calendar
import gzip
import gdal
import osr
import tarfile
from pyproj import Transformer
import rasterio
from rasterio.transform import Affine
import scipy.io
import itertools
import pandas as pd
import scipy.ndimage
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################
# (Function 1) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind

########################################################################################################################
# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of model data
path_model = '/Volumes/Elements/Datasets/model_data/gldas/'
# Path of model output SM
path_model_sm = '/Volumes/Elements2/VIIRS/SM_model/'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/Elements2/VIIRS/SM_downscaled/'
# Path of 9 km SMAP SM
path_smap = '/Volumes/Elements/Datasets/SMAP'
# Path of VIIRS data
path_viirs_lst = '/Volumes/Seagate_6TB/VIIRS/LST/'
# Path of VIIRS data regridded output
path_viirs_output = '/Users/binfang/Downloads/Processing/VIIRS/'
path_lst_geo = '/Volumes/Elements2/VIIRS/LST_geo/'
path_lst_ease = '/Volumes/Elements2/VIIRS/LST_ease/'
path_viirs_lai = '/Volumes/Seagate_6TB/VIIRS/LAI/'
path_lai_geo = '/Volumes/Elements2/VIIRS/LAI_geo/'
path_lai_ease = '/Volumes/Elements2/VIIRS/LAI_ease/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_geo_400m', 'lon_world_geo_400m', 'lat_world_ease_400m', 'lon_world_ease_400m',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_12_5km', 'lon_world_ease_12_5km',
                'row_world_ease_400m_ind', 'col_world_ease_400m_ind', 'row_world_ease_9km_ind', 'col_world_ease_9km_ind',
                'row_world_ease_400m_from_geo_400m_ind', 'col_world_ease_400m_from_geo_400m_ind',
                'col_world_ease_400m_from_9km_ind', 'row_world_ease_400m_from_9km_ind',
                'row_world_ease_400m_from_25km_ind', 'col_world_ease_400m_from_25km_ind',
                'row_world_ease_9km_from_400m_ext33km_ind', 'col_world_ease_9km_from_400m_ext33km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

row_world_geo_400m_ind = np.arange(len(lat_world_geo_400m))
col_world_geo_400m_ind = np.arange(len(lon_world_geo_400m))

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2010-01-01'
end_date = '2023-12-31'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
date_seq_ymd = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))
    date_seq_ymd.append(date_str.strftime('%Y%m%d'))

# Count how many days for a specific year
yearname = np.linspace(2010, 2023, 14, dtype='int')
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

date_seq_array = np.array([int(date_seq[x]) for x in range(len(date_seq))])
daysofyear_cumsum = np.cumsum(daysofyear)
date_seq_array_cumsum = np.hsplit(date_seq_array, daysofyear_cumsum)[:-1] # split by each month
date_seq_ymd_group = np.split(date_seq_ymd, daysofyear_cumsum)[:-1]

# Generate index for viirs LAI
viirs_lai_days = np.arange(1, 366, 8)
oneyeardays_sq = np.arange(1, 367)
# Find the cloest day to any day with LAI data
viirs_lai_days_name = np.array([viirs_lai_days[np.where(np.absolute(oneyeardays_sq[x] - viirs_lai_days)
                               == np.amin((np.absolute(oneyeardays_sq[x] - viirs_lai_days))))].item(0)
                               for x in range(len(oneyeardays_sq))])
viirs_lai_days_ind = np.array([np.where(np.absolute(oneyeardays_sq[x] - viirs_lai_days)
                               == np.amin((np.absolute(oneyeardays_sq[x] - viirs_lai_days))))[0][0]
                               for x in range(len(oneyeardays_sq))])

########################################################################################################################
# 0. Process viirs LST data

# For 400 m viirs data: lat: -60~75, lon: -180~180
# Geographic projection extent:
# Geographic projection dimensions: 33750, 90000 (full size dimensions: 45000, 90000)
# Geographic grid size: 3750*3750
# Row:9, col: 24

# Ease-grid projection dimensions: 36540, 86760
# Ease-grid grid size: 4060*3615
# Row:9, col: 24
lat_extent_max = 75
lat_extent_min = -60
lon_extent_max = 180
lon_extent_min = -180
interdist_ease_400m = 400.358009339824

# Subset 400 m lat/lon tables by viirs real extent
[lat_world_geo_400m_sub, row_world_geo_400m_ind_sub, lon_world_geo_400m_sub, col_world_geo_400m_ind_sub] = \
    coordtable_subset(lat_world_geo_400m, lon_world_geo_400m,
                      lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min)

# Remove out-of-range indices of 400 m
row_ind_max = row_world_geo_400m_ind_sub[-1]
row_ind_min = row_world_geo_400m_ind_sub[0]

row_world_ease_400m_from_geo_400m_ind_rev = []
for i in range(len(row_world_ease_400m_from_geo_400m_ind)):
    row_ind = row_world_ease_400m_from_geo_400m_ind[i][(row_world_ease_400m_from_geo_400m_ind[i] <= row_ind_max) &
                                                      (row_world_ease_400m_from_geo_400m_ind[i] >= row_ind_min)]
    row_world_ease_400m_from_geo_400m_ind_rev.append(row_ind)
    del(row_ind)
row_world_ease_400m_from_geo_400m_ind_rev = np.array(row_world_ease_400m_from_geo_400m_ind_rev, dtype=object)

# Divide row/col indices by image block indices
row_divide_ind = np.arange(0, len(row_world_ease_400m_from_geo_400m_ind_rev), 4060)
row_world_ease_400m_from_geo_400m_ind_group = np.split(row_world_ease_400m_from_geo_400m_ind_rev, row_divide_ind, axis=0)[1:]
col_divide_ind = np.arange(0, len(col_world_ease_400m_from_geo_400m_ind), 3615)
col_world_ease_400m_from_geo_400m_ind_group = np.split(col_world_ease_400m_from_geo_400m_ind, col_divide_ind, axis=0)[1:]

# Create tables for tile numbers
row_world_geo_400m_ind_tile = np.array([np.repeat(x, 3750) for x in range(9)]).ravel()
col_world_geo_400m_ind_tile = np.array([np.repeat(x, 3750) for x in range(24)]).ravel()
row_divide_ind_geo = np.arange(0, len(row_world_geo_400m_ind_sub), 3750)
row_world_geo_400m_ind_group = np.split(row_world_geo_400m_ind_sub, row_divide_ind_geo, axis=0)[1:]
col_divide_ind_geo = np.arange(0, len(col_world_geo_400m_ind_sub), 3750)
col_world_geo_400m_ind_group = np.split(col_world_geo_400m_ind_sub, col_divide_ind_geo, axis=0)[1:]

# Divide lat/lon tables by image block
row_divide_ind = np.arange(0, len(lat_world_ease_400m), 4060)
lat_world_ease_400m_group = np.split(lat_world_ease_400m, row_divide_ind, axis=0)[1:]
col_divide_ind = np.arange(0, len(lon_world_ease_400m), 3615)
lon_world_ease_400m_group = np.split(lon_world_ease_400m, col_divide_ind, axis=0)[1:]

viirs_mat_fill = np.empty([3750, 3750], dtype='float32')
viirs_mat_fill[:] = np.nan


########################################################################################################################
# 1.1 Combine tile data by day
tile_name_full = ['T' + str(x).zfill(3) for x in range(1, 217)]
utc_converter = np.tile(np.arange(-11, 13), 9)
overpass_name = ['_night', '_day']

for iyr in [12]:#range(len(yearname)):
    tile_name = sorted(glob.glob(path_viirs_lst + str(yearname[iyr]) + '/*', recursive=True))
    tile_name_base = [tile_name[x].split('/')[-1].split('.')[0][-4:] for x in range(len(tile_name))]
    tile_name_ind = [int(tile_name[x].split('/')[-1].split('.')[0][-4:][1:])-1 for x in range(len(tile_name))]
    tile_time_convert = utc_converter[tile_name_ind]
    overpass_time = (np.repeat(date_seq_array_cumsum[iyr], 2) * 10000 +
                     np.tile([130, 1330], len(date_seq_array_cumsum[iyr])))

    for ite in range(len(tile_name)):
        tar = tarfile.open(tile_name[ite], 'r')
        viirs_files = tar.getnames()
        viirs_files_time = [viirs_files[x].split('.')[0][-12:].replace('_', '') for x in range(len(viirs_files))]
        viirs_files_time_local = [datetime.datetime.strptime(viirs_files_time[x], '%Y%j%H%M') +
                                  datetime.timedelta(hours=int(tile_time_convert[ite])) for x in range(len(viirs_files_time))]
        viirs_files_time_local = [datetime.datetime.strftime(viirs_files_time_local[x], '%Y%j%H%M')
                                  for x in range(len(viirs_files_time_local))]
        viirs_files_time_local = np.array([int(viirs_files_time_local[x]) for x in range(len(viirs_files_time_local))])

        overpass_time_ind = np.array([np.argmin(np.absolute(viirs_files_time_local[x] - overpass_time)).item()
                            for x in range(len(viirs_files_time_local))])
        overpass_time_ind_unique = np.unique(overpass_time_ind)
        df_overpass_time_ind = pd.DataFrame({'number': overpass_time_ind})
        overpass_time_ind_group = df_overpass_time_ind.groupby(by='number').groups
        overpass_time_ind_group_ind = [value[1].tolist() for value in overpass_time_ind_group.items()]

        # Create the folder for tile data
        viirs_tile_output_path = path_lst_geo + str(yearname[iyr]) + '/' + tile_name_base[ite] + '/'
        if os.path.exists(viirs_tile_output_path) == False:
            os.makedirs(viirs_tile_output_path)
        else:
            pass

        for igp in range(len(overpass_time_ind_group_ind)):
            viirs_mat_all = []
            for ife in range(len(overpass_time_ind_group_ind[igp])):
                gzip_file = tar.extractfile(viirs_files[overpass_time_ind_group_ind[igp][ife]])
                with gzip.open(gzip_file, 'rb') as file:
                    viirs_mat = np.frombuffer(file.read(), dtype='f')
                    viirs_mat = np.reshape(viirs_mat, (3750, 3750)).copy()
                    viirs_mat[viirs_mat == -9999] = np.nan
                    viirs_mat_all.append(viirs_mat)
                file.close()
                del(gzip_file, viirs_mat)
                # print(viirs_files[overpass_time_ind_group_ind[igp][ife]])

            viirs_mat_day = np.nanmean(np.stack(viirs_mat_all), axis=0)
            viirs_mat_day = np.flipud(viirs_mat_day)

            # Write combined viirs daily data to file
            viirs_output_path = (viirs_tile_output_path + 'viirs_lst_' + str(overpass_time[overpass_time_ind_unique[igp]])[0:7] +
                                    str(overpass_name[overpass_time_ind_unique[igp] % 2]))

            with h5py.File(viirs_output_path + '.hdf5', 'w') as f:
                f.create_dataset('viirs_lst', data=viirs_mat_day, compression='lzf')
            f.close()
            print(viirs_output_path)
            del(viirs_mat_day, viirs_mat_all, viirs_output_path)

        tar.close()

        del(viirs_files, viirs_files_time, viirs_files_time_local, overpass_time_ind, df_overpass_time_ind,
            overpass_time_ind_group, overpass_time_ind_group_ind, viirs_tile_output_path)




# 1.2 Reproject 400 m viirs data from geographic to EASR-grid projection
overpass_name = ['_night', '_day']

for iyr in [12]:#range(len(yearname)):
    tile_name = sorted(glob.glob(path_viirs_lst + str(yearname[iyr]) + '/*/', recursive=True))
    tile_name_base = [int(tile_name[x].split('/')[-2][1:]) for x in range(len(tile_name))]

    for irow in range(9):
        for icol in range(24):

            tile_num = irow * 24 + icol + 1
            if tile_num in tile_name_base:

                # Extract row/col tables from grouped files for resampling
                row_world_ease_400m_from_geo_400m_ind_torep = row_world_ease_400m_from_geo_400m_ind_group[irow]
                col_world_ease_400m_from_geo_400m_ind_torep = col_world_ease_400m_from_geo_400m_ind_group[icol]

                row_world_ease_400m_from_geo_400m_ind_torep_num = np.array(list(itertools.chain(*row_world_ease_400m_from_geo_400m_ind_torep)))
                row_world_ease_400m_from_geo_400m_ind_torep_tile = row_world_geo_400m_ind_tile[row_world_ease_400m_from_geo_400m_ind_torep_num
                                                                                               - row_ind_min]
                row_world_ease_400m_from_geo_400m_ind_torep_tile_unique = np.unique(row_world_ease_400m_from_geo_400m_ind_torep_tile)
                col_world_ease_400m_from_geo_400m_ind_torep_num = np.array(list(itertools.chain(*col_world_ease_400m_from_geo_400m_ind_torep)))
                col_world_ease_400m_from_geo_400m_ind_torep_tile = col_world_geo_400m_ind_tile[col_world_ease_400m_from_geo_400m_ind_torep_num]
                col_world_ease_400m_from_geo_400m_ind_torep_tile_unique = np.unique(col_world_ease_400m_from_geo_400m_ind_torep_tile)

                # Convert from world row/col index tables to local (indices start from upper left tile)
                row_world_ease_400m_from_geo_400m_ind_torep_local = (row_world_ease_400m_from_geo_400m_ind_torep - row_ind_min -
                                                                     row_world_ease_400m_from_geo_400m_ind_torep_tile_unique[0] * 3750)
                col_world_ease_400m_from_geo_400m_ind_torep_local = (col_world_ease_400m_from_geo_400m_ind_torep -
                                                                     col_world_ease_400m_from_geo_400m_ind_torep_tile_unique[0] * 3750)


                # Load all adjacent tiles of data for reprojecting
                tiles_load = (row_world_ease_400m_from_geo_400m_ind_torep_tile_unique[:, None] * 24 +
                              col_world_ease_400m_from_geo_400m_ind_torep_tile_unique + 1).ravel()
                tiles_load = [str(tiles_load[x]).zfill(3) for x in range(len(tiles_load))]


                for idt in range(daysofyear[iyr]):

                    viirs_lst_delta_tiles_am = []
                    viirs_lst_delta_tiles_pm = []
                    for ite in range(len(tiles_load)):

                        if idt != daysofyear[iyr]-1:
                            file_read_all = [(path_lst_geo + str(yearname[iyr]) + '/T' + tiles_load[ite] + '/' + 'viirs_lst_'
                                             + str(yearname[iyr]) + str(idt + 1 + x).zfill(3)) + overpass_name[y] + '.hdf5'
                                             for x in range(2) for y in range(2)]
                        else:
                            file_read_all = [(path_lst_geo + str(yearname[iyr] + z) + '/T' + tiles_load[ite] + '/' + 'viirs_lst_'
                                             + str(yearname[iyr] + z) + str(x).zfill(3)) + overpass_name[y] + '.hdf5'
                                             for z in range(2) for x in [1, idt+1] for y in range(2)]
                            file_read_all = file_read_all[2:6]

                        viirs_lst_1day = []
                        for itm in range(len(file_read_all)):
                            if os.path.exists(file_read_all[itm]) == True:
                                f_viirs = h5py.File(file_read_all[itm], 'r')
                                varname_list_viirs = list(f_viirs.keys())
                                viirs_lst = f_viirs[varname_list_viirs[0]][()]
                                viirs_lst = np.flipud(viirs_lst)
                            else:
                                viirs_lst = viirs_mat_fill
                            viirs_lst_1day.append(viirs_lst)
                            del(viirs_lst)
                            # print(file_read)
                        viirs_lst_delta_am = viirs_lst_1day[1] - viirs_lst_1day[0]
                        viirs_lst_delta_pm = viirs_lst_1day[1] - viirs_lst_1day[2]

                        viirs_lst_delta_tiles_am.append(viirs_lst_delta_am)
                        viirs_lst_delta_tiles_pm.append(viirs_lst_delta_pm)
                        del(viirs_lst_delta_am, viirs_lst_delta_pm, viirs_lst_1day)

                    viirs_lst_delta_tiles_am = np.vstack(viirs_lst_delta_tiles_am)
                    viirs_lst_delta_tiles_pm = np.vstack(viirs_lst_delta_tiles_pm)
                    viirs_lst_delta_tiles = list((viirs_lst_delta_tiles_am, viirs_lst_delta_tiles_pm))
                    # if len(tiles_load) == 2:
                    #     viirs_lst_delta_tiles_am = np.vstack(viirs_lst_delta_tiles_am)
                    # else:
                    #     pass

                    # Regrid the viirs data in EASE grid projection
                    viirs_mat_ease_all = []
                    for ife in range(2):
                        viirs_mat_ease = np.array \
                            ([np.nanmean(viirs_lst_delta_tiles[ife][row_world_ease_400m_from_geo_400m_ind_torep_local[x], :], axis=0)
                                for x in range(len(row_world_ease_400m_from_geo_400m_ind_torep_local))])
                        viirs_mat_ease = np.array \
                            ([np.nanmean(viirs_mat_ease[:, col_world_ease_400m_from_geo_400m_ind_torep_local[y]], axis=1)
                                for y in range(len(col_world_ease_400m_from_geo_400m_ind_torep_local))])
                        viirs_mat_ease = np.fliplr(np.rot90(viirs_mat_ease, 3))
                        viirs_mat_ease_all.append(viirs_mat_ease)
                        del(viirs_mat_ease)


                    viirs_tile_output_ease_path = path_lst_ease + str(yearname[iyr]) + '/T' + str(tile_num).zfill(3) + '/'
                    viirs_file_output_ease_path = ('viirs_lst_delta_' + str(yearname[iyr]) + str(idt + 1).zfill(3) + '_T'
                                                   + str(tile_num).zfill(3) + '.hdf5')

                    if os.path.exists(viirs_tile_output_ease_path) == False:
                        os.makedirs(viirs_tile_output_ease_path)
                    else:
                        pass

                    with h5py.File(viirs_tile_output_ease_path + viirs_file_output_ease_path, 'w') as f:
                        f.create_dataset('viirs_lst_delta_am', data=viirs_mat_ease_all[0], compression='lzf')
                        f.create_dataset('viirs_lst_delta_pm', data=viirs_mat_ease_all[1], compression='lzf')
                    f.close()

                    del(viirs_mat_ease_all, viirs_lst_delta_tiles)
                    print(viirs_file_output_ease_path)

            else:
                pass


########################################################################################################################
# 2.1 Process viirs LAI data

for iyr in [12]:#range(len(yearname)):
    file_name = sorted(glob.glob(path_viirs_lai + str(yearname[iyr]-1) + '/*'))
    file_name_doy = np.array([int(file_name[x].split('_')[-2]) for x in range(len(file_name))])
    file_name_doy_unique = np.unique(file_name_doy)
    file_divide_ind = np.arange(0, 8*len(file_name_doy_unique), 8)
    file_name_split = np.hsplit(np.arange(len(file_name_doy)), file_divide_ind)[1:]

    for idt in range(len(file_name_split)):
        file_quad_all = []
        for ife in range(len(file_name_split[idt])):
            file_quad = rasterio.open(file_name[file_name_split[idt][ife]]).read().squeeze()
            file_quad = np.array(file_quad, dtype='float32')
            file_quad[np.where(file_quad <= 0)] = np.nan
            file_quad[np.where(file_quad >= 100)] = np.nan
            file_quad = file_quad * 0.1
            file_quad_all.append(file_quad)
            del(file_quad)

        # Combine all blocks to form a global map
        row1 = np.hstack((file_quad_all[0], file_quad_all[1], file_quad_all[2], file_quad_all[3]))
        row2 = np.hstack((file_quad_all[4], file_quad_all[5], file_quad_all[6], file_quad_all[7]))
        viirs_lai_1day = np.vstack((row1, row2))
        viirs_lai_1day = viirs_lai_1day[3750: ,]
        # viirs_lai_1day = np.reshape(viirs_lai_1day, (3750, 810000))

        mat_divide_vind = np.arange(0, 9 * 3750, 3750)
        mat_divide_hind = np.arange(0, 24 * 3750, 3750)
        viirs_lai_1day_split_vert = np.vsplit(viirs_lai_1day, mat_divide_vind)[1:]
        viirs_lai_1day_split = [np.hsplit(viirs_lai_1day_split_vert[x], mat_divide_hind)[1:]
                                for x in range(len(viirs_lai_1day_split_vert))]
        viirs_lai_1day_split = list(itertools.chain(*viirs_lai_1day_split))

        for ite in range(len(viirs_lai_1day_split)):
            viirs_lai_tile_output_ease_path = (path_lai_geo + str(yearname[iyr]) + '/T' + str(ite+1).zfill(3)) + '/'
            viirs_lai_file_output_ease_path = ('viirs_lai_' + str(yearname[iyr]) + str(file_name_doy_unique[idt]).zfill(3)
                                               + '_T' + str(ite+1).zfill(3) + '.hdf5')

            if os.path.exists(viirs_lai_tile_output_ease_path) == False:
                os.makedirs(viirs_lai_tile_output_ease_path)
            else:
                pass

            with h5py.File(viirs_lai_tile_output_ease_path + viirs_lai_file_output_ease_path, 'w') as f:
                f.create_dataset('viirs_lai', data=viirs_lai_1day_split[ite], compression='lzf')
            f.close()
            print(viirs_lai_file_output_ease_path)

        del(viirs_lai_1day_split, viirs_lai_1day_split_vert, row1, row2, viirs_lai_1day, mat_divide_vind, mat_divide_hind)


# 2.2 Reproject viirs LAI data from geographic to EASR-grid projection

for irow in range(9):
    for icol in range(24):
        tile_num = irow * 24 + icol + 1
        # Extract row/col tables from grouped files for resampling
        row_world_ease_400m_from_geo_400m_ind_torep = row_world_ease_400m_from_geo_400m_ind_group[irow]
        col_world_ease_400m_from_geo_400m_ind_torep = col_world_ease_400m_from_geo_400m_ind_group[icol]

        row_world_ease_400m_from_geo_400m_ind_torep_num = np.array(list(itertools.chain(*row_world_ease_400m_from_geo_400m_ind_torep)))
        row_world_ease_400m_from_geo_400m_ind_torep_tile = row_world_geo_400m_ind_tile[row_world_ease_400m_from_geo_400m_ind_torep_num - row_ind_min]
        row_world_ease_400m_from_geo_400m_ind_torep_tile_unique = np.unique(row_world_ease_400m_from_geo_400m_ind_torep_tile)
        col_world_ease_400m_from_geo_400m_ind_torep_num = np.array(list(itertools.chain(*col_world_ease_400m_from_geo_400m_ind_torep)))
        col_world_ease_400m_from_geo_400m_ind_torep_tile = col_world_geo_400m_ind_tile[col_world_ease_400m_from_geo_400m_ind_torep_num]
        col_world_ease_400m_from_geo_400m_ind_torep_tile_unique = np.unique(col_world_ease_400m_from_geo_400m_ind_torep_tile)

        # Convert from world row/col index tables to local (indices start from upper left tile)
        row_world_ease_400m_from_geo_400m_ind_torep_local = (row_world_ease_400m_from_geo_400m_ind_torep - row_ind_min -
                                                             row_world_ease_400m_from_geo_400m_ind_torep_tile_unique[0] * 3750)
        col_world_ease_400m_from_geo_400m_ind_torep_local = (col_world_ease_400m_from_geo_400m_ind_torep -
                                                             col_world_ease_400m_from_geo_400m_ind_torep_tile_unique[0] * 3750)


        # Load all adjacent tiles of data for reprojecting
        tiles_load = (row_world_ease_400m_from_geo_400m_ind_torep_tile_unique[:, None] * 24 +
                      col_world_ease_400m_from_geo_400m_ind_torep_tile_unique + 1).ravel()
        tiles_load = [str(tiles_load[x]).zfill(3) for x in range(len(tiles_load))]

        for iyr in [12]:#range(len(yearname)):
            for idt in range(len(viirs_lai_days)):

                viirs_lai_tiles = []
                for ite in range(len(tiles_load)):
                    file_read = (path_lai_geo + str(yearname[iyr]) + '/T' + tiles_load[ite] + '/' + 'viirs_lai_'
                                 + str(yearname[iyr]) + str(viirs_lai_days[idt]).zfill(3) + '_T' + tiles_load[ite]
                                 + '.hdf5')
                    f_viirs = h5py.File(file_read, 'r')
                    varname_list_viirs = list(f_viirs.keys())
                    viirs_lai = f_viirs[varname_list_viirs[0]][()]
                    viirs_lai_tiles.append(viirs_lai)
                    del(viirs_lai)
                    print(file_read)

                    viirs_lai_tiles = np.vstack(viirs_lai_tiles)
                # if len(tiles_load) == 2:
                #     viirs_lai_tiles = np.vstack(viirs_lai_tiles)
                # else:
                #     pass

                # Regrid the viirs data in EASE grid projection
                viirs_mat_ease = np.array \
                    ([np.nanmean(viirs_lai_tiles[row_world_ease_400m_from_geo_400m_ind_torep_local[x], :], axis=0)
                      for x in range(len(row_world_ease_400m_from_geo_400m_ind_torep_local))])
                viirs_mat_ease = np.array \
                    ([np.nanmean(viirs_mat_ease[:, col_world_ease_400m_from_geo_400m_ind_torep_local[y]], axis=1)
                      for y in range(len(col_world_ease_400m_from_geo_400m_ind_torep_local))])
                viirs_mat_ease = np.fliplr(np.rot90(viirs_mat_ease, 3))

                viirs_tile_output_ease_path = path_lai_ease + str(yearname[iyr]) + '/T' + str(tile_num).zfill(3) + '/'
                viirs_file_output_ease_path = ('viirs_lai_' + str(yearname[iyr]) + str(viirs_lai_days[idt]).zfill(3) + '_T'
                                               + str(tile_num).zfill(3) + '.hdf5')

                if os.path.exists(viirs_tile_output_ease_path) == False:
                    os.makedirs(viirs_tile_output_ease_path)
                else:
                    pass

                with h5py.File(viirs_tile_output_ease_path + viirs_file_output_ease_path, 'w') as f:
                    f.create_dataset('viirs_lai_ease', data=viirs_mat_ease, compression='lzf')
                f.close()

                del(viirs_mat_ease, viirs_lai_tiles)
                print(viirs_file_output_ease_path)


########################################################################################################################
# 3. Implement the VIS/IR downscaling model on VIIRS LST difference to calculate 400 m soil moisture

# Load downscaling model coefficients by month for am/pm overpasses
os.chdir(path_model)
f = h5py.File("ds_model_coef.hdf5", "r")
varname_list = list(f.keys())

varname_list_mat_am = varname_list[0:12]
coef_mat_am = []
for x in range(len(varname_list_mat_am)):
    mat_am_read = f[varname_list_mat_am[x]][()]
    coef_mat_am.append(mat_am_read)
    del(mat_am_read)

varname_list_mat_pm = varname_list[12:24]
coef_mat_pm = []
for x in range(len(varname_list_mat_pm)):
    mat_pm_read = f[varname_list_mat_pm[x]][()]
    coef_mat_pm.append(mat_pm_read)
    del(mat_pm_read)

f.close()

# Divide row/col indices by image block indices
row_divide_ind_ease = np.arange(0, len(row_world_ease_400m_from_25km_ind), 4060)
row_world_ease_400m_from_25km_ind_group = np.split(row_world_ease_400m_from_25km_ind, row_divide_ind_ease, axis=0)[1:]
col_divide_ind_ease = np.arange(0, len(col_world_ease_400m_from_25km_ind), 3615)
col_world_ease_400m_from_25km_ind_group = np.split(col_world_ease_400m_from_25km_ind, col_divide_ind_ease, axis=0)[1:]

for iyr in [12]:#range(len(yearname)):

    tile_name = sorted(glob.glob(path_lai_ease + str(yearname[iyr]) + '/*/', recursive=True))
    tile_name_base = [int(tile_name[x].split('/')[-2][1:]) for x in range(len(tile_name))]
    doy_month_ind = np.concatenate([np.repeat(x+1, daysofmonth_seq[x, iyr]) for x in range(0, iyr)])

    for irow in [0, 1, 2]:#range(9):
        for icol in [12, 13, 14]:#range(24):

            tile_num = irow * 24 + icol + 1
            if tile_num in tile_name_base:
                # Subset indices from the corresponding tile
                row_world_ease_400m_from_25km_ind_sub = row_world_ease_400m_from_25km_ind_group[irow]
                col_world_ease_400m_from_25km_ind_sub = col_world_ease_400m_from_25km_ind_group[icol]
                col_meshgrid_from_25km, row_meshgrid_from_25km = np.meshgrid(col_world_ease_400m_from_25km_ind_sub,
                                                                           row_world_ease_400m_from_25km_ind_sub)
                col_meshgrid_from_25km = col_meshgrid_from_25km.reshape(1, -1)
                row_meshgrid_from_25km = row_meshgrid_from_25km.reshape(1, -1)
                coef_mat_am_sub_all = [coef_mat_am[x][row_meshgrid_from_25km, col_meshgrid_from_25km, :].squeeze()
                                            for x in range(len(coef_mat_am))]
                coef_mat_pm_sub_all = [coef_mat_pm[x][row_meshgrid_from_25km, col_meshgrid_from_25km, :].squeeze()
                                            for x in range(len(coef_mat_pm))]

                # Load viirs LST/LAI data for calculating 400 m sm
                viirs_lst_file_list = sorted(glob.glob(path_lst_ease + str(yearname[iyr]) + '/T' + str(tile_num).zfill(3) + '/*'))
                viirs_lai_file_list = sorted(glob.glob(path_lai_ease + str(yearname[iyr]) + '/T' + str(tile_num).zfill(3) + '/*'))
                for idt in range(len(viirs_lst_file_list)):
                    f_lst = h5py.File(viirs_lst_file_list[idt], 'r')
                    lst_varname_list = list(f_lst.keys())
                    viirs_lst_mat_am = f_lst[lst_varname_list[0]][()].ravel()
                    viirs_lst_mat_pm = f_lst[lst_varname_list[1]][()].ravel()
                    f_lst.close()

                    f_lai = h5py.File(viirs_lai_file_list[viirs_lai_days_ind[idt]], 'r')
                    lai_varname_list = list(f_lai.keys())
                    viirs_lai_mat = f_lai[lai_varname_list[0]][()].ravel()
                    viirs_lai_mat_scale = ((viirs_lai_mat - np.nanmin(viirs_lai_mat))/
                                           (np.nanmax(viirs_lai_mat) - np.nanmin(viirs_lai_mat)))
                    viirs_lai_mat_scale[np.isnan(viirs_lai_mat_scale)] = 0
                    viirs_lai_mat_scale = np.fix(viirs_lai_mat_scale*10).astype(int)
                    viirs_lai_mat_scale[viirs_lai_mat_scale == 10] = 9
                    f_lai.close()

                    # sm of am overpass
                    coef_mat_am_coef = coef_mat_am_sub_all[doy_month_ind[idt]-1][:, ::2]
                    coef_mat_am_intc = coef_mat_am_sub_all[doy_month_ind[idt]-1][:, 1::2]
                    coef_mat_am_coef_slc = coef_mat_am_coef[np.arange(len(viirs_lst_mat_am)), viirs_lai_mat_scale]
                    coef_mat_am_intc_slc = coef_mat_am_intc[np.arange(len(viirs_lst_mat_am)), viirs_lai_mat_scale]

                    model_sm_400m_am = coef_mat_am_coef_slc * viirs_lst_mat_am + coef_mat_am_intc_slc
                    model_sm_400m_am = np.reshape(model_sm_400m_am, (4060, 3615))

                    # sm of pm overpass
                    coef_mat_pm_coef = coef_mat_pm_sub_all[doy_month_ind[idt]-1][:, ::2]
                    coef_mat_pm_intc = coef_mat_pm_sub_all[doy_month_ind[idt]-1][:, 1::2]
                    coef_mat_pm_coef_slc = coef_mat_pm_coef[np.arange(len(viirs_lst_mat_pm)), viirs_lai_mat_scale]
                    coef_mat_pm_intc_slc = coef_mat_pm_intc[np.arange(len(viirs_lst_mat_pm)), viirs_lai_mat_scale]

                    model_sm_400m_pm = coef_mat_pm_coef_slc * viirs_lst_mat_pm + coef_mat_pm_intc_slc
                    model_sm_400m_pm = np.reshape(model_sm_400m_pm, (4060, 3615))

                    # Write to file
                    model_sm_tile_output_path = path_model_sm + str(yearname[iyr]) + '/T' + str(tile_num).zfill(3) + '/'
                    model_sm_file_output_path = ('model_sm_400m_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '_T'
                                                 + str(tile_num).zfill(3) + '.hdf5')

                    if os.path.exists(model_sm_tile_output_path) == False:
                        os.makedirs(model_sm_tile_output_path)
                    else:
                        pass

                    with h5py.File(model_sm_tile_output_path + model_sm_file_output_path, 'w') as f:
                        f.create_dataset('model_sm_am', data=model_sm_400m_am, compression='lzf')
                        f.create_dataset('model_sm_pm', data=model_sm_400m_pm, compression='lzf')
                    f.close()

                    print(model_sm_file_output_path)
                    del(viirs_lst_mat_am, viirs_lst_mat_pm, viirs_lai_mat, viirs_lai_mat_scale, coef_mat_am_coef, coef_mat_am_intc,
                        coef_mat_am_coef_slc, coef_mat_am_intc_slc, model_sm_400m_am, coef_mat_pm_coef, coef_mat_pm_intc,
                        coef_mat_pm_coef_slc, coef_mat_pm_intc_slc, model_sm_400m_pm, model_sm_tile_output_path, model_sm_file_output_path)

                del(row_world_ease_400m_from_25km_ind_sub, col_world_ease_400m_from_25km_ind_sub, row_meshgrid_from_25km,
                    col_meshgrid_from_25km, coef_mat_am_sub_all, coef_mat_pm_sub_all)

            else:
                pass

########################################################################################################################
# 4. Downscale the 400m soil moisture model output by 9 km SMAP L2 soil moisture data

# Divide row/col indices by image block indices (9 km)
row_divide_ind_ease = np.arange(0, len(row_world_ease_400m_from_9km_ind), 4060)
row_world_ease_400m_from_9km_ind_group = np.split(row_world_ease_400m_from_9km_ind, row_divide_ind_ease, axis=0)[1:]
row_world_ease_400m_ind_group = np.split(row_world_ease_400m_ind, row_divide_ind_ease, axis=0)[1:]
col_divide_ind_ease = np.arange(0, len(col_world_ease_400m_from_9km_ind), 3615)
col_world_ease_400m_from_9km_ind_group = np.split(col_world_ease_400m_from_9km_ind, col_divide_ind_ease, axis=0)[1:]
col_world_ease_400m_ind_group = np.split(col_world_ease_400m_ind, col_divide_ind_ease, axis=0)[1:]

# Col#1: 400m indices, Col#2: 9km indices
row_world_ease_400m_ind_tile = np.repeat(np.arange(9), 4060)
row_world_ease_9km_from_400m_ext33km_ind_tile = [np.repeat(x, len(row_world_ease_9km_from_400m_ext33km_ind[x]))
                                                 for x in range(len(row_world_ease_9km_from_400m_ext33km_ind))]
row_world_ease_9km_from_400m_ext33km_ind_tile = np.array(list(itertools.chain(*row_world_ease_9km_from_400m_ext33km_ind_tile)))
row_world_ease_9km_from_400m_ext33km_ind_flat = np.array(list(itertools.chain(*row_world_ease_9km_from_400m_ext33km_ind.tolist())))
row_world_ease_9km_from_400m_ext33km_ind_flat = np.stack((row_world_ease_9km_from_400m_ext33km_ind_flat,
                                                                row_world_ease_9km_from_400m_ext33km_ind_tile), axis=1)

col_world_ease_400m_ind_tile = np.repeat(np.arange(24), 3615)
col_world_ease_9km_from_400m_ext33km_ind_tile = [np.repeat(x, len(col_world_ease_9km_from_400m_ext33km_ind[x]))
                                                 for x in range(len(col_world_ease_9km_from_400m_ext33km_ind))]
col_world_ease_9km_from_400m_ext33km_ind_tile = np.array(list(itertools.chain(*col_world_ease_9km_from_400m_ext33km_ind_tile)))
col_world_ease_9km_from_400m_ext33km_ind_flat = np.array(list(itertools.chain(*col_world_ease_9km_from_400m_ext33km_ind.tolist())))
col_world_ease_9km_from_400m_ext33km_ind_flat = np.stack((col_world_ease_9km_from_400m_ext33km_ind_flat,
                                                                col_world_ease_9km_from_400m_ext33km_ind_tile), axis=1)

smap_sm_mat_fill = np.empty([4060, 3615], dtype='float32')
smap_sm_mat_fill[:] = np.nan

# Define georeference info for geotiff output
dst_srs = osr.SpatialReference()
dst_srs.ImportFromEPSG(6933)  # EASE grid projection
dst_wkt = dst_srs.ExportToWkt()
band_name = ['AM', 'PM']

for iyr in [12]:#range(len(yearname)):

    tile_name = sorted(glob.glob(path_model_sm + str(yearname[iyr]) + '/*/', recursive=True))
    tile_name_base = [int(tile_name[x].split('/')[-2][1:]) for x in range(len(tile_name))]
    doy_month_ind = np.concatenate([np.repeat(x+1, daysofmonth_seq[x, iyr]) for x in range(0, iyr)])

    for irow in range(9):
        for icol in range(24):

            tile_num = irow * 24 + icol + 1

            # Find start/end row/col indices
            row_start_9km = row_world_ease_9km_from_400m_ext33km_ind_flat[
                row_world_ease_9km_from_400m_ext33km_ind_flat[:, 0] == row_world_ease_400m_ind_group[irow][0]][0, 1]
            row_end_9km = row_world_ease_9km_from_400m_ext33km_ind_flat[
                row_world_ease_9km_from_400m_ext33km_ind_flat[:, 0] == row_world_ease_400m_ind_group[irow][-1]][-1, 1]
            row_agg_9km_start = row_world_ease_9km_from_400m_ext33km_ind[row_start_9km]
            row_agg_9km_end = row_world_ease_9km_from_400m_ext33km_ind[row_end_9km]
            row_agg_9km_start_ind = row_agg_9km_start[0]
            row_agg_9km_start_tile = row_world_ease_400m_ind_tile[row_agg_9km_start_ind]
            row_agg_9km_end_ind = row_agg_9km_end[-1]
            row_agg_9km_end_tile = row_world_ease_400m_ind_tile[row_agg_9km_end_ind]
            row_start_ind = row_world_ease_400m_ind_group[irow][0]
            row_end_ind = row_world_ease_400m_ind_group[irow][-1]
            row_first_tile_ind = row_world_ease_400m_ind_group[row_agg_9km_start_tile][0]
            row_diff_tofirst = row_agg_9km_start_ind - row_first_tile_ind
            row_agg_9km_local = row_world_ease_9km_from_400m_ext33km_ind[row_start_9km:row_end_9km+1] - row_agg_9km_start_ind
            row_end_tile_ind = row_world_ease_400m_ind_group[row_agg_9km_end_tile][-1]
            row_diff_toend = row_agg_9km_end_ind - row_first_tile_ind
            row_400m_sub = row_world_ease_400m_ind_group[irow][0] - row_agg_9km_start_ind

            col_start_9km = col_world_ease_9km_from_400m_ext33km_ind_flat[
                col_world_ease_9km_from_400m_ext33km_ind_flat[:, 0] == col_world_ease_400m_ind_group[icol][0]][0, 1]
            col_end_9km = col_world_ease_9km_from_400m_ext33km_ind_flat[
                col_world_ease_9km_from_400m_ext33km_ind_flat[:, 0] == col_world_ease_400m_ind_group[icol][-1]][-1, 1]
            col_agg_9km_start = col_world_ease_9km_from_400m_ext33km_ind[col_start_9km]
            col_agg_9km_end = col_world_ease_9km_from_400m_ext33km_ind[col_end_9km]
            col_agg_9km_start_ind = col_agg_9km_start[0]
            col_agg_9km_start_tile = col_world_ease_400m_ind_tile[col_agg_9km_start_ind]
            col_agg_9km_end_ind = col_agg_9km_end[-1]
            col_agg_9km_end_tile = col_world_ease_400m_ind_tile[col_agg_9km_end_ind]
            col_start_ind = col_world_ease_400m_ind_group[icol][0]
            col_end_ind = col_world_ease_400m_ind_group[icol][-1]
            col_first_tile_ind = col_world_ease_400m_ind_group[col_agg_9km_start_tile][0]
            col_diff_tofirst = col_agg_9km_start_ind - col_first_tile_ind
            col_agg_9km_local = col_world_ease_9km_from_400m_ext33km_ind[col_start_9km:col_end_9km+1] - col_agg_9km_start_ind
            col_end_tile_ind = col_world_ease_400m_ind_group[col_agg_9km_end_tile][-1]
            col_diff_toend = col_agg_9km_end_ind - col_first_tile_ind
            col_400m_sub = col_world_ease_400m_ind_group[icol][0] - col_agg_9km_start_ind

            # Disaggregation row/col indices for 400 m from 9 km
            row_world_ease_400m_from_9km_ind_sub = row_world_ease_400m_from_9km_ind_group[irow]
            col_world_ease_400m_from_9km_ind_sub = col_world_ease_400m_from_9km_ind_group[icol]
            col_meshgrid_from_9km, row_meshgrid_from_9km = np.meshgrid(col_world_ease_400m_from_9km_ind_sub,
                                                                         row_world_ease_400m_from_9km_ind_sub)
            col_meshgrid_from_9km = col_meshgrid_from_9km.reshape(1, -1).ravel()
            row_meshgrid_from_9km = row_meshgrid_from_9km.reshape(1, -1).ravel()
            col_meshgrid_from_9km = col_meshgrid_from_9km - col_start_9km
            row_meshgrid_from_9km = row_meshgrid_from_9km - row_start_9km


            # Load in SMAP 9 km and modeled 400 m SM data for downscaling
            for idt in range(daysofyear[iyr]):
                date_read = date_seq_ymd_group[iyr][idt]
                # SMAP 9km SM
                smap_sm_file = path_smap + '/9km/smap_sm_9km_' + str(yearname[iyr]) + date_read[4:6] + '.hdf5'
                f_smap = h5py.File(smap_sm_file, "r")
                varname_list = list(f_smap.keys())
                smap_sm_9km_am = f_smap[varname_list[0]][row_start_9km:row_end_9km+1, col_start_9km:col_end_9km+1,
                              int(date_read[6:])-1]
                smap_sm_9km_pm = f_smap[varname_list[1]][row_start_9km:row_end_9km+1, col_start_9km:col_end_9km+1,
                              int(date_read[6:])-1]
                f_smap.close()

                # Modeled 400m SM
                tiles_load = [x * 24 + y + 1 for x in range(row_agg_9km_start_tile, row_agg_9km_end_tile+1)
                              for y in range(col_agg_9km_start_tile, col_agg_9km_end_tile+1)]
                tile_row_size = row_agg_9km_end_tile+1 - row_agg_9km_start_tile
                tile_col_size = col_agg_9km_end_tile+1 - col_agg_9km_start_tile

                model_sm_400m_am_all = []
                model_sm_400m_pm_all = []
                for ife in range(len(tiles_load)):
                    model_sm_file = (path_model_sm + str(yearname[iyr]) + '/T' + str(tiles_load[ife]).zfill(3) + '/'
                                          + 'model_sm_400m_' + str(yearname[iyr]) + str(idt + 1).zfill(3) + '_T'
                                          + str(tiles_load[ife]).zfill(3) + '.hdf5')

                    if os.path.exists(model_sm_file) == True:
                        f_model = h5py.File(model_sm_file, "r")
                        varname_list_model = list(f_model.keys())
                        model_sm_400m_am = f_model[varname_list_model[0]][()]
                        model_sm_400m_pm = f_model[varname_list_model[1]][()]
                        f_model.close()
                    else:
                        model_sm_400m_am = smap_sm_mat_fill
                        model_sm_400m_pm = smap_sm_mat_fill
                    model_sm_400m_am_all.append(model_sm_400m_am)
                    model_sm_400m_pm_all.append(model_sm_400m_pm)
                    del(model_sm_400m_am, model_sm_400m_pm)
                    # print(model_sm_file)

                # Combine all blocks to form 3*3 big matrix
                model_sm_400m_am_all_div = [model_sm_400m_am_all[i:i + tile_col_size]
                                            for i in range(0, len(model_sm_400m_am_all), tile_col_size)]
                row_all_am = [np.hstack(model_sm_400m_am_all_div[x]) for x in range(len(model_sm_400m_am_all_div))]
                model_sm_400m_am_tods = np.vstack(row_all_am)
                model_sm_400m_am_sub = model_sm_400m_am_tods[row_diff_tofirst:row_diff_toend+1,
                                        col_diff_tofirst:col_diff_toend+1]

                model_sm_400m_pm_all_div = [model_sm_400m_pm_all[i:i + tile_col_size]
                                            for i in range(0, len(model_sm_400m_pm_all), tile_col_size)]
                row_all_pm = [np.hstack(model_sm_400m_pm_all_div[x]) for x in range(len(model_sm_400m_pm_all_div))]
                model_sm_400m_pm_tods = np.vstack(row_all_pm)
                model_sm_400m_pm_sub = model_sm_400m_pm_tods[row_diff_tofirst:row_diff_toend + 1,
                                        col_diff_tofirst:col_diff_toend + 1]

                del(model_sm_400m_am_all, model_sm_400m_pm_all, model_sm_400m_am_all_div, model_sm_400m_pm_all_div,
                    row_all_am, row_all_pm, model_sm_400m_am_tods, model_sm_400m_pm_tods)

                # aggregate to 9km by row/col indices
                # AM
                model_sm_400m_am_agg = np.array \
                    ([np.nanmean(model_sm_400m_am_sub[row_agg_9km_local[x], :], axis=0)
                      for x in range(len(row_agg_9km_local))])
                model_sm_400m_am_agg = np.array \
                    ([np.nanmean(model_sm_400m_am_agg[:, col_agg_9km_local[y]], axis=1)
                      for y in range(len(col_agg_9km_local))])
                model_sm_400m_am_agg = np.fliplr(np.rot90(model_sm_400m_am_agg, 3))
                model_sm_400m_am_delta = smap_sm_9km_am - model_sm_400m_am_agg
                model_sm_400m_am_delta_disagg = np.array(
                    [model_sm_400m_am_delta[row_meshgrid_from_9km[x], col_meshgrid_from_9km[x]]
                     for x in range(len(row_meshgrid_from_9km))])
                model_sm_400m_am_delta_disagg = model_sm_400m_am_delta_disagg.reshape(4060, 3615)
                model_sm_400m_am_sub_output = model_sm_400m_am_sub[row_400m_sub:row_400m_sub+4060,
                                              col_400m_sub:col_400m_sub+3615]
                smap_sm_400m_ds_am = model_sm_400m_am_sub_output + model_sm_400m_am_delta_disagg
                smap_sm_400m_ds_am[smap_sm_400m_ds_am < 0] = np.nan
                smap_sm_400m_ds_am[smap_sm_400m_ds_am > 1] = np.nan

                # PM
                model_sm_400m_pm_agg = np.array \
                    ([np.nanmean(model_sm_400m_pm_sub[row_agg_9km_local[x], :], axis=0)
                      for x in range(len(row_agg_9km_local))])
                model_sm_400m_pm_agg = np.array \
                    ([np.nanmean(model_sm_400m_pm_agg[:, col_agg_9km_local[y]], axis=1)
                      for y in range(len(col_agg_9km_local))])
                model_sm_400m_pm_agg = np.fliplr(np.rot90(model_sm_400m_pm_agg, 3))
                model_sm_400m_pm_delta = smap_sm_9km_pm - model_sm_400m_pm_agg
                model_sm_400m_pm_delta_disagg = np.array(
                    [model_sm_400m_pm_delta[row_meshgrid_from_9km[x], col_meshgrid_from_9km[x]]
                     for x in range(len(row_meshgrid_from_9km))])
                model_sm_400m_pm_delta_disagg = model_sm_400m_pm_delta_disagg.reshape(4060, 3615)
                model_sm_400m_pm_sub_output = model_sm_400m_pm_sub[row_400m_sub:row_400m_sub+4060,
                                              col_400m_sub:col_400m_sub+3615]
                smap_sm_400m_ds_pm = model_sm_400m_pm_sub_output + model_sm_400m_pm_delta_disagg
                smap_sm_400m_ds_pm[smap_sm_400m_ds_pm < 0] = np.nan
                smap_sm_400m_ds_pm[smap_sm_400m_ds_pm > 1] = np.nan

                smap_sm_400m_ds = list((smap_sm_400m_ds_am, smap_sm_400m_ds_pm))

                # Save the daily downscaled 400 m SM to Geotiff files
                # Create output path
                smap_sm_ds_tile_output_path = path_smap_sm_ds + str(yearname[iyr]) + '/T' + str(tile_num).zfill(3) + '/'
                smap_sm_ds_file_output_path = ('smap_sm_400m_' + str(yearname[iyr]) + str(idt + 1).zfill(3) + '_T'
                                             + str(tile_num).zfill(3) + '.tif')

                if os.path.exists(smap_sm_ds_tile_output_path) == False:
                    os.makedirs(smap_sm_ds_tile_output_path)
                else:
                    pass

                # Create a raster of EASE grid projection at 1 km resolution
                lat_output = lat_world_ease_400m_group[irow]
                lon_output = lon_world_ease_400m_group[icol]
                transformer = Transformer.from_crs("epsg:4326", "epsg:6933", always_xy=True)
                [lon_ul_ease, lat_ul_ease] = transformer.transform(lon_output[0], lat_output[0])
                gts = (lon_ul_ease - (interdist_ease_400m/2), interdist_ease_400m, 0, lat_ul_ease + (interdist_ease_400m/2),
                       0, -interdist_ease_400m)
                out_ds_tiff = gdal.GetDriverByName('GTiff').Create \
                    (smap_sm_ds_tile_output_path + smap_sm_ds_file_output_path,
                     len(lon_output), len(lat_output), 2,  # Number of bands
                     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
                out_ds_tiff.SetGeoTransform(gts)
                out_ds_tiff.SetProjection(dst_wkt)

                # Loop write each band to Geotiff file
                for idf in range(2):
                    out_ds_tiff.GetRasterBand(idf + 1).WriteArray(smap_sm_400m_ds[idf])
                    out_ds_tiff.GetRasterBand(idf + 1).SetNoDataValue(0)
                    out_ds_tiff.GetRasterBand(idf + 1).SetDescription(band_name[idf])
                out_ds_tiff = None  # close dataset to write to disc

                print(smap_sm_ds_file_output_path)
                del(smap_sm_400m_ds, smap_sm_400m_ds_am, smap_sm_400m_ds_pm, out_ds_tiff, gts)
                del(model_sm_400m_am_agg, model_sm_400m_am_delta, model_sm_400m_am_delta_disagg, model_sm_400m_am_sub_output,
                    model_sm_400m_am_sub, model_sm_400m_pm_agg, model_sm_400m_pm_delta, model_sm_400m_pm_delta_disagg,
                    model_sm_400m_pm_sub_output, model_sm_400m_pm_sub)






with tarfile.open('/Volumes/KINGSTON/FINAL_LST_T085.tar.gz', 'r') as tar:
    # Get a list of all members (files) in the .tar file
    file_list = tar.getnames()
    file_object = tar.extractfile(file_list[0])

    viirs_mat_all = []
    for ife in range(len(file_list)):
        with gzip.open(file_list[ife], 'rb') as file:
            viirs_mat = np.frombuffer(file.read(), dtype='f')
            viirs_mat = np.reshape(viirs_mat, (3750, 3750)).copy()
            viirs_mat[viirs_mat == -9999] = np.nan
            viirs_mat_all.append(viirs_mat)
            del (viirs_mat)
        file.close()




with tarfile.open('/Volumes/KINGSTON/FINAL_LST_T080.tar.gz', 'r') as tar:
    file_list = tar.getnames()
    viirs_mat_all = []
    for ife in range(100):#range(len(file_list)):
        gzip_file = tar.extractfile(file_list[ife])
        with gzip.open(gzip_file, 'rb') as file:
            viirs_mat = np.frombuffer(file.read(), dtype='f')
            viirs_mat = np.reshape(viirs_mat, (3750, 3750)).copy()
            viirs_mat[viirs_mat == -9999] = np.nan
            viirs_mat_all.append(viirs_mat)
            del(viirs_mat, gzip_file)
        file.close()
        print(file_list[ife])
tar.close()

