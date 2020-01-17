import os
import osr
import glob
import gdal
import numpy as np
import matplotlib.pyplot as plt
import datetime
import h5py
from netCDF4 import Dataset
import calendar
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of source MODIS data
path_modis = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/HDF'
# Path of output MODIS data
path_modis_op = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Output'
# Path of MODIS data for SM downscaling model input
path_modis_model = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Model_Input'
# Path of AMSR2 data
path_smap = '/Volumes/MyPassport/SMAP_Project/NewData/SMAP'
# Path of processed data
path_procdata = '/Users/binfang/Downloads/Processing/processed_data'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of GPM data
path_gpm = '/Users/binfang/Downloads/Processing/GPM/'
# Path of AMSR2 data
path_amsr2 = '/Volumes/MyPassport/SMAP_Project/Datasets/AMSR2'
# Path of output data
path_model_op = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Model_Output'
# Path of MODIS data for SM downscaling model input
path_model_ip = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Model_Input'
# Path of downscaled SM
path_amsr2_sm_ds = '/Users/binfang/Downloads/Processing/processed_data/AMSR2_ds'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
amsr2_sm_9km_name = ['amsr2_sm_9km_am', 'amsr2_sm_9km_pm']
subfolders = np.arange(2015, 2019+1, 1)
subfolders = [str(i).zfill(4) for i in subfolders]

# Load in variables
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_geo_1km', 'lon_world_geo_1km', 'lat_world_geo_10km', 'lon_world_geo_10km',
                'col_world_ease_1km_from_9km_ind', 'row_world_ease_1km_from_9km_ind',
                'row_world_ease_1km_from_geo_1km_ind', 'col_world_ease_1km_from_geo_1km_ind',
                'row_world_ease_9km_from_geo_10km_ind', 'col_world_ease_9km_from_geo_10km_ind',
                'row_world_ease_9km_from_1km_ext33km_ind', 'col_world_ease_9km_from_1km_ext33km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2019-10-31'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))

# Count how many days for a specific year
yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    if idt == 0:
        f_date = datetime.date(yearname[idt], monthnum[3], 1)
        l_date = datetime.date(yearname[idt], monthnum[-1], 31)
        delta_1y = l_date - f_date
        daysofyear.append(delta_1y.days + 1)
    else:
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

# Convert the 1 km from 9 km/25 km match table files to 1-d linear
col_meshgrid_from_9km, row_meshgrid_from_9km = np.meshgrid(col_world_ease_1km_from_9km_ind, row_world_ease_1km_from_9km_ind)
col_meshgrid_from_9km = col_meshgrid_from_9km.reshape(1, -1)
row_meshgrid_from_9km = row_meshgrid_from_9km.reshape(1, -1)

# col_meshgrid_from_25km, row_meshgrid_from_25km = np.meshgrid(col_world_ease_1km_from_25km_ind, row_world_ease_1km_from_25km_ind)
# col_meshgrid_from_25km = col_meshgrid_from_25km.reshape(1, -1)
# row_meshgrid_from_25km = row_meshgrid_from_25km.reshape(1, -1)


# # MODIS data layer information for extraction
# # n^th of Layers to be extracted. The information of number of layers can be acquired by function GetSubDatasets()
# subdataset_id_lst = [0, 1, 4, 5]  # For MODIS LST data: extract LST_Day_1km, QC_Day, LST_Night_1km, QC_Night
# band_n_lst = 2  # For MODIS LST data: save LST_Day_1km, LST_Night_1km
# subdataset_id_ndvi = [0, 2] # For MODIS NDVI data: extract NDVI, VI Quality
# band_n_ndvi = 1 # For MODIS NDVI data: save NDVI
#
# # Set the boundary coordinates of the map to subset (World)
# lat_roi_max = 90
# lat_roi_min = -90
# lon_roi_max = 180
# lon_roi_min = -180
#
# modis_folders = [lst_folder, ndvi_folder]
# subdataset_id = [subdataset_id_lst, subdataset_id_ndvi]
# band_n = [band_n_lst, band_n_ndvi]
# modis_var_names = ['modis_lst_1km', 'modis_ndvi_1km']
#
# # Define target SRS
# dst_srs = osr.SpatialReference()
# dst_srs.ImportFromEPSG(6933)  # EASE grid projection
# dst_wkt = dst_srs.ExportToWkt()
# gts = (-17367530.44516138, 1000.89502334956, 0, 7314540.79258289, 0, -1000.89502334956)


########################################################################################################################
# 1. Process AMSR2 LPRM SM data

# # Extract lat/lon information
# os.chdir(path_amsr2 + '/' + str(yearname[0]))
# amsr2_files_1day = sorted(glob.glob('*.nc4'))
#
# rootgrp = Dataset(amsr2_files_1day[0], mode='r')
# lat_world_geo_10km = rootgrp.variables['Latitude'][:]
# lat_world_geo_10km = np.squeeze((lat_world_geo_10km))
# lat_world_geo_10km = np.ma.getdata(lat_world_geo_10km).reshape(1, -1)
# lat_world_geo_10km = np.fliplr(lat_world_geo_10km).ravel()
#
# lon_world_geo_10km = rootgrp.variables['Longitude'][:]
# lon_world_geo_10km = np.squeeze((lon_world_geo_10km))
# lon_world_geo_10km = np.ma.getdata(lon_world_geo_10km)

# Process the data by each year

matsize_amsr2_1day = [len(lat_world_geo_10km), len(lon_world_geo_10km)]
amsr2_mat_init_1day = np.empty(matsize_amsr2_1day, dtype='float32')
amsr2_mat_init_1day[:] = np.nan

matsize_amsr2_ease_1day = [len(lat_world_ease_9km), len(lon_world_ease_9km)]
amsr2_mat_init_ease_1day = np.empty(matsize_amsr2_ease_1day, dtype='float32')
amsr2_mat_init_ease_1day[:] = np.nan

for iyr in [4]:#range(len(daysofyear)):

    os.chdir(path_amsr2 + '/' + str(yearname[iyr]))
    amsr2_files_year = sorted(glob.glob('*.nc4'))

    # Group AMSR2 data by month
    for imo in range(len(monthnum)):

        os.chdir(path_amsr2 + '/' + str(yearname[iyr]))
        amsr2_files_group_1month = [amsr2_files_year.index(i) for i in amsr2_files_year if str(yearname[iyr]) + monthname[imo] in i]

        # Process each month
        if len(amsr2_files_group_1month) != 0:
            amsr2_files_month = [amsr2_files_year[amsr2_files_group_1month[i]] for i in range(len(amsr2_files_group_1month))]

            # Create initial empty matrices for monthly AMSR2 final output data
            matsize_amsr2 = [matsize_amsr2_ease_1day[0], matsize_amsr2_ease_1day[1], daysofmonth_seq[imo, iyr]]
            amsr2_mat_month_am = np.empty(matsize_amsr2, dtype='float32')
            amsr2_mat_month_am[:] = np.nan
            amsr2_mat_month_pm = np.copy(amsr2_mat_month_am)

            # Extract AMSR2 data layers and rebind to daily
            for idt in range(daysofmonth_seq[imo, iyr]):
                amsr2_files_group_1day = [amsr2_files_month.index(i) for i in amsr2_files_month if
                                         str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                amsr2_files_1day = [amsr2_files_month[amsr2_files_group_1day[i]] for i in
                                    range(len(amsr2_files_group_1day))]
                amsr2_files_group_1day_am = [amsr2_files_1day.index(i) for i in amsr2_files_1day if
                                         '_D_SOILM3_V001_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                amsr2_files_group_1day_pm = [amsr2_files_1day.index(i) for i in amsr2_files_1day if
                                         '_A_SOILM3_V001_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                amsr2_mat_group_1day = \
                    np.empty([matsize_amsr2_1day[0], matsize_amsr2_1day[1], len(amsr2_files_group_1day)], dtype='float32')
                amsr2_mat_group_1day[:] = np.nan

                # Read swath files within a day and stack
                for ife in range(len(amsr2_files_1day)):
                    amsr2_mat_1file = np.copy(amsr2_mat_init_1day)
                    fe_amsr2= h5py.File(amsr2_files_1day[ife], "r")
                    varname_list_amsr2 = list(fe_amsr2.keys())
                    # Extract variables
                    sm_c1 = fe_amsr2[varname_list_amsr2[8]][()]
                    sm_c1 = sm_c1/100
                    sm_c1[np.where((sm_c1 <= 0) | (sm_c1 > 0.5))] = np.nan
                    sm_c2 = fe_amsr2[varname_list_amsr2[10]][()]
                    sm_c2 = sm_c2/100
                    sm_c2[np.where((sm_c2 <= 0) | (sm_c2 > 0.5))] = np.nan
                    sm = np.nanmean(np.stack((sm_c1, sm_c2), axis=2), axis=2)
                    sm = np.flipud(np.rot90(sm)).astype('float32')

                    amsr2_mat_group_1day[:, :, ife] = sm
                    print(amsr2_files_1day[ife])
                    fe_amsr2.close()

                    del(amsr2_mat_1file, fe_amsr2, varname_list_amsr2, sm, sm_c1, sm_c2)

                amsr2_mat_1day_am = np.nanmean(amsr2_mat_group_1day[:, :, amsr2_files_group_1day_am], axis=2)
                amsr2_mat_1day_pm = np.nanmean(amsr2_mat_group_1day[:, :, amsr2_files_group_1day_pm], axis=2)
                amsr2_mat_1day = np.stack((amsr2_mat_1day_am, amsr2_mat_1day_pm), axis=2)

                del(amsr2_mat_group_1day, amsr2_files_group_1day, amsr2_files_1day, amsr2_mat_1day_am, amsr2_mat_1day_pm)


                # Resample to 9 km EASE grid projection
                amsr2_mat_ease_1day_2tm = []
                for itm in range(2):
                    amsr2_mat_ease_1day = np.copy(amsr2_mat_init_ease_1day)
                    amsr2_mat_ease_1day = np.array\
                        ([np.nanmean(amsr2_mat_1day[row_world_ease_9km_from_geo_10km_ind[x], :, itm], axis=0)
                          for x in range(len(lat_world_ease_9km))])
                    amsr2_mat_ease_1day = np.array\
                        ([np.nanmean(amsr2_mat_ease_1day[:, col_world_ease_9km_from_geo_10km_ind[y]], axis=1)
                          for y in range(len(lon_world_ease_9km))])
                    amsr2_mat_ease_1day = np.fliplr(np.rot90(amsr2_mat_ease_1day, 3))
                    amsr2_mat_ease_1day_2tm.append(amsr2_mat_ease_1day)

                amsr2_mat_ease_1day_2tm = np.array(amsr2_mat_ease_1day_2tm)

                amsr2_mat_month_am[:, :, idt] = amsr2_mat_ease_1day_2tm[0, :, :]
                amsr2_mat_month_pm[:, :, idt] = amsr2_mat_ease_1day_2tm[1, :, :]
                del(amsr2_mat_ease_1day_2tm, amsr2_mat_1day)



            # Save file
            os.chdir(path_procdata)
            var_name = ['amsr2_sm_9km_am_' + str(yearname[iyr]) + monthname[imo],
                        'amsr2_sm_9km_pm_' + str(yearname[iyr]) + monthname[imo]]
            data_name = ['amsr2_mat_month_am', 'amsr2_mat_month_pm']

            with h5py.File('amsr2_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5', 'w') as f:
                for idv in range(len(var_name)):
                    f.create_dataset(var_name[idv], data=eval(data_name[idv]))
            f.close()
            del(amsr2_mat_month_am, amsr2_mat_month_pm)

        else:
            pass




########################################################################################################################
# 2. Downscale the 1km soil moisture model output by 9 km AMSR2 soil moisture data

# Create initial EASE grid projection matrices
amsr2_sm_1km_agg_init = np.empty([len(lat_world_ease_9km), len(lon_world_ease_9km)], dtype='float32')
amsr2_sm_1km_agg_init[:] = np.nan
amsr2_sm_1km_disagg_init = np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km)], dtype='float32')
amsr2_sm_1km_disagg_init = amsr2_sm_1km_disagg_init.reshape(1, -1)
amsr2_sm_1km_disagg_init[:] = np.nan
amsr2_sm_1km_ds_init = np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km), 2], dtype='float32')
amsr2_sm_1km_ds_init[:] = np.nan

for iyr in [4]:#range(len(yearname)):

    for imo in range(3, 9):#range(len(monthname)):

        # Load in amsr2 9km SM data
        amsr2_file_path = path_procdata + '/amsr2_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

        # Find the if the file exists in the directory
        if os.path.exists(amsr2_file_path) == True:
            f_amsr2_9km = h5py.File(amsr2_file_path, "r")
            varname_list_amsr2_ip = list(f_amsr2_9km.keys())
            for x in range(len(varname_list_amsr2_ip)):
                var_obj = f_amsr2_9km[varname_list_amsr2_ip[x]][()]
                exec(amsr2_sm_9km_name[x] + '= var_obj')
                del(var_obj)
            f_amsr2_9km.close()

            amsr2_sm_9km = np.concatenate((amsr2_sm_9km_am, amsr2_sm_9km_pm), axis=2)
            del(amsr2_sm_9km_am, amsr2_sm_9km_pm)

            # Load in MODIS LST data
            month_begin = daysofmonth_seq[0:imo, iyr].sum()
            month_end = daysofmonth_seq[0:imo + 1, iyr].sum()
            month_lenth = month_end - month_begin
            for idt in range(month_lenth):

                amsr2_sm_1km_file_path = path_model_op + '/' + str(yearname[iyr]) + '/smap_sm_1km_' + str(yearname[iyr]) + \
                                        str(month_begin+idt+1).zfill(3) + '.tif'

                if os.path.exists(amsr2_sm_1km_file_path) == True:
                    ds_amsr2_sm_1km = gdal.Open(amsr2_sm_1km_file_path)
                    amsr2_sm_1km = ds_amsr2_sm_1km.ReadAsArray()
                    amsr2_sm_1km = np.transpose(amsr2_sm_1km, (1, 2, 0))
                    amsr2_sm_1km_ds_output = np.copy(amsr2_sm_1km_ds_init)

                    for idf in range(2):
                        # Aggregate 1km SM model output to 9 km resolution, and calculate its difference with 9 km amsr2 SM
                        amsr2_sm_1km_agg = np.copy(amsr2_sm_1km_agg_init)

                        amsr2_sm_1km_1file = amsr2_sm_1km[:, :, idf]
                        amsr2_sm_1km_1file_1dim = amsr2_sm_1km_1file.reshape(1, -1)
                        amsr2_sm_1km_1file_ind = np.where(~np.isnan(amsr2_sm_1km_1file_1dim))[1]

                        amsr2_sm_1km_agg = np.array \
                            ([np.nanmean(amsr2_sm_1km_1file[row_world_ease_9km_from_1km_ext33km_ind[x], :], axis=0)
                              for x in range(len(lat_world_ease_9km))])
                        amsr2_sm_1km_agg = np.array \
                            ([np.nanmean(amsr2_sm_1km_agg[:, col_world_ease_9km_from_1km_ext33km_ind[y]], axis=1)
                              for y in range(len(lon_world_ease_9km))])
                        amsr2_sm_1km_agg = np.fliplr(np.rot90(amsr2_sm_1km_agg, 3))
                        amsr2_sm_1km_delta = amsr2_sm_9km[:, :, month_lenth*idf+idt] - amsr2_sm_1km_agg
                        # amsr2_sm_1km_delta = amsr2_sm_1km_delta.reshape(1, -1)

                        amsr2_sm_1km_delta_disagg = np.array([amsr2_sm_1km_delta[row_meshgrid_from_9km[0, amsr2_sm_1km_1file_ind[x]],
                                                  col_meshgrid_from_9km[0, amsr2_sm_1km_1file_ind[x]]]
                                      for x in range(len(amsr2_sm_1km_1file_ind))])
                        amsr2_sm_1km_disagg = np.copy(amsr2_sm_1km_disagg_init)
                        amsr2_sm_1km_disagg[0, amsr2_sm_1km_1file_ind] = amsr2_sm_1km_delta_disagg
                        amsr2_sm_1km_disagg = amsr2_sm_1km_disagg.reshape(len(lat_world_ease_1km), len(lon_world_ease_1km))

                        amsr2_sm_1km_ds = amsr2_sm_1km_1file + amsr2_sm_1km_disagg
                        amsr2_sm_1km_ds[np.where(amsr2_sm_1km_ds <= 0)] = np.nan
                        amsr2_sm_1km_ds_output[:, :, idf] = amsr2_sm_1km_ds
                        del(amsr2_sm_1km_agg, amsr2_sm_1km_1file, amsr2_sm_1km_1file_1dim, amsr2_sm_1km_1file_ind, amsr2_sm_1km_delta,
                            amsr2_sm_1km_delta_disagg, amsr2_sm_1km_disagg, amsr2_sm_1km_ds)

                    # Save the daily 1 km SM model output to Geotiff files
                    # Build output path
                    os.chdir(path_amsr2_sm_ds + '/' + str(yearname[iyr]))

                    # Create a raster of EASE grid projection at 1 km resolution
                    out_ds_tiff = gdal.GetDriverByName('GTiff').Create\
                        ('amsr2_sm_1km_ds_' + str(yearname[iyr]) + str(month_begin+idt+1).zfill(3) + '.tif',
                         len(lon_world_ease_1km), len(lat_world_ease_1km), 2, # Number of bands
                         gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
                    out_ds_tiff.SetGeoTransform(ds_amsr2_sm_1km.GetGeoTransform())
                    out_ds_tiff.SetProjection(ds_amsr2_sm_1km.GetProjection())

                    # Loop write each band to Geotiff file
                    for idf in range(2):
                        out_ds_tiff.GetRasterBand(idf + 1).WriteArray(amsr2_sm_1km_ds_output[:, :, idf])
                        out_ds_tiff.GetRasterBand(idf + 1).SetNoDataValue(0)
                    out_ds_tiff = None  # close dataset to write to disc

                    print(str(yearname[iyr]) + str(month_begin+idt+1).zfill(3))
                    del (amsr2_sm_1km_ds_output, ds_amsr2_sm_1km, out_ds_tiff)

                else:
                    pass

        else:
            pass


