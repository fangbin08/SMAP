import os
import osr
import glob
import gdal
import numpy as np
import matplotlib.pyplot as plt
import datetime
import h5py
import calendar
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of processed data
path_procdata = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of source MODIS data
path_modis = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/HDF'
# Path of source output MODIS data
path_modis_op = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Output'
# Path of MODIS data for SM downscaling model input
path_modis_model_ip = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Model_Input'
# Path of SM model output
path_model_op = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP_ds/Model_Output'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP_ds/Downscale'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_ease_25km', 'lon_world_ease_25km', 'row_world_ease_1km_from_25km_ind',
                'col_world_ease_1km_from_25km_ind', 'row_world_ease_1km_from_9km_ind', 'col_world_ease_1km_from_9km_ind',
                'row_world_ease_9km_from_1km_ext33km_ind', 'col_world_ease_9km_from_1km_ext33km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
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

modis_ndvi_days = np.array([9,  25,  41,  57,  73,  89, 105, 121, 137, 153, 169, 185, 201,
       217, 233, 249, 265, 281, 297, 313, 329, 345, 361])
oneyeardays_sq = np.arange(1, 367)
modis_ndvi_days_ind = np.array([modis_ndvi_days[np.where(np.absolute(oneyeardays_sq[x] - modis_ndvi_days) ==
                                                np.amin((np.absolute(oneyeardays_sq[x] - modis_ndvi_days))))].item(0)
                       for x in range(len(oneyeardays_sq))])

# Convert the 1 km from 9 km/25 km match table files to 1-d linear
col_meshgrid_from_9km, row_meshgrid_from_9km = np.meshgrid(col_world_ease_1km_from_9km_ind, row_world_ease_1km_from_9km_ind)
col_meshgrid_from_9km = col_meshgrid_from_9km.reshape(1, -1)
row_meshgrid_from_9km = row_meshgrid_from_9km.reshape(1, -1)

col_meshgrid_from_25km, row_meshgrid_from_25km = np.meshgrid(col_world_ease_1km_from_25km_ind, row_world_ease_1km_from_25km_ind)
col_meshgrid_from_25km = col_meshgrid_from_25km.reshape(1, -1)
row_meshgrid_from_25km = row_meshgrid_from_25km.reshape(1, -1)


########################################################################################################################
# 1. Implement the VIS/IR downscaling model on MODIS LST difference to calculate 1 km soil moisture

# Load in model coefficient files
os.chdir(path_procdata)
f = h5py.File("ds_model_coef.hdf5", "r")
varname_list = list(f.keys())
varname_list = varname_list[0:24] # Load in only monthly linear regression model coefficients
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


matsize_smap_sm_model_1day = [len(lat_world_ease_1km), len(lon_world_ease_1km)]
smap_sm_model_mat_init_1day = np.empty(matsize_smap_sm_model_1day, dtype='float32')
smap_sm_model_mat_init_1day[:] = np.nan
smap_sm_model_mat_init_1day_1dim = smap_sm_model_mat_init_1day.reshape(1, -1)

for iyr in range(len(yearname)):

    for idt in range(daysofyear[iyr]):

        # Load in MODIS LST data
        modis_lst_file_path_1 = path_modis_model_ip + lst_folder + str(yearname[iyr]) + '/' + \
                                'modis_lst_1km_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '.tif'
        if idt < daysofyear[iyr]:
            modis_lst_file_path_2 = path_modis_model_ip + lst_folder + str(yearname[iyr]) + '/' + \
                                    'modis_lst_1km_' + str(yearname[iyr]) + str(idt+2).zfill(3) + '.tif'
        elif iyr < len(yearname)-1:
            modis_lst_file_path_2 = path_modis_model_ip + lst_folder + str(yearname[iyr+1]) + '/' + \
                                    'modis_lst_1km_' + str(yearname[iyr+1]) + str(1).zfill(3) + '.tif'
        else:
            modis_lst_file_path_2 = 'None'

        # Find the if the files exist in the directory
        if os.path.exists(modis_lst_file_path_1) == True and os.path.exists(modis_lst_file_path_2) == True:

            ds_lst_1 = gdal.Open(modis_lst_file_path_1)
            ds_lst_2 = gdal.Open(modis_lst_file_path_2)
            ds_lst_1_day = ds_lst_1.GetRasterBand(1).ReadAsArray()
            ds_lst_1_night = ds_lst_1.GetRasterBand(2).ReadAsArray()
            ds_lst_2_night = ds_lst_2.GetRasterBand(2).ReadAsArray()

            ds_lst_am = ds_lst_1_day - ds_lst_1_night
            ds_lst_pm = ds_lst_1_day - ds_lst_2_night

            ds_lst_am = ds_lst_am.reshape(1, -1)
            ds_lst_pm = ds_lst_pm.reshape(1, -1)
            ds_lst_am_ind = np.where(~np.isnan(ds_lst_am))[1]
            ds_lst_pm_ind = np.where(~np.isnan(ds_lst_pm))[1]
            ds_lst_am_nonnan = ds_lst_am[0, ds_lst_am_ind]
            ds_lst_pm_nonnan = ds_lst_pm[0, ds_lst_pm_ind]

            # Load in MODIS NDVI data
            modis_ndvi_file_path = path_modis_model_ip + ndvi_folder + str(yearname[iyr]) + '/' + \
                                   'modis_ndvi_1km_' + str(yearname[iyr]) + str(modis_ndvi_days_ind[idt]).zfill(3) + '.tif'

            ds_ndvi = gdal.Open(modis_ndvi_file_path)
            ds_ndvi_idx = ds_ndvi.GetRasterBand(1).ReadAsArray()
            ds_ndvi_idx = np.nan_to_num(ds_ndvi_idx)
            ds_ndvi_idx = np.fix(ds_ndvi_idx*10).astype(int)
            ds_ndvi_idx[np.where(ds_ndvi_idx >= 10)] = 9

            del(ds_lst_2, ds_lst_1_day, ds_lst_1_night, ds_lst_2_night, ds_ndvi)


            # Extract coefficient and intercept from the position indices of corresponding monthly model file
            ds_ndvi_idx = ds_ndvi_idx.reshape(1, -1)

            month_id = str(datetime.datetime.strptime(str(yearname[iyr]) + '+' + str(idt+1).zfill(3), '%Y+%j').month).zfill(2)
            exec('coef_mat_am = ' + 'coef_mat_am_' + month_id)
            exec('coef_mat_pm = ' + 'coef_mat_pm_' + month_id)

            # AM SM
            coef_mat_am_coef = np.array([coef_mat_am[row_meshgrid_from_25km[0, ds_lst_am_ind[x]],
                                                     col_meshgrid_from_25km[0, ds_lst_am_ind[x]],
                                                     ds_ndvi_idx[0, ds_lst_am_ind[x]]*2] for x in range(len(ds_lst_am_ind))])
            coef_mat_am_intc = np.array([coef_mat_am[row_meshgrid_from_25km[0, ds_lst_am_ind[x]],
                                                     col_meshgrid_from_25km[0, ds_lst_am_ind[x]],
                                                     ds_ndvi_idx[0, ds_lst_am_ind[x]]*2+1] for x in range(len(ds_lst_am_ind))])
            smap_sm_1km_am_model_nonnan = coef_mat_am_coef * ds_lst_am_nonnan + coef_mat_am_intc

            smap_sm_1km_am_model = np.copy(smap_sm_model_mat_init_1day_1dim)
            smap_sm_1km_am_model[0, ds_lst_am_ind] = smap_sm_1km_am_model_nonnan
            smap_sm_1km_am_model[np.where(smap_sm_1km_am_model <= 0)] = np.nan
            smap_sm_1km_am_model = smap_sm_1km_am_model.reshape(matsize_smap_sm_model_1day)

            # PM SM
            coef_mat_pm_coef = np.array([coef_mat_pm[row_meshgrid_from_25km[0, ds_lst_pm_ind[x]],
                                                     col_meshgrid_from_25km[0, ds_lst_pm_ind[x]],
                                                     ds_ndvi_idx[0, ds_lst_pm_ind[x]]*2] for x in range(len(ds_lst_pm_ind))])
            coef_mat_pm_intc = np.array([coef_mat_pm[row_meshgrid_from_25km[0, ds_lst_pm_ind[x]],
                                                     col_meshgrid_from_25km[0, ds_lst_pm_ind[x]],
                                                     ds_ndvi_idx[0, ds_lst_pm_ind[x]]*2+1] for x in range(len(ds_lst_pm_ind))])
            smap_sm_1km_pm_model_nonnan = coef_mat_pm_coef * ds_lst_pm_nonnan + coef_mat_pm_intc

            smap_sm_1km_pm_model = np.copy(smap_sm_model_mat_init_1day_1dim)
            smap_sm_1km_pm_model[0, ds_lst_pm_ind] = smap_sm_1km_pm_model_nonnan
            smap_sm_1km_pm_model[np.where(smap_sm_1km_pm_model <= 0)] = np.nan
            smap_sm_1km_pm_model = smap_sm_1km_pm_model.reshape(matsize_smap_sm_model_1day)

            del(ds_lst_am, ds_lst_pm, ds_lst_am_ind, ds_lst_pm_ind, ds_lst_am_nonnan, ds_lst_pm_nonnan, ds_ndvi_idx,
                month_id, coef_mat_am_coef, coef_mat_am_intc, smap_sm_1km_am_model_nonnan, coef_mat_pm_coef,
                coef_mat_pm_intc, smap_sm_1km_pm_model_nonnan)


            # Save the daily 1 km SM model output to Geotiff files
            # Build output path
            os.chdir(path_model_op + '/' + str(yearname[iyr]))

            # Create a raster of EASE grid projection at 1 km resolution
            out_ds_tiff = gdal.GetDriverByName('GTiff').Create('smap_sm_1km_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '.tif',
                                                               len(lon_world_ease_1km), len(lat_world_ease_1km), 2, # Number of bands
                                                               gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
            out_ds_tiff.SetGeoTransform(ds_lst_1.GetGeoTransform())
            out_ds_tiff.SetProjection(ds_lst_1.GetProjection())

            # Write each band to Geotiff file
            out_ds_tiff.GetRasterBand(1).WriteArray(smap_sm_1km_am_model)
            out_ds_tiff.GetRasterBand(1).SetNoDataValue(0)
            out_ds_tiff.GetRasterBand(2).WriteArray(smap_sm_1km_pm_model)
            out_ds_tiff.GetRasterBand(2).SetNoDataValue(0)
            out_ds_tiff = None  # close dataset to write to disc

            print(str(yearname[iyr]) + str(idt+1).zfill(3))
            del(smap_sm_1km_am_model, smap_sm_1km_pm_model, ds_lst_1, out_ds_tiff)

        else:
            pass



########################################################################################################################
# 2. Downscale the 1km soil moisture model output by 9 km SMAP L2 soil moisture data

# Create initial EASE grid projection matrices
smap_sm_1km_agg_init = np.empty([len(lat_world_ease_9km), len(lon_world_ease_9km)], dtype='float32')
smap_sm_1km_agg_init[:] = np.nan
smap_sm_1km_disagg_init = np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km)], dtype='float32')
smap_sm_1km_disagg_init = smap_sm_1km_disagg_init.reshape(1, -1)
smap_sm_1km_disagg_init[:] = np.nan
smap_sm_1km_ds_init = np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km), 2], dtype='float32')
smap_sm_1km_ds_init[:] = np.nan

for iyr in range(len(yearname)):

    for imo in range(len(monthname)):

        # Load in SMAP 9km SM data
        smap_file_path = path_procdata + '/smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

        # Find the if the file exists in the directory
        if os.path.exists(smap_file_path) == True:
            f_smap_9km = h5py.File(smap_file_path, "r")
            varname_list_smap_ip = list(f_smap_9km.keys())
            for x in range(len(varname_list_smap_ip)):
                var_obj = f_smap_9km[varname_list_smap_ip[x]][()]
                exec(smap_sm_9km_name[x] + '= var_obj')
                del(var_obj)
            f_smap_9km.close()

            smap_sm_9km = np.concatenate((smap_sm_9km_am, smap_sm_9km_pm), axis=2)
            del(smap_sm_9km_am, smap_sm_9km_pm)

            # Load in MODIS LST data
            month_begin = daysofmonth_seq[0:imo, iyr].sum()
            month_end = daysofmonth_seq[0:imo + 1, iyr].sum()
            month_lenth = month_end - month_begin
            for idt in range(month_lenth):

                smap_sm_1km_file_path = path_model_op + '/' + str(yearname[iyr]) + '/smap_sm_1km_' + str(yearname[iyr]) + \
                                        str(month_begin+idt+1).zfill(3) + '.tif'

                if os.path.exists(smap_sm_1km_file_path) == True:
                    ds_smap_sm_1km = gdal.Open(smap_sm_1km_file_path)
                    smap_sm_1km = ds_smap_sm_1km.ReadAsArray()
                    smap_sm_1km = np.transpose(smap_sm_1km, (1, 2, 0))
                    smap_sm_1km_ds_output = np.copy(smap_sm_1km_ds_init)

                    for idf in range(2):
                        # Aggregate 1km SM model output to 9 km resolution, and calculate its difference with 9 km SMAP SM
                        smap_sm_1km_agg = np.copy(smap_sm_1km_agg_init)

                        smap_sm_1km_1file = smap_sm_1km[:, :, idf]
                        smap_sm_1km_1file_1dim = smap_sm_1km_1file.reshape(1, -1)
                        smap_sm_1km_1file_ind = np.where(~np.isnan(smap_sm_1km_1file_1dim))[1]

                        smap_sm_1km_agg = np.array \
                            ([np.nanmean(smap_sm_1km_1file[row_world_ease_9km_from_1km_ext33km_ind[x], :], axis=0)
                              for x in range(len(lat_world_ease_9km))])
                        smap_sm_1km_agg = np.array \
                            ([np.nanmean(smap_sm_1km_agg[:, col_world_ease_9km_from_1km_ext33km_ind[y]], axis=1)
                              for y in range(len(lon_world_ease_9km))])
                        smap_sm_1km_agg = np.fliplr(np.rot90(smap_sm_1km_agg, 3))
                        smap_sm_1km_delta = smap_sm_9km[:, :, month_lenth*idf+idt] - smap_sm_1km_agg
                        # smap_sm_1km_delta = smap_sm_1km_delta.reshape(1, -1)

                        smap_sm_1km_delta_disagg = np.array([smap_sm_1km_delta[row_meshgrid_from_9km[0, smap_sm_1km_1file_ind[x]],
                                                  col_meshgrid_from_9km[0, smap_sm_1km_1file_ind[x]]]
                                      for x in range(len(smap_sm_1km_1file_ind))])
                        smap_sm_1km_disagg = np.copy(smap_sm_1km_disagg_init)
                        smap_sm_1km_disagg[0, smap_sm_1km_1file_ind] = smap_sm_1km_delta_disagg
                        smap_sm_1km_disagg = smap_sm_1km_disagg.reshape(len(lat_world_ease_1km), len(lon_world_ease_1km))

                        smap_sm_1km_ds = smap_sm_1km_1file + smap_sm_1km_disagg
                        smap_sm_1km_ds[np.where(smap_sm_1km_ds <= 0)] = np.nan
                        smap_sm_1km_ds_output[:, :, idf] = smap_sm_1km_ds
                        del(smap_sm_1km_agg, smap_sm_1km_1file, smap_sm_1km_1file_1dim, smap_sm_1km_1file_ind, smap_sm_1km_delta,
                            smap_sm_1km_delta_disagg, smap_sm_1km_disagg, smap_sm_1km_ds)

                    # Save the daily 1 km SM model output to Geotiff files
                    # Build output path
                    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))

                    # Create a raster of EASE grid projection at 1 km resolution
                    out_ds_tiff = gdal.GetDriverByName('GTiff').Create\
                        ('smap_sm_1km_ds_' + str(yearname[iyr]) + str(month_begin+idt+1).zfill(3) + '.tif',
                         len(lon_world_ease_1km), len(lat_world_ease_1km), 2, # Number of bands
                         gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
                    out_ds_tiff.SetGeoTransform(ds_smap_sm_1km.GetGeoTransform())
                    out_ds_tiff.SetProjection(ds_smap_sm_1km.GetProjection())

                    # Loop write each band to Geotiff file
                    for idf in range(2):
                        out_ds_tiff.GetRasterBand(idf + 1).WriteArray(smap_sm_1km_ds_output[:, :, idf])
                        out_ds_tiff.GetRasterBand(idf + 1).SetNoDataValue(0)
                    out_ds_tiff = None  # close dataset to write to disc

                    print(str(yearname[iyr]) + str(month_begin+idt+1).zfill(3))
                    del (smap_sm_1km_ds_output, ds_smap_sm_1km, out_ds_tiff)

                else:
                    pass

        else:
            pass
