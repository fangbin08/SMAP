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
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data/nldas'
# Path of 9 km SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'
# Path of downscaled SM
path_smap_sm_ds = '/Users/binfang/Downloads/Processing/Downscale'
# Path of VIIRS data input
path_viirs_input = '/Volumes/MyPassport/SMAP_Project/Datasets/VIIRS/Model_Input'
# Path of VIIRS data output
path_viirs_output = '/Users/binfang/Downloads/Processing/VIIRS/Model_Output'

smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Interdistance of EASE grid projection grids
interdist_ease_400m = 400.358009339824
interdist_ease_9km = 9009.093602916

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_conus_max', 'lat_conus_min', 'lon_conus_max', 'lon_conus_min',
                'lat_conus_ease_400m', 'lon_conus_ease_400m', 'lat_conus_ease_9km', 'lon_conus_ease_9km',
                'lat_conus_ease_12_5km', 'lon_conus_ease_12_5km', 'row_conus_ease_400m_ind', 'col_conus_ease_400m_ind',
                'row_conus_ease_9km_ind', 'col_conus_ease_9km_ind',
                'row_conus_ease_400m_from_9km_ind', 'col_conus_ease_400m_from_9km_ind',
                'row_conus_ease_400m_from_12_5km_ind', 'col_conus_ease_400m_from_12_5km_ind',
                'row_conus_ease_9km_from_400m_ext33km_ind', 'col_conus_ease_9km_from_400m_ext33km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2015-01-01'
end_date = '2020-12-31'

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


viirs_lai_days = np.arange(1, 366, 8)
oneyeardays_sq = np.arange(1, 367)
viirs_lai_days_ind = np.array([viirs_lai_days[np.where(np.absolute(oneyeardays_sq[x] - viirs_lai_days) ==
                                                np.amin((np.absolute(oneyeardays_sq[x] - viirs_lai_days))))].item(0)
                       for x in range(len(oneyeardays_sq))])

# Convert the 400 m from 9 km/25 km match table files to 1-d linear
col_meshgrid_from_9km, row_meshgrid_from_9km = np.meshgrid(col_conus_ease_400m_from_9km_ind, row_conus_ease_400m_from_9km_ind)
col_meshgrid_from_9km = col_meshgrid_from_9km.reshape(1, -1)
row_meshgrid_from_9km = row_meshgrid_from_9km.reshape(1, -1)

col_meshgrid_from_12_5km, row_meshgrid_from_12_5km = np.meshgrid(col_conus_ease_400m_from_12_5km_ind, row_conus_ease_400m_from_12_5km_ind)
col_meshgrid_from_12_5km = col_meshgrid_from_12_5km.reshape(1, -1)
row_meshgrid_from_12_5km = row_meshgrid_from_12_5km.reshape(1, -1)


########################################################################################################################
# 1. Implement the VIS/IR downscaling model on VIIRS LST difference to calculate 1 km soil moisture

# Load in model coefficient files
os.chdir(path_model)
f = h5py.File("ds_model_coef.hdf5", "r")
varname_list = list(f.keys())
varname_list = varname_list[3:9] + varname_list[15:21] # Load in only monthly linear regression model coefficients
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


matsize_smap_sm_model_1day = [len(lat_conus_ease_400m), len(lon_conus_ease_400m)]
smap_sm_model_mat_init_1day = np.empty(matsize_smap_sm_model_1day, dtype='float32')
smap_sm_model_mat_init_1day[:] = np.nan
smap_sm_model_mat_init_1day_1dim = smap_sm_model_mat_init_1day.reshape(1, -1)

for iyr in [3, 4]:#range(len(yearname)):

    for idt in range(90, 273):#range(daysofyear[iyr]):

        # Load in VIIRS LST data
        viirs_lst_file_path_1 = path_viirs_input + '/LST/' + str(yearname[iyr]) + '/' + \
                                'viirs_lst_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '.tif'
        if idt < daysofyear[iyr]-1:
            viirs_lst_file_path_2 = path_viirs_input + '/LST/' + str(yearname[iyr]) + '/' + \
                                    'viirs_lst_' + str(yearname[iyr]) + str(idt+2).zfill(3) + '.tif'
        elif idt == daysofyear[iyr]-1 and iyr < len(yearname)-1:
            viirs_lst_file_path_2 = path_viirs_input + '/LST/' + str(yearname[iyr+1]) + '/' + \
                                    'viirs_lst_' + str(yearname[iyr+1]) + str(1).zfill(3) + '.tif'
        else:
            viirs_lst_file_path_2 = 'None'

        # Find the if the files exist in the directory
        if os.path.exists(viirs_lst_file_path_1) == True and os.path.exists(viirs_lst_file_path_2) == True:

            ds_lst_1 = gdal.Open(viirs_lst_file_path_1)
            ds_lst_2 = gdal.Open(viirs_lst_file_path_2)
            ds_lst_1_day = ds_lst_1.GetRasterBand(1).ReadAsArray()
            ds_lst_1_night = ds_lst_1.GetRasterBand(2).ReadAsArray()
            ds_lst_2_night = ds_lst_2.GetRasterBand(2).ReadAsArray()

            # Define the geo-reference parameters of the map file
            georef_tuple = ds_lst_1.GetGeoTransform()
            geoprj_tuple = ds_lst_1.GetProjection()

            ds_lst_am = ds_lst_1_day - ds_lst_1_night
            ds_lst_pm = ds_lst_1_day - ds_lst_2_night

            ds_lst_am = ds_lst_am.reshape(1, -1)
            ds_lst_pm = ds_lst_pm.reshape(1, -1)
            ds_lst_am_ind = np.where(~np.isnan(ds_lst_am))[1]
            ds_lst_pm_ind = np.where(~np.isnan(ds_lst_pm))[1]
            ds_lst_am_nonnan = ds_lst_am[0, ds_lst_am_ind]
            ds_lst_pm_nonnan = ds_lst_pm[0, ds_lst_pm_ind]

            # Load in VIIRS lai data
            viirs_lai_file_path = path_viirs_input + '/LAI/' + str(yearname[iyr]) + '/' + \
                                   'viirs_lai_' + str(yearname[iyr]) + str(viirs_lai_days_ind[idt]).zfill(3) + '.tif'

            ds_lai = gdal.Open(viirs_lai_file_path)
            ds_lai_idx = ds_lai.GetRasterBand(1).ReadAsArray()
            ds_lai_idx = np.nan_to_num(ds_lai_idx)
            ds_lai_idx = np.fix(ds_lai_idx*10).astype(int)
            ds_lai_idx[np.where(ds_lai_idx >= 10)] = 9

            del(ds_lst_2, ds_lst_1_day, ds_lst_1_night, ds_lst_2_night, ds_lai)


            # Extract coefficient and intercept from the position indices of corresponding monthly model file
            ds_lai_idx = ds_lai_idx.reshape(1, -1)

            month_id = str(datetime.datetime.strptime(str(yearname[iyr]) + '+' + str(idt+1).zfill(3), '%Y+%j').month).zfill(2)
            exec('coef_mat_am = ' + 'coef_mat_am_' + month_id)
            exec('coef_mat_pm = ' + 'coef_mat_pm_' + month_id)

            # AM SM
            coef_mat_am_coef = np.array([coef_mat_am[row_meshgrid_from_12_5km[0, ds_lst_am_ind[x]],
                                                     col_meshgrid_from_12_5km[0, ds_lst_am_ind[x]],
                                                     ds_lai_idx[0, ds_lst_am_ind[x]]*2] for x in range(len(ds_lst_am_ind))])
            coef_mat_am_intc = np.array([coef_mat_am[row_meshgrid_from_12_5km[0, ds_lst_am_ind[x]],
                                                     col_meshgrid_from_12_5km[0, ds_lst_am_ind[x]],
                                                     ds_lai_idx[0, ds_lst_am_ind[x]]*2+1] for x in range(len(ds_lst_am_ind))])
            smap_sm_400m_am_model_nonnan = coef_mat_am_coef * ds_lst_am_nonnan + coef_mat_am_intc

            smap_sm_400m_am_model = np.copy(smap_sm_model_mat_init_1day_1dim)
            smap_sm_400m_am_model[0, ds_lst_am_ind] = smap_sm_400m_am_model_nonnan
            smap_sm_400m_am_model[np.where(smap_sm_400m_am_model <= 0)] = np.nan
            smap_sm_400m_am_model = smap_sm_400m_am_model.reshape(matsize_smap_sm_model_1day)

            # PM SM
            coef_mat_pm_coef = np.array([coef_mat_pm[row_meshgrid_from_12_5km[0, ds_lst_pm_ind[x]],
                                                     col_meshgrid_from_12_5km[0, ds_lst_pm_ind[x]],
                                                     ds_lai_idx[0, ds_lst_pm_ind[x]]*2] for x in range(len(ds_lst_pm_ind))])
            coef_mat_pm_intc = np.array([coef_mat_pm[row_meshgrid_from_12_5km[0, ds_lst_pm_ind[x]],
                                                     col_meshgrid_from_12_5km[0, ds_lst_pm_ind[x]],
                                                     ds_lai_idx[0, ds_lst_pm_ind[x]]*2+1] for x in range(len(ds_lst_pm_ind))])
            smap_sm_400m_pm_model_nonnan = coef_mat_pm_coef * ds_lst_pm_nonnan + coef_mat_pm_intc

            smap_sm_400m_pm_model = np.copy(smap_sm_model_mat_init_1day_1dim)
            smap_sm_400m_pm_model[0, ds_lst_pm_ind] = smap_sm_400m_pm_model_nonnan
            smap_sm_400m_pm_model[np.where(smap_sm_400m_pm_model <= 0)] = np.nan
            smap_sm_400m_pm_model = smap_sm_400m_pm_model.reshape(matsize_smap_sm_model_1day)

            del(ds_lst_am, ds_lst_pm, ds_lst_am_ind, ds_lst_pm_ind, ds_lst_am_nonnan, ds_lst_pm_nonnan, ds_lai_idx,
                month_id, coef_mat_am_coef, coef_mat_am_intc, smap_sm_400m_am_model_nonnan, coef_mat_pm_coef,
                coef_mat_pm_intc, smap_sm_400m_pm_model_nonnan, coef_mat_am, coef_mat_pm)


            # Save the daily 1 km SM model output to Geotiff files
            # Build output path
            os.chdir(path_viirs_output + '/' + str(yearname[iyr]))

            # Create a raster of EASE grid projection at 1 km resolution
            out_ds_tiff = gdal.GetDriverByName('GTiff').Create('smap_sm_400m_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '.tif',
                                                               len(lon_conus_ease_400m), len(lat_conus_ease_400m), 2, # Number of bands
                                                               gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
            out_ds_tiff.SetGeoTransform(georef_tuple)
            out_ds_tiff.SetProjection(geoprj_tuple)

            # Write each band to Geotiff file
            out_ds_tiff.GetRasterBand(1).WriteArray(smap_sm_400m_am_model)
            out_ds_tiff.GetRasterBand(1).SetNoDataValue(0)
            out_ds_tiff.GetRasterBand(2).WriteArray(smap_sm_400m_pm_model)
            out_ds_tiff.GetRasterBand(2).SetNoDataValue(0)
            out_ds_tiff = None  # close dataset to write to disc

            print(str(yearname[iyr]) + str(idt+1).zfill(3))
            del(smap_sm_400m_am_model, smap_sm_400m_pm_model, ds_lst_1, out_ds_tiff, georef_tuple, geoprj_tuple)

        else:
            pass



########################################################################################################################
# 2. Downscale the 400m soil moisture model output by 9 km SMAP L2 soil moisture data

# Create initial EASE grid projection matrices
smap_sm_400m_agg_init = np.empty([len(lat_conus_ease_9km), len(lon_conus_ease_9km)], dtype='float32')
smap_sm_400m_agg_init[:] = np.nan
smap_sm_400m_disagg_init = np.empty([len(lat_conus_ease_400m), len(lon_conus_ease_400m)], dtype='float32')
smap_sm_400m_disagg_init = smap_sm_400m_disagg_init.reshape(1, -1)
smap_sm_400m_disagg_init[:] = np.nan
smap_sm_400m_ds_init = np.empty([len(lat_conus_ease_400m), len(lon_conus_ease_400m), 2], dtype='float32')
smap_sm_400m_ds_init[:] = np.nan

for iyr in range(len(yearname)):

    for imo in range(len(monthname)):

        # Load in SMAP 9km SM data
        smap_file_path = path_smap + '/9km/' + 'smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

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
            smap_sm_9km_nldas = smap_sm_9km[row_conus_ease_9km_ind[0]:row_conus_ease_9km_ind[-1]+1,
                                col_conus_ease_9km_ind[0]:col_conus_ease_9km_ind[-1]+1, :]
            del(smap_sm_9km_am, smap_sm_9km_pm, smap_sm_9km)

            # Load in VIIRS SM data
            month_begin = daysofmonth_seq[0:imo, iyr].sum()
            month_end = daysofmonth_seq[0:imo + 1, iyr].sum()
            month_lenth = month_end - month_begin
            for idt in range(month_lenth):

                smap_sm_400m_file_path = path_viirs_output + '/' + str(yearname[iyr]) + '/smap_sm_400m_' + str(yearname[iyr]) + \
                                        str(month_begin+idt+1).zfill(3) + '.tif'

                if os.path.exists(smap_sm_400m_file_path) == True:
                    ds_smap_sm_400m = gdal.Open(smap_sm_400m_file_path)
                    smap_sm_400m = ds_smap_sm_400m.ReadAsArray()
                    smap_sm_400m = np.transpose(smap_sm_400m, (1, 2, 0))
                    smap_sm_400m_ds_output = np.copy(smap_sm_400m_ds_init)

                    for idf in range(2):
                        # Aggregate 400m SM model output to 9 km resolution, and calculate its difference with 9 km SMAP SM
                        smap_sm_400m_agg = np.copy(smap_sm_400m_agg_init)

                        smap_sm_400m_1file = smap_sm_400m[:, :, idf]
                        smap_sm_400m_1file_1dim = smap_sm_400m_1file.reshape(1, -1)
                        smap_sm_400m_1file_ind = np.where(~np.isnan(smap_sm_400m_1file_1dim))[1]

                        smap_sm_400m_agg = np.array \
                            ([np.nanmean(smap_sm_400m_1file[row_conus_ease_9km_from_400m_ext33km_ind[x], :], axis=0)
                              for x in range(len(lat_conus_ease_9km))])
                        smap_sm_400m_agg = np.array \
                            ([np.nanmean(smap_sm_400m_agg[:, col_conus_ease_9km_from_400m_ext33km_ind[y]], axis=1)
                              for y in range(len(lon_conus_ease_9km))])
                        smap_sm_400m_agg = np.fliplr(np.rot90(smap_sm_400m_agg, 3))
                        smap_sm_400m_delta = smap_sm_9km_nldas[:, :, month_lenth*idf+idt] - smap_sm_400m_agg
                        # smap_sm_400m_delta = smap_sm_400m_delta.reshape(1, -1)

                        smap_sm_400m_delta_disagg = np.array([smap_sm_400m_delta[row_meshgrid_from_9km[0, smap_sm_400m_1file_ind[x]],
                                                                               col_meshgrid_from_9km[0, smap_sm_400m_1file_ind[x]]]
                                                             for x in range(len(smap_sm_400m_1file_ind))])
                        smap_sm_400m_disagg = np.copy(smap_sm_400m_disagg_init)
                        smap_sm_400m_disagg[0, smap_sm_400m_1file_ind] = smap_sm_400m_delta_disagg
                        smap_sm_400m_disagg = smap_sm_400m_disagg.reshape(len(lat_conus_ease_400m), len(lon_conus_ease_400m))

                        smap_sm_400m_ds = smap_sm_400m_1file + smap_sm_400m_disagg
                        smap_sm_400m_ds[np.where(smap_sm_400m_ds <= 0)] = np.nan
                        smap_sm_400m_ds_output[:, :, idf] = smap_sm_400m_ds
                        del(smap_sm_400m_agg, smap_sm_400m_1file, smap_sm_400m_1file_1dim, smap_sm_400m_1file_ind, smap_sm_400m_delta,
                            smap_sm_400m_delta_disagg, smap_sm_400m_disagg, smap_sm_400m_ds)

                    # Save the daily 1 km SM model output to Geotiff files
                    # Build output path
                    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))

                    # Create a raster of EASE grid projection at 1 km resolution
                    out_ds_tiff = gdal.GetDriverByName('GTiff').Create\
                        ('smap_sm_400m_ds_' + str(yearname[iyr]) + str(month_begin+idt+1).zfill(3) + '.tif',
                         len(lon_conus_ease_400m), len(lat_conus_ease_400m), 2, # Number of bands
                         gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
                    out_ds_tiff.SetGeoTransform(ds_smap_sm_400m.GetGeoTransform())
                    out_ds_tiff.SetProjection(ds_smap_sm_400m.GetProjection())

                    # Loop write each band to Geotiff file
                    for idf in range(2):
                        out_ds_tiff.GetRasterBand(idf + 1).WriteArray(smap_sm_400m_ds_output[:, :, idf])
                        out_ds_tiff.GetRasterBand(idf + 1).SetNoDataValue(0)
                    out_ds_tiff = None  # close dataset to write to disc

                    print(str(yearname[iyr]) + str(month_begin+idt+1).zfill(3))
                    del (smap_sm_400m_ds_output, ds_smap_sm_400m, out_ds_tiff)

                else:
                    pass

        else:
            pass

