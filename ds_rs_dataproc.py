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

# (Function 1) Extract data layers from MODIS HDF5 files, filter low quality pixels and write to GeoTiff
# (The function only works for MODIS LST and NDVI data sets!)

def hdf_subdataset_extraction(hdf_files, subdataset_id, band_n):

    # Open the dataset
    global band_ds
    hdf_ds = gdal.Open(hdf_files, gdal.GA_ReadOnly)

    # Loop read data of specified bands from subdataset_id
    size_1dim = gdal.Open(hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly).ReadAsArray().astype(np.int16).shape
    band_array = np.empty([size_1dim[0], size_1dim[1], len(subdataset_id)//2])
    for idn in range(len(subdataset_id)//2):
        band_ds = gdal.Open(hdf_ds.GetSubDatasets()[subdataset_id[idn*2]][0], gdal.GA_ReadOnly)
        band_ds_arr = band_ds.ReadAsArray().astype(np.float32)
        band_ds_arr_qa = gdal.Open(hdf_ds.GetSubDatasets()[subdataset_id[idn*2+1]][0], gdal.GA_ReadOnly)

        # Find if the dataset is MODIS NDVI or MODIS LST
        if len(subdataset_id) > 2: # MODIS LST
            band_ds_arr = band_ds_arr * 0.02
            band_ds_arr_qa = band_ds_arr_qa.ReadAsArray().astype(np.uint8)
            band_ds_arr_qa = band_ds_arr_qa.reshape(-1, 1)
            band_ds_arr_qa_bin = [np.binary_repr(band_ds_arr_qa[x].item(), width=8) for x in range(len(band_ds_arr_qa))]
            ind_qa = np.array([i for i in range(len(band_ds_arr_qa_bin)) if band_ds_arr_qa_bin[i][0] == '0'
                               and band_ds_arr_qa_bin[i][6] == '0'])
        else: # MODIS NDVI
            band_ds_arr = band_ds_arr * 0.0001
            band_ds_arr_qa = band_ds_arr_qa.ReadAsArray().astype(np.uint16)
            band_ds_arr_qa = band_ds_arr_qa.reshape(-1, 1)
            band_ds_arr_qa_bin = [np.binary_repr(band_ds_arr_qa[x].item(), width=16) for x in range(len(band_ds_arr_qa))]
            ind_qa = np.array([i for i in range(len(band_ds_arr_qa_bin)) if band_ds_arr_qa_bin[i][-1] == '0'])

        band_ds_arr_qa_mask = np.arange(band_ds_arr.shape[0]*band_ds_arr.shape[1])
        band_ds_arr_qa_mask = np.isin(band_ds_arr_qa_mask, ind_qa).astype(int)
        band_ds_arr_qa_mask = band_ds_arr_qa_mask.reshape(band_ds_arr.shape[0], band_ds_arr.shape[1])
        band_ds_arr = band_ds_arr * band_ds_arr_qa_mask
        band_ds_arr[np.where(band_ds_arr <= 0)] = np.nan

        # Write into numpy array
        band_array[:, :, idn] = band_ds_arr
        del(band_ds_arr, band_ds_arr_qa, band_ds_arr_qa_bin, ind_qa, band_ds_arr_qa_mask)

    # Build output path
    # band_path = os.path.join(path_modis_op, os.path.basename(os.path.splitext(hdf_files)[0]) + "-ctd" + ".tif")
    # Write raster
    out_ds = gdal.GetDriverByName('MEM').Create('', band_ds.RasterXSize, band_ds.RasterYSize, band_n, #Number of bands
                                  gdal.GDT_Float32)
    out_ds.SetGeoTransform(band_ds.GetGeoTransform())
    out_ds.SetProjection(band_ds.GetProjection())

    # Loop write each band to Geotiff file
    for idb in range(len(subdataset_id)//2):
        out_ds.GetRasterBand(idb+1).WriteArray(band_array[:, :, idb])
        out_ds.GetRasterBand(idb+1).SetNoDataValue(0)
    # out_ds = None  #close dataset to write to disc

    return out_ds

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of source MODIS data
path_modis = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/HDF'
# Path of output MODIS data
path_modis_op = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Output'
# Path of MODIS data for SM downscaling model input
path_modis_model = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Model_Input'
# Path of SMAP data
path_smap = '/Volumes/MyPassport/SMAP_Project/NewData/SMAP'
# Path of processed data
path_procdata = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
subfolders = np.arange(2015, 2019+1, 1)
subfolders = [str(i).zfill(4) for i in subfolders]

# Load in variables
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_geo_1km', 'lon_world_geo_1km', 'row_world_ease_1km_from_geo_1km_ind',
                'col_world_ease_1km_from_geo_1km_ind']

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

# MODIS data layer information for extraction
# n^th of Layers to be extracted. The information of number of layers can be acquired by function GetSubDatasets()
subdataset_id_lst = [0, 1, 4, 5]  # For MODIS LST data: extract LST_Day_1km, QC_Day, LST_Night_1km, QC_Night
band_n_lst = 2  # For MODIS LST data: save LST_Day_1km, LST_Night_1km
subdataset_id_ndvi = [0, 2] # For MODIS NDVI data: extract NDVI, VI Quality
band_n_ndvi = 1 # For MODIS NDVI data: save NDVI

# Set the boundary coordinates of the map to subset (World)
lat_roi_max = 90
lat_roi_min = -90
lon_roi_max = 180
lon_roi_min = -180

modis_folders = [lst_folder, ndvi_folder]
subdataset_id = [subdataset_id_lst, subdataset_id_ndvi]
band_n = [band_n_lst, band_n_ndvi]
modis_var_names = ['modis_lst_1km', 'modis_ndvi_1km']

# Define target SRS
dst_srs = osr.SpatialReference()
dst_srs.ImportFromEPSG(6933)  # EASE grid projection
dst_wkt = dst_srs.ExportToWkt()
gts = (-17367530.44516138, 1000.89502334956, 0, 7314540.79258289, 0, -1000.89502334956)


########################################################################################################################
# 1. Extract, mosaic and reproject MODIS tile data to EASE grid projection at 1 km
# 1.1 Extract data layers from MODIS HDF5 files and write to GeoTiff

# for ifo in range(len(modis_folders)): # MODIS LST and NDVI subfolders
#
#     for iyr in range(4, len(yearname)):
#
#         for imo in range(len(monthname)):
#
#             os.chdir(path_modis + modis_folders[ifo] + subfolders[iyr] + '/' + monthname[imo])
#             hdf_files = sorted(glob.glob('*.hdf'))
#
#             for idt in range(len(hdf_files)):
#                 # ctd_file_path = path_modis_op + modis_folders[ifo] + subfolders[iyr] + '/' + monthname[imo]
#                 hdf_subdataset_extraction(hdf_files[idt], subdataset_id[ifo], band_n[ifo])
#                 print(hdf_files[idt]) # Print the file being processed


# # 1.2 Group the Geotiff files by dates from their names and
# # build virtual dataset VRT files for mosaicking MODIS geotiff files in the list
#
# vrt_options = gdal.BuildVRTOptions(resampleAlg='near', addAlpha=None, bandList=[1])
#
# for ifo in range(len(modis_folders)): # MODIS LST and NDVI subfolders
#
#     for iyr in range(len(subfolders)):
#
#         for imo in range(len(monthname)):
#
#             os.chdir(path_modis + modis_folders[ifo] + subfolders[iyr] + '/' + monthname[imo])
#             hdf_files = sorted(glob.glob('*.hdf'))
#
#             hdf_file_name = [hdf_files[x].split('.')[1] for x in range(len(hdf_files))]
#             hdf_file_name_unique = sorted(list(set(hdf_file_name)))
#
#             # Group the MODIS tile files by each day
#             hdf_files_group = []
#             for idt in range(len(hdf_file_name_unique)):
#                 hdf_files_group_1day = [hdf_files.index(i) for i in hdf_files if hdf_file_name_unique[idt] in i]
#                 hdf_files_group.append(hdf_files_group_1day)
#
#             # build virtual dataset VRT file
#             for idt in [1]:#range(len(hdf_files_group)):
#                 if len(hdf_files_group[idt]) != 0:
#                     hdf_files_toBuild = [hdf_files[i] for i in hdf_files_group[idt]]
#                     vrt_files_name = '_'.join(hdf_files[hdf_files_group[idt][0]].split('.')[0:2])
#                     gdal.BuildVRT('mosaic_sinu_' + vrt_files_name + '.vrt', hdf_files_toBuild, options=vrt_options)
#                     exec('mosaic_sinu_' + vrt_files_name + '= None')
#                     print('mosaic_sinu_' + vrt_files_name + '.vrt')
#                 else:
#                     pass



# 1.3 Mosaic the list of MODIS geotiff files and reproject to lat/lon projection

modis_mat_ease_1day_init = np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km)], dtype='float32')
modis_mat_ease_1day_init[:] = np.nan

for ifo in range(len(modis_folders)): # MODIS LST and NDVI subfolders

    for iyr in range(4, len(subfolders)):

        for imo in range(6, len(monthname)):

            path_month = path_modis + modis_folders[ifo] + subfolders[iyr] + '/' + monthname[imo]
            os.chdir(path_month)
            hdf_files = sorted(glob.glob('*.hdf'))

            if len(hdf_files) != 0:

                hdf_file_name = [hdf_files[x].split('.')[1] for x in range(len(hdf_files))]
                hdf_file_name_unique = sorted(list(set(hdf_file_name)))

                # Group the MODIS tile files by each day
                for idt in range(len(hdf_file_name_unique)):

                    hdf_files_toBuild_ind = [hdf_files.index(i) for i in hdf_files if hdf_file_name_unique[idt] in i]
                    hdf_files_toBuild = [hdf_files[i] for i in hdf_files_toBuild_ind]

                    hdf_files_list = []
                    for idf in range(len(hdf_files_toBuild)):
                        extr_file = hdf_subdataset_extraction(path_month + '/' + hdf_files_toBuild[idf],
                                                              subdataset_id[ifo], band_n[ifo])
                        hdf_files_list.append(extr_file)  # Append the processed hdf to the file list being merged
                        print(hdf_files_toBuild[idf])  # Print the file being processed
                        del(extr_file)

                    # Open file and warp the target raster dimensions and geotransform
                    out_ds = gdal.Warp('', hdf_files_list, format='MEM', outputBounds=[-180, -90, 180, 90], xRes=0.01, yRes=0.01,
                                       dstSRS='EPSG:4326', warpOptions=['SKIP_NOSOURCE=YES'], errorThreshold=0,
                                       resampleAlg=gdal.GRA_NearestNeighbour)

                    modis_mat = out_ds.ReadAsArray()
                    modis_mat[np.where(modis_mat <= 0)] = np.nan
                    modis_mat = np.atleast_3d(modis_mat)
                    # For MODIS LST data layers
                    if modis_mat.shape[0] == 2:
                        modis_mat = np.transpose(modis_mat, (1, 2, 0))
                    else:
                        pass

                    # Create initial EASE grid projection matrices at 1 km
                    modis_mat_ease = \
                        np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km), modis_mat.shape[2]], dtype='float32')
                    modis_mat_ease[:] = np.nan

                    for idm in range(modis_mat.shape[2]):
                        modis_mat_ease_1day = np.copy(modis_mat_ease_1day_init)
                        modis_mat_1day = modis_mat[:, :, idm]
                        modis_mat_ease_1day = np.array \
                            ([np.nanmean(modis_mat_1day[row_world_ease_1km_from_geo_1km_ind[x], :], axis=0)
                              for x in range(len(lat_world_ease_1km))])
                        modis_mat_ease_1day = np.array \
                            ([np.nanmean(modis_mat_ease_1day[:, col_world_ease_1km_from_geo_1km_ind[y]], axis=1)
                              for y in range(len(lon_world_ease_1km))])
                        modis_mat_ease_1day = np.fliplr(np.rot90(modis_mat_ease_1day, 3))
                        modis_mat_ease[:, :, idm] = modis_mat_ease_1day
                        del(modis_mat_ease_1day, modis_mat_1day)

                    del(modis_mat, out_ds, hdf_files_list, hdf_files_toBuild, hdf_files_toBuild_ind)


                    # 1.4 Save the daily MODIS LST/NDVI data to Geotiff files
                    # Build output path
                    path_writefile = path_modis_model + modis_folders[ifo] + subfolders[iyr]
                    # os.chdir(path_modis_model + modis_folders[ifo] + subfolders[iyr])

                    # Create a raster of EASE grid projection at 1 km resolution
                    out_ds_tiff = gdal.GetDriverByName('GTiff').Create\
                        (path_writefile + '/' + modis_var_names[ifo] + '_' + hdf_file_name_unique[idt][1:] + '.tif',
                         len(lon_world_ease_1km), len(lat_world_ease_1km), band_n[ifo],  # Number of bands
                         gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
                    out_ds_tiff.SetGeoTransform(gts)
                    out_ds_tiff.SetProjection(dst_wkt)

                    # Loop write each band to Geotiff file
                    for idl in range(band_n[ifo]):
                        out_ds_tiff.GetRasterBand(idl + 1).WriteArray(modis_mat_ease[:, :, idl])
                        out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
                    out_ds_tiff = None  # close dataset to write to disc

                    print(modis_var_names[ifo] + '_' + hdf_file_name_unique[idt][1:])
                    del(modis_mat_ease)

            else:
                pass

            del (hdf_files, path_month, hdf_file_name_unique)







########################################################################################################################
# 2. Process SMAP enhanced L2 radiometer half-orbit SM 9 km data

matsize_smap_1day = [len(lat_world_ease_9km), len(lon_world_ease_9km)]
smap_mat_init_1day = np.empty(matsize_smap_1day, dtype='float32')
smap_mat_init_1day[:] = np.nan


for iyr in range(len(daysofyear)):

    os.chdir(path_smap + '/' + str(yearname[iyr]))
    smap_files_year = sorted(glob.glob('*.h5'))

    # Group SMAP data by month
    for imo in range(len(monthnum)):

        os.chdir(path_smap + '/' + str(yearname[iyr]))
        smap_files_group_1month = [smap_files_year.index(i) for i in smap_files_year if str(yearname[iyr]) + monthname[imo] in i]

        # Process each month
        if len(smap_files_group_1month) != 0:
            smap_files_month = [smap_files_year[smap_files_group_1month[i]] for i in range(len(smap_files_group_1month))]

            # Create initial empty matrices for monthly SMAP final output data
            matsize_smap = [matsize_smap_1day[0], matsize_smap_1day[1], daysofmonth_seq[imo, iyr]]
            smap_mat_month_am = np.empty(matsize_smap, dtype='float32')
            smap_mat_month_am[:] = np.nan
            smap_mat_month_pm = np.copy(smap_mat_month_am)

            # Extract SMAP data layers and rebind to daily
            for idt in range(daysofmonth_seq[imo, iyr]):
                smap_files_group_1day = [smap_files_month.index(i) for i in smap_files_month if
                                         str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_files_1day = [smap_files_month[smap_files_group_1day[i]] for i in
                                    range(len(smap_files_group_1day))]
                smap_files_group_1day_am = [smap_files_1day.index(i) for i in smap_files_1day if
                                         'D_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_files_group_1day_pm = [smap_files_1day.index(i) for i in smap_files_1day if
                                         'A_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_mat_group_1day = \
                    np.empty([matsize_smap_1day[0], matsize_smap_1day[1], len(smap_files_group_1day)], dtype='float32')
                smap_mat_group_1day[:] = np.nan

                # Read swath files within a day and stack
                for ife in range(len(smap_files_1day)):
                    smap_mat_1file = np.copy(smap_mat_init_1day)
                    fe_smap = h5py.File(smap_files_1day[ife], "r")
                    group_list_smap = list(fe_smap.keys())
                    smap_data_group = fe_smap[group_list_smap[1]]
                    varname_list_smap = list(smap_data_group.keys())
                    # Extract variables
                    col_ind = smap_data_group[varname_list_smap[0]][()]
                    row_ind = smap_data_group[varname_list_smap[1]][()]
                    sm_flag = smap_data_group[varname_list_smap[14]][()]
                    sm = smap_data_group[varname_list_smap[20]][()]
                    sm[np.where(sm == -9999)] = np.nan
                    sm[np.where((sm_flag == 7) & (sm_flag == 15))] = np.nan # Refer to the results of np.binary_repr

                    smap_mat_1file[row_ind, col_ind] = sm
                    smap_mat_group_1day[:, :, ife] = smap_mat_1file
                    print(smap_files_1day[ife])
                    fe_smap.close()

                    del(smap_mat_1file, fe_smap, group_list_smap, smap_data_group, varname_list_smap, col_ind, row_ind,
                        sm_flag, sm)

                smap_mat_1day_am = np.nanmean(smap_mat_group_1day[:, :, smap_files_group_1day_am], axis=2)
                smap_mat_1day_pm = np.nanmean(smap_mat_group_1day[:, :, smap_files_group_1day_pm], axis=2)
                # plt.imshow(np.nanmean(np.concatenate((np.atleast_3d(smap_mat_1day_am),
                # np.atleast_3d(smap_mat_1day_pm)), axis=2), axis=2))
                del(smap_mat_group_1day, smap_files_group_1day, smap_files_1day)

                smap_mat_month_am[:, :, idt] = smap_mat_1day_am
                smap_mat_month_pm[:, :, idt] = smap_mat_1day_pm
                del(smap_mat_1day_am, smap_mat_1day_pm)

            # Save file
            os.chdir(path_procdata)
            var_name = ['smap_sm_9km_am_' + str(yearname[iyr]) + monthname[imo],
                        'smap_sm_9km_pm_' + str(yearname[iyr]) + monthname[imo]]
            data_name = ['smap_mat_month_am', 'smap_mat_month_pm']

            with h5py.File('smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5', 'w') as f:
                for idv in range(len(var_name)):
                    f.create_dataset(var_name[idv], data=eval(data_name[idv]))
            f.close()
            del(smap_mat_month_am, smap_mat_month_pm)

        else:
            pass



