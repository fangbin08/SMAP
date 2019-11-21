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

# (Function 1) Extract data layers from MODIS HDF5 files and write to GeoTiff
def hdf_subdataset_extraction(hdf_files, dst_dir, subdataset_id, band_n):

    # Open the dataset
    global band_ds
    hdf_ds = gdal.Open(hdf_files, gdal.GA_ReadOnly)

    # Loop read data of specified bands from subdataset_id
    size_1dim = gdal.Open(hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly).ReadAsArray().astype(np.int16).shape
    band_array = np.empty([size_1dim[0], size_1dim[1], len(subdataset_id)], dtype=int)
    for idn in range(len(subdataset_id)):
        band_ds = gdal.Open(hdf_ds.GetSubDatasets()[subdataset_id[idn]][0], gdal.GA_ReadOnly)
        # Read into numpy array
        band_array[:, :, idn] = band_ds.ReadAsArray().astype(np.int16)

    # Build output path
    band_path = os.path.join(dst_dir, os.path.basename(os.path.splitext(hdf_files)[0]) + "-ctd" + ".tif")
    # Write raster
    out_ds = gdal.GetDriverByName('GTiff').Create(band_path, band_ds.RasterXSize, band_ds.RasterYSize, band_n, #Number of bands
                                  gdal.GDT_Int16, ['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(band_ds.GetGeoTransform())
    out_ds.SetProjection(band_ds.GetProjection())

    # Loop write each band to Geotiff file
    for idb in range(len(subdataset_id)):
        out_ds.GetRasterBand(idb+1).WriteArray(band_array[:, :, idb])
        out_ds.GetRasterBand(idb+1).SetNoDataValue(0)
    out_ds = None  #close dataset to write to disc


########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of source GLDAS data
path_modis = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS'
# Path of source LTDR NDVI data
path_smap = '/Volumes/MyPassport/SMAP_Project/NewData/SMAP'
# Path of processed data
path_procdata = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'

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

# Find the indices of each month in the list of days between 1981 - 2018
nlpyear = 1999 # non leap year
lpyear = 2000 # leap year
daysofmonth_nlp = np.array([calendar.monthrange(nlpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_nlp = [np.arange(daysofmonth_nlp[0:x].sum(), daysofmonth_nlp[0:x+1].sum()) for x in range(0, len(monthnum))]
daysofmonth_lp = np.array([calendar.monthrange(lpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_lp = [np.arange(daysofmonth_lp[0:x].sum(), daysofmonth_lp[0:x+1].sum()) for x in range(0, len(monthnum))]
ind_iflpr = np.array([int(calendar.isleap(yearname[x])) for x in range(len(yearname))]) # Find out leap years
# Generate a sequence of the days of months for all years
daysofmonth_seq = np.array([np.tile(daysofmonth_nlp[x], len(yearname)) for x in range(0, len(monthnum))])
daysofmonth_seq[1, :] = daysofmonth_seq[1, :] + ind_iflpr # Add leap days to February


########################################################################################################################
# 1. Extract data layers from MODIS HDF5 files and write to GeoTiff

src_dir = '/Users/binfang/Downloads/MODIS_new/test/input'
dst_dir = '/Users/binfang/Downloads/MODIS_new/test/output'
subdataset_id = [0, 1, 4, 5] # n^th of Layers to be extracted. For MODIS LST data: LST_Day_1km, QC_Day, LST_Night_1km, QC_Night
                             # The information of number of layers can be acquired by GetSubDatasets()
band_n = 4

# Set the boundary coordinates of the map to subset (CONUS)
lat_roi_max = 53
lat_roi_min = 25
lon_roi_max = -67
lon_roi_min = -125

os.chdir(src_dir)
hdf_files = sorted(glob.glob('*.hdf'))

for idt in range(len(hdf_files)):
    hdf_subdataset_extraction(hdf_files[idt], dst_dir, subdataset_id, band_n)
    print(hdf_files[idt]) # Print the file being processed


########################################################################################################################
# 2. Group the Geotiff files by dates from their names and
#    build virtual dataset VRT files for mosaicking MODIS geotiff files in the list

os.chdir(dst_dir)
tif_files = sorted(glob.glob('*.tif'))
tif_files_group = []
for idt in range(len(date_seq)):
    tif_files_group_1day = [tif_files.index(i) for i in tif_files if 'A' + date_seq[idt] in i]
    tif_files_group.append(tif_files_group_1day)

vrt_options = gdal.BuildVRTOptions(resampleAlg='near', addAlpha=None, bandList=None)
for idt in range(len(tif_files_group)):
    if len(tif_files_group[idt]) != 0:
        tif_files_toBuild = [tif_files[i] for i in tif_files_group[idt]]
        vrt_files_name = '_'.join(tif_files[tif_files_group[idt][0]].split('.')[0:2])
        gdal.BuildVRT('mosaic_sinu_' + vrt_files_name + '.vrt', tif_files_toBuild, options=vrt_options)
        exec('mosaic_sinu_' + vrt_files_name + '= None')


########################################################################################################################
# 3. Mosaic the list of MODIS geotiff files and reproject to lat/lon projection

# Define target SRS
dst_srs = osr.SpatialReference()
dst_srs.ImportFromEPSG(4326) # WGS 84 projection
dst_wkt = dst_srs.ExportToWkt()
error_threshold = 0.1  # error threshold
resampling = gdal.GRA_NearestNeighbour

vrt_files = sorted(glob.glob('*.vrt'))
for idt in range(len(vrt_files)):
    # Open file
    src_ds = gdal.Open(vrt_files[idt])
    mos_file_name = '_'.join(os.path.splitext(vrt_files[idt])[0].split('_')[2:4])
    # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
    tmp_ds = gdal.AutoCreateWarpedVRT(src_ds, None, dst_wkt, resampling, error_threshold)
    # Crop to CONUS extent and create the final warped raster
    dst_ds = gdal.Translate(mos_file_name + '.tif', tmp_ds,
                            projWin=[lon_roi_min, lat_roi_max, lon_roi_max, lat_roi_min])
    dst_ds = None

    print(mos_file_name)


########################################################################################################################
# 4. Process SMAP enhanced L2 radiometer half-orbit SM 9 km data

matsize_smap_1day = [len(lat_world_ease_9km), len(lon_world_ease_9km)]
smap_mat_init_1day = np.empty(matsize_smap_1day, dtype='float32')
smap_mat_init_1day[:] = np.nan


for iyr in [0]:#range(len(daysofyear)):

    os.chdir(path_smap + '/' + str(yearname[iyr]))
    smap_files_year = sorted(glob.glob('*.h5'))

    # Group SMAP data by month
    for imo in [3, 4]:#range(len(monthnum)):

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

            # 4.1 Extract SMAP data layers and rebind to daily
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



