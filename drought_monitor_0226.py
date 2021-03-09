import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import glob
import h5py
import gdal
import fiona
import rasterio
import calendar
import datetime
import osr
import pandas as pd
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, Affine
from rasterio.windows import Window
from rasterio.crs import CRS
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from itertools import chain
from sklearn import preprocessing

# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


#########################################################################################
# (Function 1) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind

#########################################################################################
# (Function 2) Generate lat/lon tables of Geographic projection in world/CONUS

def geo_coord_gen(lat_geo_extent_max, lat_geo_extent_min, lon_geo_extent_max, lon_geo_extent_min, cellsize):
    lat_geo_output = np.linspace(lat_geo_extent_max - cellsize / 2, lat_geo_extent_min + cellsize / 2,
                                    num=int((lat_geo_extent_max - lat_geo_extent_min) / cellsize))
    lat_geo_output = np.round(lat_geo_output, decimals=3)
    lon_geo_output = np.linspace(lon_geo_extent_min + cellsize / 2, lon_geo_extent_max - cellsize / 2,
                                    num=int((lon_geo_extent_max - lon_geo_extent_min) / cellsize))
    lon_geo_output = np.round(lon_geo_output, decimals=3)

    return lat_geo_output, lon_geo_output

#########################################################################################
# (Function 3) Convert latitude and longitude to the corresponding row and col in the
# EASE grid VERSION 2 used at CATDS since processor version 2.7, January 2014

def geo2easeGridV2(latitude, longitude, interdist, num_row, num_col):
    # Constant
    a = 6378137  # equatorial radius
    f = 1 / 298.257223563  # flattening
    b = 6356752.314  # polar radius b=a(1-f)
    e = 0.0818191908426  # eccentricity sqrt(2f-f^2)
    c = interdist  # interdistance pixel
    nl = num_row  # Number of lines
    nc = num_col  # Number of columns
    s0 = (nl - 1) / 2
    r0 = (nc - 1) / 2
    phi0 = 0
    lambda0 = 0  # map reference longitude
    phi1 = 30  # latitude true scale
    k0 = np.cos(np.deg2rad(phi1)) / np.sqrt(1 - (e ** 2 * np.sin(np.deg2rad(phi1)) ** 2))
    q = (1 - e ** 2) * ((np.sin(np.deg2rad(latitude)) / (1 - e ** 2 * np.sin(np.deg2rad(latitude)) ** 2)) -
                        (1 / (2 * e)) * np.log(
                (1 - e * np.sin(np.deg2rad(latitude))) / (1 + e * np.sin(np.deg2rad(latitude)))))
    x = a * k0 * (longitude - lambda0) * np.pi / 180
    y = a * q / (2 * k0)
    # as Brodzik et al
    column = np.round(r0 + (x / c)).astype(int)
    row = np.round(s0 - (y / c)).astype(int)

    del a, f, b, e, c, nl, nc, s0, r0, phi0, lambda0, phi1, k0, q, x, y

    return row, column

#########################################################################################
# (Function 4) Find and map the corresponding index numbers for the low spatial resolution
# row/col tables from the high spatial resolution row/col tables. The output is 1-dimensional
# nested list array containing index numbers. (Aggregate)

def find_easeind_lofrhi(lat_hires, lon_hires, interdist_lowres, num_row_lowres, num_col_lowres, row_lowres_ind, col_lowres_ind):

    lon_meshgrid, lat_meshgrid = np.meshgrid(lon_hires, lat_hires)

    # Select only the first row + first column to find the row/column indices
    lat_meshgrid_array = np.concatenate((lat_meshgrid[:, 0], lat_meshgrid[0, :]), axis=0)
    lon_meshgrid_array = np.concatenate((lon_meshgrid[:, 0], lon_meshgrid[0, :]), axis=0)

    [row_ind_toresp, col_ind_toresp] = \
        geo2easeGridV2(lat_meshgrid_array, lon_meshgrid_array, interdist_lowres,
                       num_row_lowres, num_col_lowres)

    row_ind_toresp = row_ind_toresp[:(len(lat_hires))]
    col_ind_toresp = col_ind_toresp[(len(lat_hires)):]

    # Assign the low resolution grids with corresponding high resolution grids index numbers
    row_ease_dest_init = []
    for x in range(len(row_lowres_ind)):
        row_ind = np.where(row_ind_toresp == row_lowres_ind[x])
        row_ind = np.array(row_ind).ravel()
        row_ease_dest_init.append(row_ind)

    row_ease_dest_ind = np.asarray(row_ease_dest_init)

    col_ease_dest_init = []
    for x in range(len(col_lowres_ind)):
        col_ind = np.where(col_ind_toresp == col_lowres_ind[x])
        col_ind = np.array(col_ind).ravel()
        col_ease_dest_init.append(col_ind)

    col_ease_dest_ind = np.asarray(col_ease_dest_init)

    # Assign the empty to-be-resampled grids with index numbers of corresponding nearest destination grids
    for x in range(len(row_ease_dest_ind)):
        if len(row_ease_dest_ind[x]) == 0 and x != 0 and x != len(row_ease_dest_ind)-1:
            # Exclude the first and last elements
            row_ease_dest_ind[x] = np.array([row_ease_dest_ind[x - 1], row_ease_dest_ind[x + 1]]).ravel()
        else:
            pass

    for x in range(len(col_ease_dest_ind)):
        if len(col_ease_dest_ind[x]) == 0 and x != 0 and x != len(col_ease_dest_ind)-1:
            # Exclude the first and last elements
            col_ease_dest_ind[x] = np.array([col_ease_dest_ind[x - 1], col_ease_dest_ind[x + 1]]).ravel()
        else:
            pass

    return row_ease_dest_ind, col_ease_dest_ind


########################################################################################################################
# Function 5. Subset and reproject the Geotiff data to WGS84 projection

def sub_n_reproj(input_mat, kwargs_sub, sub_window, output_crs):
    # Get the georeference and bounding parameters of subset image
    kwargs_sub.update({
        'height': sub_window.height,
        'width': sub_window.width,
        'transform': rasterio.windows.transform(sub_window, kwargs_sub['transform'])})

    # Generate subset rasterio dataset
    input_ds_subset = MemoryFile().open(**kwargs_sub)
    input_mat = np.expand_dims(input_mat, axis=0)
    input_ds_subset.write(input_mat)

    # Reproject a dataset
    transform_reproj, width_reproj, height_reproj = \
        calculate_default_transform(input_ds_subset.crs, output_crs,
                                    input_ds_subset.width, input_ds_subset.height, *input_ds_subset.bounds)
    kwargs_reproj = input_ds_subset.meta.copy()
    kwargs_reproj.update({
            'crs': output_crs,
            'transform': transform_reproj,
            'width': width_reproj,
            'height': height_reproj
        })

    output_ds = MemoryFile().open(**kwargs_reproj)
    reproject(source=rasterio.band(input_ds_subset, 1), destination=rasterio.band(output_ds, 1),
              src_transform=input_ds_subset.transform, src_crs=input_ds_subset.crs,
              dst_transform=transform_reproj, dst_crs=output_crs, resampling=Resampling.nearest)

    return output_ds

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of processed data
path_procdata = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/1km/gldas'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of Australia soil data
path_aus_soil = '/Volumes/MyPassport/SMAP_Project/Datasets/Australia'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200605'
# Path of preview
path_preview = '/Users/binfang/Documents/SMAP_Project/results/results_191202/preview'
# Path of swdi data
path_swdi = '/Volumes/MyPassport/SMAP_Project/Datasets/Australia/swdi'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data'
# Path of processed data
path_processed = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/1km/gldas'
# Path of processed data 2
path_processed_2 = '/Users/binfang/Downloads/Processing/processed_data'
# Path of GPM
path_gpm = '/Volumes/MyPassport/SMAP_Project/Datasets/GPM'
# Path of ISMN
path_ismn = '/Volumes/MyPassport/SMAP_Project/Datasets/ISMN/Ver_1/processed_data'
# Path of GLDAS
path_gldas = '/Volumes/MyPassport/SMAP_Project/Datasets/GLDAS'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'cellsize_1km', 'cellsize_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_geo_10km', 'lon_world_geo_10km',
                'lat_world_ease_25km', 'lon_world_ease_25km', 'row_world_ease_9km_from_1km_ind', 'interdist_ease_1km',
                'col_world_ease_9km_from_1km_ind', 'size_world_ease_1km', 'row_conus_ease_1km_ind', 'col_conus_ease_1km_ind',
                'col_world_ease_1km_from_25km_ind', 'row_world_ease_1km_from_25km_ind']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Load in Australian variables
os.chdir(path_workspace)
f = h5py.File("aus_parameters.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()

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


# Generate land/water mask provided by GLDAS/NASA
os.chdir(path_lmask)
lmask_file = open('EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin', 'r')
lmask_ease_25km = np.fromfile(lmask_file, dtype=np.dtype('uint8'))
lmask_ease_25km = np.reshape(lmask_ease_25km, [len(lat_world_ease_25km), len(lon_world_ease_25km)]).astype(float)
lmask_ease_25km[np.where(lmask_ease_25km != 0)] = np.nan
lmask_ease_25km[np.where(lmask_ease_25km == 0)] = 1
# lmask_ease_25km[np.where((lmask_ease_25km == 101) | (lmask_ease_25km == 255))] = 0
lmask_file.close()

# Find the indices of land pixels by the 25-km resolution land-ocean mask
[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind] = np.where(~np.isnan(lmask_ease_25km))

# Convert the 1 km from 25 km match table files to 1-d linear
col_meshgrid_from_25km, row_meshgrid_from_25km = np.meshgrid(col_world_ease_1km_from_25km_ind, row_world_ease_1km_from_25km_ind)
col_meshgrid_from_25km = col_meshgrid_from_25km.reshape(1, -1)
row_meshgrid_from_25km = row_meshgrid_from_25km.reshape(1, -1)


########################################################################################################################
# 1. Extract the geographic information of the Australian soil data

src_tf = gdal.Open(path_aus_soil + '/australia_soil_data/CLY_000_005_EV_N_P_AU_NAT_C_20140801.tif')
src_tf_arr = src_tf.ReadAsArray().astype(np.float32)

cellsize_aussoil = src_tf.GetGeoTransform()[1]
size_aus = src_tf_arr.shape
lat_aus_max = src_tf.GetGeoTransform()[3] - cellsize_aussoil/2
lon_aus_min = src_tf.GetGeoTransform()[0] - cellsize_aussoil/2
lat_aus_min = lat_aus_max - cellsize_aussoil*(size_aus[0]-1)
lon_aus_max = lon_aus_min + cellsize_aussoil*(size_aus[1]-1)

lat_aus_90m = np.linspace(lat_aus_max, lat_aus_min, size_aus[0])
lon_aus_90m = np.linspace(lon_aus_min, lon_aus_max, size_aus[1])

# Subset the Australian region
[lat_aus_ease_1km, row_aus_ease_1km_ind, lon_aus_ease_1km, col_aus_ease_1km_ind] = coordtable_subset\
    (lat_world_ease_1km, lon_world_ease_1km, lat_aus_max, lat_aus_min, lon_aus_max, lon_aus_min)
[lat_aus_ease_25km, row_aus_ease_25km_ind, lon_aus_ease_25km, col_aus_ease_25km_ind] = coordtable_subset\
    (lat_world_ease_25km, lon_world_ease_25km, lat_aus_max, lat_aus_min, lon_aus_max, lon_aus_min)
[lat_aus_ease_9km, row_aus_ease_9km_ind, lon_aus_ease_9km, col_aus_ease_9km_ind] = coordtable_subset\
    (lat_world_ease_9km, lon_world_ease_9km, lat_aus_max, lat_aus_min, lon_aus_max, lon_aus_min)

# Save variables
os.chdir(path_workspace)
var_name = ['col_aus_ease_1km_ind', 'row_aus_ease_1km_ind', 'col_aus_ease_9km_ind', 'row_aus_ease_9km_ind',
            'lat_aus_90m', 'lon_aus_90m', 'lat_aus_ease_1km', 'lon_aus_ease_1km', 'lat_aus_ease_9km', 'lon_aus_ease_9km',
            'lat_aus_ease_25km', 'row_aus_ease_25km_ind', 'lon_aus_ease_25km', 'col_aus_ease_25km_ind']

with h5py.File('aus_parameters.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()


# Aggregate the Australian soil data from 90 m to 1 km
# Generate the aggregate table for 1 km from 90 m
[row_aus_ease_1km_from_90m_ind, col_aus_ease_1km_from_90m_ind] = \
    find_easeind_lofrhi(lat_aus_90m, lon_aus_90m, interdist_ease_1km,
                        size_world_ease_1km[0], size_world_ease_1km[1], row_aus_ease_1km_ind, col_aus_ease_1km_ind)

os.chdir(path_aus_soil + '/australia_soil_data')
aussoil_files = sorted(glob.glob('*.tif'))

aussoil_1day_init = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km)], dtype='float32')
aussoil_1day_init[:] = np.nan
aussoil_data = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), len(aussoil_files)], dtype='float32')
aussoil_data[:] = np.nan

for idt in range(len(aussoil_files)):
    aussoil_1day = np.copy(aussoil_1day_init)
    src_tf = gdal.Open(aussoil_files[idt])
    src_tf_arr = src_tf.ReadAsArray().astype(np.float32)
    aussoil_1day = np.array\
        ([np.nanmean(src_tf_arr[row_aus_ease_1km_from_90m_ind[x], :], axis=0)
          for x in range(len(lat_aus_ease_1km))])
    aussoil_1day = np.array \
        ([np.nanmean(aussoil_1day[:, col_aus_ease_1km_from_90m_ind[y]], axis=1)
          for y in range(len(lon_aus_ease_1km))])
    aussoil_1day = np.fliplr(np.rot90(aussoil_1day, 3))
    aussoil_1day[np.where(aussoil_1day <= 0)] = np.nan
    aussoil_data[:, :, idt] = aussoil_1day
    del(aussoil_1day, src_tf, src_tf_arr)
    print(idt)

# Create a raster of EASE grid projection at 1 km resolution
out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_aus_soil + '/aussoil_data.tif',
     len(lon_aus_ease_1km), len(lat_aus_ease_1km), len(aussoil_files),  # Number of bands
     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
# out_ds_tiff.SetGeoTransform(dst_tran)
# out_ds_tiff.SetProjection(src_tf.GetProjection())

# Loop write each band to Geotiff file
for idl in range(len(aussoil_files)):
    out_ds_tiff.GetRasterBand(idl + 1).WriteArray(aussoil_data[:, :, idl])
    out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
    out_ds_tiff.GetRasterBand(idl + 1).SetDescription(aussoil_files[idl].split('_')[0])
out_ds_tiff = None  # close dataset to write to disc

del(aussoil_data, src_tf_arr)


########################################################################################################################
# 2. Calculate the Soil water deficit index (SWDI) required input parameters

# Load in Australian variables
os.chdir(path_workspace)
f = h5py.File("aus_parameters.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()


aussoil_tf = gdal.Open(path_aus_soil + '/aussoil_data.tif')
aussoil_arr = aussoil_tf.ReadAsArray().astype(np.float32)

cly = aussoil_arr[0, :, :]/100  # Clay
snd = aussoil_arr[1, :, :]/100  # Sand
soc = aussoil_arr[2, :, :]/100  # Soil organic carbon

om = soc/0.58  # Convert from Soil organic carbon to soil organic matter


# 2.1 Calculate the parameters for calculating SWDI
theta_wp_fs = -0.024*snd + 0.487*cly + 0.006*om + 0.005*(snd*om) - 0.013*(cly*om) + 0.068*(snd*cly) + 0.031
theta_wp = theta_wp_fs + (0.14 * theta_wp_fs - 0.02)
theta_fc_fs = -0.251*snd + 0.195*cly + 0.011*om + 0.006*(snd*om) - 0.027*(cly*om) + 0.452*(snd*cly) + 0.299
theta_fc = theta_fc_fs + [1.283*(theta_fc_fs**2) - 0.374 * theta_fc_fs - 0.015]
theta_fc = np.squeeze(theta_fc)
theta_awc = theta_fc - theta_wp

del(aussoil_tf, aussoil_arr)

# Get geographic referencing information of Ausralia
smap_sm_aus_1km_data = gdal.Open('/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/1km/gldas/2019/smap_sm_1km_ds_2019001.tif')
smap_sm_aus_1km_data = smap_sm_aus_1km_data.ReadAsArray().astype(np.float32)[0, :, :]
output_crs = 'EPSG:6933'
sub_window_aus_1km = Window(col_aus_ease_1km_ind[0], row_aus_ease_1km_ind[0], len(col_aus_ease_1km_ind), len(row_aus_ease_1km_ind))
kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                  'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                  'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956,
                                      7314540.79258289)}
smap_sm_aus_1km_output = sub_n_reproj(smap_sm_aus_1km_data, kwargs_1km_sub, sub_window_aus_1km, output_crs)
kwargs = smap_sm_aus_1km_output.meta.copy()


# Load in SMAP 1 km SM
for iyr in range(len(yearname)):
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    for idt in range(len(tif_files)):
        sm_tf = gdal.Open(tif_files[idt])
        sm_arr = sm_tf.ReadAsArray().astype(np.float32)
        sm_arr = sm_arr[:, row_aus_ease_1km_ind[0]:row_aus_ease_1km_ind[-1]+1, col_aus_ease_1km_ind[0]:col_aus_ease_1km_ind[-1]+1]
        sm_arr = np.nanmean(sm_arr, axis=0)
        swdi = (sm_arr - theta_fc) / theta_awc * 10
        swdi = np.expand_dims(swdi, axis=0)

        name = 'aus_swdi_' + os.path.splitext(tif_files[idt])[0].split('_')[-1]
        with rasterio.open(path_swdi + '/' + str(yearname[iyr]) + '/' + name + '.tif', 'w', **kwargs) as dst_file:
            dst_file.write(swdi)
        print(name)

        del(sm_tf, sm_arr, name, swdi)



# 2.2 Generate the seasonal SWDI maps

# Extract the days for each season
seasons_div_norm = np.array([0, 90, 181, 273, 365])
seasons_div_leap = np.array([0, 91, 182, 274, 366])

ind_season_all_years = []
for iyr in range(len(yearname)):
    # os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    # tif_files = sorted(glob.glob('*.tif'))
    # names_daily = np.array([int(os.path.splitext(tif_files[idt])[0].split('_')[-1][-3:]) for idt in range(len(tif_files))])

    # Divide by seasons
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    names_daily = np.array([int(os.path.splitext(tif_files[idt])[0].split('_')[-1][-3:]) for idt in range(len(tif_files))])
    ind_season_all = []

    if iyr != 1:
        for i in range(len(seasons_div_norm)-1):
            ind_season = np.where((names_daily > seasons_div_norm[i]) & (names_daily <= seasons_div_norm[i+1]))[0]
            ind_season_all.append(ind_season)
            del (ind_season)
    else:
        for i in range(len(seasons_div_leap)-1):
            ind_season = np.where((names_daily > seasons_div_leap[i]) & (names_daily <= seasons_div_leap[i+1]))[0]
            ind_season_all.append(ind_season)
            del(ind_season)

    ind_season_all_years.append(ind_season_all)


# Average the SWDI data by seasons and map
swdi_arr_avg_all = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), 4], dtype='float32')
swdi_arr_avg_all[:] = np.nan
for iyr in [4]:#range(len(yearname)):
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    swdi_arr_avg_all = []
    for ise in range(len(ind_season_all_years[iyr])):
        season_list = ind_season_all_years[iyr][ise]

        swdi_arr_all = []
        for idt in range(len(season_list)):
            swdi_tf = gdal.Open(tif_files[season_list[idt]])
            swdi_arr = swdi_tf.ReadAsArray().astype(np.float32)
            swdi_arr_all.append(swdi_arr)
            print(tif_files[season_list[idt]])

        swdi_arr_all = np.array(swdi_arr_all)
        swdi_arr_avg = np.nanmean(swdi_arr_all, axis=0)

        swdi_arr_avg_all.append(swdi_arr_avg)
        del(swdi_arr_all, swdi_arr_avg, season_list)

swdi_arr_avg_all = np.array(swdi_arr_avg_all)


# 2.3 Make the seasonally averaged SWDI maps in Australia (2019)
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

xx_wrd, yy_wrd = np.meshgrid(lon_aus_ease_1km, lat_aus_ease_1km) # Create the map matrix
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
columns = 2
rows = 2
fig = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_1km[0], lon_aus_ease_1km[-1], lat_aus_ease_1km[-1], lat_aus_ease_1km[0]])
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, swdi_arr_avg_all[ipt, :, :], vmin=-30, vmax=30, cmap='coolwarm_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=25, fontsize=16, weight='bold')
    ax.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
            verticalalignment='top', weight='bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
fig.colorbar(img, cax=cbar_ax, extend='both')
plt.show()
plt.savefig(path_results + '/swdi_aus.png')


# 2.4 Make the seasonally averaged SWDI maps in Murray-Darling River basin (2019)
# Load in watershed shapefile boundaries
path_shp_md = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_MURRAY - DARLING'
shp_md_file = "Aqueduct_river_basins_MURRAY - DARLING.shp"
shapefile_md = fiona.open(path_shp_md + '/' + shp_md_file, 'r')
crop_shape_md = [feature["geometry"] for feature in shapefile_md]
shp_md_extent = list(shapefile_md.bounds)
output_crs = 'EPSG:4326'

#Subset the region of Murray-Darling
[lat_1km_md, row_md_1km_ind, lon_1km_md, col_md_1km_ind] = \
    coordtable_subset(lat_aus_ease_1km, lon_aus_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0])

[lat_1km_md, row_md_1km_world_ind, lon_1km_md, col_md_1km_world_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0]) # world

[lat_10km_md, row_md_10km_world_ind, lon_10km_md, col_md_10km_world_ind] = \
    coordtable_subset(lat_world_geo_10km, lon_world_geo_10km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0]) # GPM


swdi_arr_avg_all_md = swdi_arr_avg_all[:, row_md_1km_ind[0]:row_md_1km_ind[-1]+1, col_md_1km_ind[0]:col_md_1km_ind[-1]+1]


# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_md_1km_all = []
for n in range(swdi_arr_avg_all_md.shape[0]):
    sub_window_md_1km = Window(col_md_1km_ind[0], row_md_1km_ind[0], len(col_md_1km_ind), len(row_md_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_aus_ease_1km),
                      'height': len(lat_aus_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                       'transform': Affine(1000.89502334956, 0.0, 10902749.489346944, 0.0, -1000.89502334956, -1269134.927662937)}
    smap_sm_md_1km_output = sub_n_reproj(swdi_arr_avg_all_md[n, :, :], kwargs_1km_sub, sub_window_md_1km, output_crs)

    masked_ds_md_1km, mask_transform_ds_md_1km = mask(dataset=smap_sm_md_1km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_1km[np.where(masked_ds_md_1km == 0)] = np.nan
    masked_ds_md_1km = masked_ds_md_1km.squeeze()

    masked_ds_md_1km_all.append(masked_ds_md_1km)

masked_ds_md_1km_all = np.asarray(masked_ds_md_1km_all)


# Make the maps at watershed
feature_shp_md = ShapelyFeature(Reader(path_shp_md + '/' + shp_md_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_md = np.array(smap_sm_md_1km_output.bounds)
extent_md = extent_md[[0, 2, 1, 3]]

xx_wrd, yy_wrd = np.meshgrid(lon_1km_md, lat_1km_md) # Create the map matrix
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
columns = 2
rows = 2
fig = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_1km_md[0], lon_1km_md[-1], lat_1km_md[-1], lat_1km_md[0]])
    ax.add_feature(feature_shp_md)
    # img = ax.pcolormesh(xx_wrd, yy_wrd, masked_ds_md_1km_all[ipt, :, :], vmin=-30, vmax=30, cmap='coolwarm_r')
    img = ax.imshow(masked_ds_md_1km_all[ipt, :, :], origin='upper', vmin=-30, vmax=30, cmap='coolwarm_r',
               extent=extent_md)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=5)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=25, fontsize=16, weight='bold')
    ax.text(140.5, -25.5, title_content[ipt], fontsize=18, horizontalalignment='left', verticalalignment='top', weight = 'bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
fig.colorbar(img, cax=cbar_ax, extend='both')
plt.show()
plt.savefig(path_results + '/swdi_md.png')


del(swdi_arr_avg_all, masked_ds_md_1km, masked_ds_md_1km_all)


# 2.5.1 Generate the annual averaged SWDI maps
swdi_arr_allyear = []
for iyr in range(len(yearname)):
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    swdi_arr_1year = []
    for idt in range(len(tif_files)):
        swdi_tf = gdal.Open(tif_files[idt])
        swdi_arr = swdi_tf.ReadAsArray().astype(np.float32)
        swdi_arr_1year.append(swdi_arr)
        print(tif_files[idt])

    swdi_arr_1year = np.array(swdi_arr_1year)
    swdi_arr_avg = np.nanmean(swdi_arr_1year, axis=0)

    swdi_arr_allyear.append(swdi_arr_avg)
    del(swdi_arr_1year, swdi_arr_avg)

swdi_arr_allyear = np.array(swdi_arr_allyear)

smap_sm_aus_1km_read = rasterio.open(path_swdi + '/2019/aus_swdi_2019001.tif')
kwargs = smap_sm_aus_1km_read.meta.copy()
kwargs.update({'count': 5})
with rasterio.open(path_aus_soil + '/allyear_swdi.tif', 'w', **kwargs) as dst_file:
    dst_file.write(swdi_arr_allyear)


# 2.5.2 Generate the monthly averaged SWDI maps
swdi_arr_empty = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km)], dtype='float32')
swdi_arr_empty[:] = np.nan
smap_sm_aus_1km_read = rasterio.open(path_swdi + '/2019/aus_swdi_2019001.tif')

swdi_arr_allyear = []
for iyr in range(len(yearname)):
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    tif_files_month = np.asarray([(datetime.datetime(yearname[iyr], 1, 1) +
                                   datetime.timedelta(int(os.path.basename(tif_files[x]).split('.')[0][-3:]) - 1)).month - 1
                                  for x in range(len(tif_files))])
    tif_files_month_ind = [np.where(tif_files_month == x)[0] for x in range(12)]

    swdi_arr_1year = []
    for imo in range(len(tif_files_month_ind)):
        swdi_arr_1month = []
        if len(tif_files_month_ind[imo]) != 0:
            # tif_files_list_month = tif_files[tif_files_month_ind[imo]]
            for idt in range(len(tif_files_month_ind[imo])):
                swdi_tf = gdal.Open(tif_files[tif_files_month_ind[imo][idt]])
                swdi_arr = swdi_tf.ReadAsArray().astype(np.float32)
                swdi_arr_1month.append(swdi_arr)
                print(tif_files[tif_files_month_ind[imo][idt]])
            swdi_arr_1month = np.stack(swdi_arr_1month, axis=2)
            swdi_arr_1month = np.nanmean(swdi_arr_1month, axis=2)
        else:
            swdi_arr_1month = swdi_arr_empty

        swdi_arr_1year.append(swdi_arr_1month)
        del(swdi_arr_1month)

    swdi_arr_allyear.append(swdi_arr_1year)
    del(swdi_arr_1year)

swdi_arr_allyear = np.asarray(swdi_arr_allyear)
swdi_arr_allyear = np.reshape(swdi_arr_allyear, (60, len(lat_aus_ease_1km), len(lon_aus_ease_1km)))

kwargs = smap_sm_aus_1km_read.meta.copy()
kwargs.update({'count': 60})
with rasterio.open(path_aus_soil + '/allyear_swdi_monthly.tif', 'w', **kwargs) as dst_file:
    dst_file.write(swdi_arr_allyear)



# 2.6 Make the annually averaged SWDI maps in Australia

swdi_tf = gdal.Open('/Volumes/MyPassport/SMAP_Project/Datasets/Australia/allyear_swdi_monthly.tif')
swdi_monthly = swdi_tf.ReadAsArray().astype(np.float32)
swdi_arr_allyear = np.array([np.nanmean(swdi_monthly[x*12:x*12+12, :, :], axis=0) for x in range(5)])
# swdi_monthly = swdi_monthly[:, stn_row_1km_ind_all, stn_col_1km_ind_all]

shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

xx_wrd, yy_wrd = np.meshgrid(lon_aus_ease_1km, lat_aus_ease_1km) # Create the map matrix
title_content = ['2015', '2016', '2017', '2018', '2019']
columns = 2
rows = 3
fig = plt.figure(figsize=(10, 12), facecolor='w', edgecolor='k')
for ipt in range(len(yearname)):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_1km[0], lon_aus_ease_1km[-1], lat_aus_ease_1km[-1], lat_aus_ease_1km[0]])
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, swdi_arr_allyear[ipt, :, :], vmin=-30, vmax=30, cmap='coolwarm_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=9)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
            verticalalignment='top', weight='bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
fig.colorbar(img, cax=cbar_ax, extend='both')
plt.show()
plt.savefig(path_results + '/swdi_aus_allyear.png')


# 2.7 Make the annual averaged SWDI maps in Murray-Darling River basin
# Load in watershed shapefile boundaries
path_shp_md = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_MURRAY - DARLING'
shp_md_file = "Aqueduct_river_basins_MURRAY - DARLING.shp"
shapefile_md = fiona.open(path_shp_md + '/' + shp_md_file, 'r')
crop_shape_md = [feature["geometry"] for feature in shapefile_md]
shp_md_extent = list(shapefile_md.bounds)
output_crs = 'EPSG:4326'

#Subset the region of Murray-Darling
[lat_1km_md, row_md_1km_ind, lon_1km_md, col_md_1km_ind] = \
    coordtable_subset(lat_aus_ease_1km, lon_aus_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0])

[lat_1km_md, row_md_1km_world_ind, lon_1km_md, col_md_1km_world_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0]) # world

[lat_10km_md, row_md_10km_world_ind, lon_10km_md, col_md_10km_world_ind] = \
    coordtable_subset(lat_world_geo_10km, lon_world_geo_10km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0]) # GPM


swdi_arr_avg_all_md = swdi_arr_allyear[:, row_md_1km_ind[0]:row_md_1km_ind[-1]+1, col_md_1km_ind[0]:col_md_1km_ind[-1]+1]


# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_md_1km_all = []
for n in range(swdi_arr_avg_all_md.shape[0]):
    sub_window_md_1km = Window(col_md_1km_ind[0], row_md_1km_ind[0], len(col_md_1km_ind), len(row_md_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_aus_ease_1km),
                      'height': len(lat_aus_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                       'transform': Affine(1000.89502334956, 0.0, 10902749.489346944, 0.0, -1000.89502334956, -1269134.927662937)}
    smap_sm_md_1km_output = sub_n_reproj(swdi_arr_avg_all_md[n, :, :], kwargs_1km_sub, sub_window_md_1km, output_crs)

    masked_ds_md_1km, mask_transform_ds_md_1km = mask(dataset=smap_sm_md_1km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_1km[np.where(masked_ds_md_1km == 0)] = np.nan
    masked_ds_md_1km = masked_ds_md_1km.squeeze()

    masked_ds_md_1km_all.append(masked_ds_md_1km)

masked_ds_md_1km_all = np.asarray(masked_ds_md_1km_all)


# Make the maps at watershed
feature_shp_md = ShapelyFeature(Reader(path_shp_md + '/' + shp_md_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_md = np.array(smap_sm_md_1km_output.bounds)
extent_md = extent_md[[0, 2, 1, 3]]

xx_wrd, yy_wrd = np.meshgrid(lon_1km_md, lat_1km_md) # Create the map matrix
title_content = ['2015', '2016', '2017', '2018', '2019']
columns = 2
rows = 3
fig = plt.figure(figsize=(10, 12), facecolor='w', edgecolor='k')
for ipt in range(len(yearname)):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_1km_md[0], lon_1km_md[-1], lat_1km_md[-1], lat_1km_md[0]])
    ax.add_feature(feature_shp_md)
    # img = ax.pcolormesh(xx_wrd, yy_wrd, masked_ds_md_1km_all[ipt, :, :], vmin=-30, vmax=30, cmap='coolwarm_r')
    img = ax.imshow(masked_ds_md_1km_all[ipt, :, :], origin='upper', vmin=-30, vmax=30, cmap='coolwarm_r',
               extent=extent_md)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=9)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(140.5, -25.5, title_content[ipt], fontsize=18, horizontalalignment='left', verticalalignment='top',
            weight='bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
fig.colorbar(img, cax=cbar_ax, extend='both')
plt.show()
plt.savefig(path_results + '/swdi_md_allyear.png')

del(swdi_arr_avg_all, masked_ds_md_1km, masked_ds_md_1km_all)


########################################################################################################################
# 3. Calculate Soil Moisture Deficit Index (SMAI) using GLDAS data

row_aus_ind = np.where((row_lmask_ease_25km_ind >= row_aus_ease_25km_ind[0]) & (row_lmask_ease_25km_ind <= row_aus_ease_25km_ind[-1]))
col_aus_ind = np.where((col_lmask_ease_25km_ind >= col_aus_ease_25km_ind[0]) & (col_lmask_ease_25km_ind <= col_aus_ease_25km_ind[-1]))

lmask_ease_25km_aus_ind = np.intersect1d(row_aus_ind, col_aus_ind)

col_aus_ease_1km_ind_mat, row_aus_ease_1km_ind_mat = np.meshgrid(col_aus_ease_1km_ind, row_aus_ease_1km_ind)
aus_ease_1km_ind_mat = np.array([row_aus_ease_1km_ind_mat.flatten(), col_aus_ease_1km_ind_mat.flatten()])
aus_ease_1km_ind = np.ravel_multi_index(aus_ease_1km_ind_mat, (len(lat_world_ease_1km), len(lon_world_ease_1km)))


# 3.1 Extract the GLDAS grids in Australia and disaggregate to 1 km resolution,
# and calculate the maximun/minimum/median of each grid of each month

os.chdir(path_model + '/gldas')
ds_model_files = sorted(glob.glob('*ds_model_[0-9]*'))

sm_25km_init = np.empty([len(lat_world_ease_25km), len(lon_world_ease_25km)], dtype='float32')
sm_25km_init[:] = np.nan

sm_1km_init = np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km)], dtype='float32')
sm_1km_init = sm_1km_init.reshape(1, -1)
sm_1km_init[:] = np.nan

sm_1km_aus = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), 36], dtype='float32')
sm_1km_aus[:] = np.nan


for imo in range(len(monthnum)):

    fe_model = h5py.File(ds_model_files[imo], "r")
    varname_list_model = list(fe_model.keys())
    sm_gldas_am = fe_model[varname_list_model[3]][lmask_ease_25km_aus_ind, :]
    sm_gldas_pm = fe_model[varname_list_model[4]][lmask_ease_25km_aus_ind, :]
    sm_gldas = np.nanmean(np.stack((sm_gldas_am, sm_gldas_pm)), axis=0)
    fe_model.close()

    sm_gldas_max = np.nanmax(sm_gldas, axis=1)
    sm_gldas_min = np.nanmin(sm_gldas, axis=1)
    sm_gldas_median = np.nanmedian(sm_gldas, axis=1)
    sm_gldas_stats = np.stack((sm_gldas_max, sm_gldas_min, sm_gldas_median), axis=-1)

    del(fe_model, varname_list_model, sm_gldas_am, sm_gldas_pm, sm_gldas, sm_gldas_max, sm_gldas_min, sm_gldas_median)

    # Reshape to 2D 5 km matrix and disaggregate to 1 km
    for n in range(3):

        mat_output_25km = np.copy(sm_25km_init)
        mat_output_25km[row_lmask_ease_25km_ind[lmask_ease_25km_aus_ind], col_lmask_ease_25km_ind[lmask_ease_25km_aus_ind]] \
            = sm_gldas_stats[:, n]

        sm_1km_output = np.copy(sm_1km_init)
        sm_1km_output_1d = np.array([mat_output_25km[row_meshgrid_from_25km[0, aus_ease_1km_ind[x]],
                                                   col_meshgrid_from_25km[0, aus_ease_1km_ind[x]]]
                                   for x in range(len(aus_ease_1km_ind))])
        sm_1km_output[0, aus_ease_1km_ind] = sm_1km_output_1d
        sm_1km_output = sm_1km_output.reshape(len(lat_world_ease_1km), len(lon_world_ease_1km))
        sm_1km_output = sm_1km_output[row_aus_ease_1km_ind[0]:row_aus_ease_1km_ind[-1]+1,
                        col_aus_ease_1km_ind[0]:col_aus_ease_1km_ind[-1]+1]

        sm_1km_aus[:, :, imo*3+n] = sm_1km_output

        del(mat_output_25km, sm_1km_output, sm_1km_output_1d)
        print(imo*3+n)

    del(sm_gldas_stats)


# Save the raster of monthly stats
out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_aus_soil + '/aus_stats.tif',
     len(lon_aus_ease_1km), len(lat_aus_ease_1km), sm_1km_aus.shape[2],  # Number of bands
     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])

# Loop write each band to Geotiff file
for idl in range(sm_1km_aus.shape[2]):
    out_ds_tiff.GetRasterBand(idl + 1).WriteArray(sm_1km_aus[:, :, idl])
    out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
out_ds_tiff = None  # close dataset to write to disc
del(out_ds_tiff)


########################################################################################################################
# 3.2 Extract the GLDAS data between 2015-2018 in Australia and calculate the yearly average

daysofmonth_seq_yearly = np.sum(daysofmonth_seq[:, 0:4], axis=1)
daysofmonth_seq_slc = daysofmonth_seq[:, 0:4]
daysofmonth_seq_slc_cumsum = np.cumsum(daysofmonth_seq_slc, axis=1)

sm_gldas_split_all = []
for imo in range(len(monthnum)):
    fe_model = h5py.File(ds_model_files[imo], "r")
    varname_list_model = list(fe_model.keys())
    sm_gldas_am = fe_model[varname_list_model[3]][lmask_ease_25km_aus_ind, -daysofmonth_seq_yearly[imo]:]
    sm_gldas_pm = fe_model[varname_list_model[4]][lmask_ease_25km_aus_ind, -daysofmonth_seq_yearly[imo]:]
    sm_gldas = np.nanmean([sm_gldas_am, sm_gldas_pm], axis=0)
    sm_gldas_split = np.split(sm_gldas, daysofmonth_seq_slc_cumsum[imo], axis=1)[0:4]
    sm_gldas_split_all.append(sm_gldas_split)
    print(imo)
    del(sm_gldas_split)

# Remove the missing data corresponding to 1 km SMAP
sm_gldas_split_all[0][0][~np.isnan(sm_gldas_split_all[0][0])] = np.nan
sm_gldas_split_all[0][1][~np.isnan(sm_gldas_split_all[0][1])] = np.nan
sm_gldas_split_all[0][2][~np.isnan(sm_gldas_split_all[0][2])] = np.nan
sm_gldas_split_all[0][3][~np.isnan(sm_gldas_split_all[0][3])] = np.nan
sm_gldas_split_all[0][3][~np.isnan(sm_gldas_split_all[0][3])] = np.nan

# Generate 2d SM yearly data
sm_gldas_25km_all = []
for iyr in range(len(yearname)-1):
    sm_gldas_year = []
    for imo in range(len(monthnum)):
        sm_gldas_1month = sm_gldas_split_all[imo][iyr]
        sm_gldas_year.append(sm_gldas_1month)
        del(sm_gldas_1month)
    sm_gldas_year = np.concatenate(sm_gldas_year, axis=1)
    sm_gldas_year = np.nanmean(sm_gldas_year, axis=1)
    sm_gldas_25km = np.copy(sm_25km_init)
    sm_gldas_25km[row_lmask_ease_25km_ind[lmask_ease_25km_aus_ind], col_lmask_ease_25km_ind[lmask_ease_25km_aus_ind]] \
        = sm_gldas_year
    sm_gldas_25km = sm_gldas_25km[row_aus_ease_25km_ind[0]:row_aus_ease_25km_ind[-1] + 1,
                    col_aus_ease_25km_ind[0]:col_aus_ease_25km_ind[-1] + 1]
    sm_gldas_25km_all.append(sm_gldas_25km)
    del(sm_gldas_year, sm_gldas_25km)

sm_gldas_25km_all = np.stack(sm_gldas_25km_all, axis=2)

# Save the raster of yearly stats
out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_aus_soil + '/sm_gldas_25km_all.tif',
     len(lon_aus_ease_25km), len(lat_aus_ease_25km), sm_gldas_25km_all.shape[2],  # Number of bands
     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])

# Loop write each band to Geotiff file
for idl in range(sm_gldas_25km_all.shape[2]):
    out_ds_tiff.GetRasterBand(idl + 1).WriteArray(sm_gldas_25km_all[:, :, idl])
    out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
out_ds_tiff = None  # close dataset to write to disc
del(out_ds_tiff)


# Generate 2d SM monthly data
sm_gldas_25km_all_monthly = []
for iyr in range(len(yearname)-1):
    sm_gldas_year = []
    for imo in range(len(monthnum)):
        sm_gldas_1month = sm_gldas_split_all[imo][iyr]
        sm_gldas_1month = np.nanmean(sm_gldas_1month, axis=1)
        # sm_gldas_year = np.concatenate(sm_gldas_year, axis=1)
        # sm_gldas_year = np.nanmean(sm_gldas_year, axis=1)
        sm_gldas_25km = np.copy(sm_25km_init)
        sm_gldas_25km[row_lmask_ease_25km_ind[lmask_ease_25km_aus_ind], col_lmask_ease_25km_ind[lmask_ease_25km_aus_ind]] \
            = sm_gldas_1month
        sm_gldas_25km = sm_gldas_25km[row_aus_ease_25km_ind[0]:row_aus_ease_25km_ind[-1] + 1,
                        col_aus_ease_25km_ind[0]:col_aus_ease_25km_ind[-1] + 1]

        sm_gldas_year.append(sm_gldas_25km)
        del(sm_gldas_25km, sm_gldas_1month)

    sm_gldas_25km_all_monthly.append(sm_gldas_year)
    del(sm_gldas_year)

sm_gldas_25km_all_monthly = np.stack(sm_gldas_25km_all_monthly, axis=2)
sm_gldas_25km_all_monthly = np.transpose(sm_gldas_25km_all_monthly, (2, 0, 1, 3))
sm_shape = sm_gldas_25km_all_monthly.shape
sm_gldas_25km_all_monthly_trans = np.reshape(sm_gldas_25km_all_monthly, [sm_shape[0]*sm_shape[1], sm_shape[2], sm_shape[3]])
sm_gldas_25km_all_monthly_trans = np.transpose(sm_gldas_25km_all_monthly_trans, (1, 2, 0))

# Save the raster of monthly stats
out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_aus_soil + '/sm_gldas_25km_all_monthly.tif',
     len(lon_aus_ease_25km), len(lat_aus_ease_25km), sm_gldas_25km_all_monthly_trans.shape[2],  # Number of bands
     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])

# Loop write each band to Geotiff file
for idl in range(sm_gldas_25km_all_monthly_trans.shape[2]):
    out_ds_tiff.GetRasterBand(idl + 1).WriteArray(sm_gldas_25km_all_monthly_trans[:, :, idl])
    out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
out_ds_tiff = None  # close dataset to write to disc
del(out_ds_tiff)




# 3.3.1 Load in SMAP 1 km SM
for iyr in range(len(yearname)):
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    for idt in range(len(tif_files)):
        sm_tf = gdal.Open(tif_files[idt])
        sm_arr = sm_tf.ReadAsArray().astype(np.float32)
        sm_arr = sm_arr[:, row_aus_ease_1km_ind[0]:row_aus_ease_1km_ind[-1]+1, col_aus_ease_1km_ind[0]:col_aus_ease_1km_ind[-1]+1]
        sm_arr = np.nanmean(sm_arr, axis=0)
        sm_arr = np.expand_dims(sm_arr, axis=0)

        name = 'aus_smap_1km_' + os.path.splitext(tif_files[idt])[0].split('_')[-1]
        with rasterio.open(path_processed_2 + '/smap_1km/' + name + '.tif', 'w', **kwargs) as dst_file:
            dst_file.write(sm_arr)
        print(name)

        del(sm_tf, sm_arr, name)

# 3.3.2 Calculate yearly averaged SMAP 1 km SM

sm_1km_init = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km)], dtype='float32')
# sm_1km_init = sm_1km_init.reshape(1, -1)
sm_1km_init[:] = np.nan

tif_files = sorted(glob.glob(path_processed_2 + '/smap_1km/' + '*.tif'))
name_split = [int(tif_files[x].split('.')[0][-7:-3]) for x in range(len(tif_files))]
name_split = np.array(name_split)
name_group = [np.where(name_split == yearname[x]) for x in range(len(yearname))]

sm_arr_all = []
for iyr in range(len(yearname)):

    tif_files_1year = name_group[iyr][0]
    sm_arr = sm_1km_init
    for idt in range(len(tif_files_1year)):
        sm_tf = gdal.Open(tif_files[tif_files_1year[idt]])
        sm_arr_new = sm_tf.ReadAsArray().astype(np.float32)
        sm_arr = np.nanmean(np.stack((sm_arr, sm_arr_new), axis=2), axis=2)
        print(tif_files[tif_files_1year[idt]])
        del(sm_tf, sm_arr_new)

    sm_arr_all.append(sm_arr)
    del(sm_arr, tif_files_1year)

sm_arr_all = np.stack(sm_arr_all, axis=0)

# Save the raster of monthly stats
out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_aus_soil + '/sm_smap_1km_all.tif',
     len(lon_aus_ease_1km), len(lat_aus_ease_1km), sm_arr_all.shape[0],  # Number of bands
     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])

# Loop write each band to Geotiff file
for idl in range(sm_arr_all.shape[0]):
    out_ds_tiff.GetRasterBand(idl + 1).WriteArray(sm_arr_all[idl, :, :])
    out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
out_ds_tiff = None  # close dataset to write to disc
del(out_ds_tiff)


########################################################################################################################
# 3.4 Calculate the SMDI by each month

# Load in SMAP 1 km SM and calculate monthly average
for iyr in range(len(yearname)):
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Divide the tif files by month
    tif_files_delta = [int(tif_files[x].split('.')[0][-3:]) - 1 for x in range(len(tif_files))]
    tif_files_month = [(datetime.date(yearname[iyr], 1, 1) + datetime.timedelta(tif_files_delta[x])).month
                       for x in range(len(tif_files_delta))]
    tif_files_month = np.array(tif_files_month)

    sm_arr_month_avg_all = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), len(monthname)], dtype='float32')
    sm_arr_month_avg_all[:] = np.nan

    for imo in range(len(monthname)):

        month_ind = list(np.where(tif_files_month == imo+1))[0]

        sm_arr_month = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), len(month_ind)], dtype='float32')
        sm_arr_month[:] = np.nan
        for idt in range(len(month_ind)):
            sm_tf = gdal.Open(tif_files[month_ind[idt]])
            sm_arr = sm_tf.ReadAsArray().astype(np.float32)
            sm_arr = sm_arr[:, row_aus_ease_1km_ind[0]:row_aus_ease_1km_ind[-1]+1,
                     col_aus_ease_1km_ind[0]:col_aus_ease_1km_ind[-1]+1]
            sm_arr = np.nanmean(sm_arr, axis=0)
            sm_arr_month[:, :, idt] = sm_arr
            print(tif_files[month_ind[idt]])
            del(sm_arr)

        sm_arr_month_avg = np.nanmean(sm_arr_month, axis=2)
        sm_arr_month_avg_all[:, :, imo] = sm_arr_month_avg
        del(sm_arr_month, month_ind, sm_arr_month_avg)


    # Save the raster of monthly SM of one year
    out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_aus_soil + '/smap_monthly_sm_' + str(yearname[iyr]) + '.tif',
         len(lon_aus_ease_1km), len(lat_aus_ease_1km), len(monthname),  # Number of bands
         gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])

    # Loop write each band to Geotiff file
    for idl in range(sm_arr_month_avg_all.shape[2]):
        out_ds_tiff.GetRasterBand(idl + 1).WriteArray(sm_arr_month_avg_all[:, :, idl])
        out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
    out_ds_tiff = None  # close dataset to write to disc
    del(out_ds_tiff)



# Load in the SMAP monthly average data and Australian statistical data

aus_stats = gdal.Open(path_aus_soil + '/aus_stats.tif')
aus_stats = aus_stats.ReadAsArray().astype(np.float32)
sm_ind_max = np.arange(0, 36, 3)
sm_ind_min = np.arange(1, 36, 3)
sm_ind_median = np.arange(2, 36, 3)
aus_sm_max = aus_stats[sm_ind_max, :, :]
aus_sm_min = aus_stats[sm_ind_min, :, :]
aus_sm_median = aus_stats[sm_ind_median, :, :]

smai_allyear = []
for iyr in range(len(yearname)):
    aus_smap_sm_monthly = gdal.Open(path_aus_soil + '/smap_monthly_sm_' + str(yearname[iyr]) + '.tif')
    aus_smap_sm_monthly = aus_smap_sm_monthly.ReadAsArray().astype(np.float32)

    aus_sm_delta = aus_smap_sm_monthly - aus_sm_median
    aus_sm_med_min = aus_sm_median - aus_sm_min
    aus_sm_max_med = aus_sm_max - aus_sm_median

    # Calculate Soil water deficit
    sm_defi_mat_init = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km)], dtype='float32')
    sm_defi_mat_init[:] = np.nan
    sm_defi_mat_all = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), len(monthname)], dtype='float32')
    sm_defi_mat_all[:] = np.nan
    for imo in range(len(monthname)):
        sm_defi = np.copy(sm_defi_mat_init)
        sm_defi_ind = np.where(aus_sm_delta[imo, :, :] <= 0)
        sm_defi_ind_opp = np.where(aus_sm_delta[imo, :, :] > 0)
        sm_defi[sm_defi_ind[0], sm_defi_ind[1]] = aus_sm_delta[imo, sm_defi_ind[0], sm_defi_ind[1]] / \
                                                  aus_sm_med_min[imo, sm_defi_ind[0], sm_defi_ind[1]] * 100
        sm_defi[sm_defi_ind_opp[0], sm_defi_ind_opp[1]] = aus_sm_delta[imo, sm_defi_ind_opp[0], sm_defi_ind_opp[1]] / \
                                                  aus_sm_max_med[imo, sm_defi_ind_opp[0], sm_defi_ind_opp[1]] * 100
        sm_defi_mat_all[:, :, imo] = sm_defi
        print(imo)
        del(sm_defi_ind, sm_defi_ind_opp, sm_defi)

    # sm_defi_mat_all[:, :, 6] = sm_defi_mat_all[:, :, 5]

    # Find the first month which is not empty
    sm_defi_mat_all_avg = np.nanmean(sm_defi_mat_all, axis=0)
    sm_defi_mat_all_avg = np.nanmean(sm_defi_mat_all_avg, axis=0)
    sm_defi_mat_all_avg_notnan = np.where(~np.isnan(sm_defi_mat_all_avg))[0][0].item()
    month_diff = np.setdiff1d(np.arange(12), np.where(~np.isnan(sm_defi_mat_all_avg))[0])

    # Calculate the monthly SMAI
    smai_all = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), len(monthname)], dtype='float32')
    smai_all[:] = np.nan
    if sm_defi_mat_all_avg_notnan == 0:
        smai_all[:, :, 0] = sm_defi_mat_all[:, :, 0]/50 # SMAI for the first month
        if len(month_diff)!=0:
            sm_defi_mat_all[:, :, month_diff[0]] = sm_defi_mat_all[:, :, month_diff[0]-1]
        else:
            pass
        for imo in range(1, len(monthname)):
            smai_all[:, :, imo] = 0.5 * smai_all[:, :, imo-1] + sm_defi_mat_all[:, :, imo]/50

    else:
        smai_all[:, :, sm_defi_mat_all_avg_notnan] = sm_defi_mat_all[:, :, sm_defi_mat_all_avg_notnan] / 50  # SMAI for the first month
        for imo in range(sm_defi_mat_all_avg_notnan + 1, len(monthname)):
            smai_all[:, :, imo] = 0.5 * smai_all[:, :, imo - 1] + sm_defi_mat_all[:, :, imo] / 50
        else:
            pass


    smai_allyear.append(smai_all)
    print(yearname[iyr])
    del(aus_smap_sm_monthly, aus_sm_delta, aus_sm_med_min, aus_sm_max_med, sm_defi_mat_all, sm_defi_mat_all_avg,
        sm_defi_mat_all_avg_notnan, smai_all)

smai_allyear = np.asarray(smai_allyear)
smai_allyear = np.transpose(smai_allyear, (0, 3, 1, 2))
smai_allyear = np.reshape(smai_allyear, (60, len(lat_aus_ease_1km), len(lon_aus_ease_1km)))

smap_sm_aus_1km_read = rasterio.open(path_swdi + '/2019/aus_swdi_2019001.tif')
kwargs = smap_sm_aus_1km_read.meta.copy()
kwargs.update({'count': 60})
with rasterio.open(path_aus_soil + '/allyear_smai_monthly.tif', 'w', **kwargs) as dst_file:
    dst_file.write(smai_allyear)


# Normalize to the range of [-4, 4]
# smai_all_norm = smai_all * (4 / np.nanmax(np.abs(smai_all)))


########################################################################################################################
# 3.5 Make the seasonally averaged SMAI maps in Australia (2019)

smai_tf = gdal.Open('/Volumes/MyPassport/SMAP_Project/Datasets/Australia/allyear_smai_monthly.tif')
smai_monthly = smai_tf.ReadAsArray().astype(np.float32)
# smai_monthly = smai_monthly[:, stn_row_1km_ind_all, stn_col_1km_ind_all]
smai_all = smai_monthly[48:, :, :]
smai_all = np.transpose(smai_all, (1, 2, 0))

shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

xx_wrd, yy_wrd = np.meshgrid(lon_aus_ease_1km, lat_aus_ease_1km) # Create the map matrix
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
columns = 2
rows = 2
fig = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_1km[0], lon_aus_ease_1km[-1], lat_aus_ease_1km[-1], lat_aus_ease_1km[0]])
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, np.nanmean(smai_all[:, :, ipt*3:ipt*3+2], axis=2), vmin=-4, vmax=4, cmap='coolwarm_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.locator_params(nbins=4)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=25, fontsize=16, weight='bold')
    ax.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
            verticalalignment='top', weight='bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar_ax.locator_params(nbins=4)
plt.show()
plt.savefig(path_results + '/smai_aus.png')



# 3.6 Make the annually averaged SMAI maps in Australia
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

xx_wrd, yy_wrd = np.meshgrid(lon_aus_ease_1km, lat_aus_ease_1km) # Create the map matrix
title_content = ['2015', '2016', '2017', '2018', '2019']
columns = 2
rows = 3
fig = plt.figure(figsize=(10, 12), facecolor='w', edgecolor='k')
for ipt in range(len(smai_monthly)//12):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_1km[0], lon_aus_ease_1km[-1], lat_aus_ease_1km[-1], lat_aus_ease_1km[0]])
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, np.nanmean(smai_monthly[ipt*12:ipt*12+11, :, :], axis=0), vmin=-4, vmax=4, cmap='coolwarm_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.locator_params(nbins=4)
    # cbar.ax.tick_params(labelsize=9)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
            verticalalignment='top', weight='bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.ax.locator_params(nbins=4)
plt.show()
plt.savefig(path_results + '/smai_aus_allyear.png')



# 3.7 Make the seasonally averaged SMAI maps in Murray-Darling River basin (2019)

# Load in watershed shapefile boundaries
path_shp_md = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_MURRAY - DARLING'
shp_md_file = "Aqueduct_river_basins_MURRAY - DARLING.shp"
shapefile_md = fiona.open(path_shp_md + '/' + shp_md_file, 'r')
crop_shape_md = [feature["geometry"] for feature in shapefile_md]
shp_md_extent = list(shapefile_md.bounds)
output_crs = 'EPSG:4326'

#Subset the region of Murray-Darling
[lat_1km_md, row_md_1km_ind, lon_1km_md, col_md_1km_ind] = \
    coordtable_subset(lat_aus_ease_1km, lon_aus_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0])

smai_all_md = smai_all[row_md_1km_ind[0]:row_md_1km_ind[-1]+1, col_md_1km_ind[0]:col_md_1km_ind[-1]+1, :]

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_md_1km_all = []
for n in range(smai_all_md.shape[2]):
    sub_window_md_1km = Window(col_md_1km_ind[0], row_md_1km_ind[0], len(col_md_1km_ind), len(row_md_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_aus_ease_1km),
                      'height': len(lat_aus_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                       'transform': Affine(1000.89502334956, 0.0, 10902749.489346944, 0.0, -1000.89502334956, -1269134.927662937)}
    smap_sm_md_1km_output = sub_n_reproj(smai_all_md[:, :, n], kwargs_1km_sub, sub_window_md_1km, output_crs)

    masked_ds_md_1km, mask_transform_ds_md_1km = mask(dataset=smap_sm_md_1km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_1km[np.where(masked_ds_md_1km == 0)] = np.nan
    masked_ds_md_1km = masked_ds_md_1km.squeeze()

    masked_ds_md_1km_all.append(masked_ds_md_1km)

masked_ds_md_1km_all = np.asarray(masked_ds_md_1km_all)


# Make the maps at watershed
feature_shp_md = ShapelyFeature(Reader(path_shp_md + '/' + shp_md_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_md = np.array(smap_sm_md_1km_output.bounds)
extent_md = extent_md[[0, 2, 1, 3]]

# xx_wrd, yy_wrd = np.meshgrid(lon_1km_md, lat_1km_md) # Create the map matrix
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
columns = 2
rows = 2
fig = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_1km_md[0], lon_1km_md[-1], lat_1km_md[-1], lat_1km_md[0]])
    ax.add_feature(feature_shp_md)
    img = ax.imshow(np.nanmean(masked_ds_md_1km_all[ipt*3:ipt*3+2, :, :], axis=0), origin='upper', vmin=-4, vmax=4, cmap='coolwarm_r',
               extent=extent_md)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=5)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=25, fontsize=16, weight='bold')
    ax.text(140.5, -25.5, title_content[ipt], fontsize=18, horizontalalignment='left',
            verticalalignment='top', weight='bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.ax.locator_params(nbins=4)
plt.show()
plt.savefig(path_results + '/smai_md.png')


# 3.8 Make the annually averaged SMAI maps in Murray-Darling River basin

# Load in watershed shapefile boundaries
path_shp_md = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_MURRAY - DARLING'
shp_md_file = "Aqueduct_river_basins_MURRAY - DARLING.shp"
shapefile_md = fiona.open(path_shp_md + '/' + shp_md_file, 'r')
crop_shape_md = [feature["geometry"] for feature in shapefile_md]
shp_md_extent = list(shapefile_md.bounds)
output_crs = 'EPSG:4326'

#Subset the region of Murray-Darling using Australia lat/lon
[lat_1km_md, row_md_1km_ind, lon_1km_md, col_md_1km_ind] = \
    coordtable_subset(lat_aus_ease_1km, lon_aus_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0])

smai_monthly_avg = np.array([np.nanmean(smai_monthly[x*12:x*12+11, row_md_1km_ind[0]:row_md_1km_ind[-1]+1,
                                        col_md_1km_ind[0]:col_md_1km_ind[-1]+1], axis=0) for x in range(len(smai_monthly)//12)])
# smai_monthly_avg_md = smai_monthly_avg[row_md_1km_ind[0]:row_md_1km_ind[-1]+1, col_md_1km_ind[0]:col_md_1km_ind[-1]+1, :]

#Subset the region of Murray-Darling using World lat/lon
[lat_1km_md_wrd, row_md_wrd_1km_ind, lon_1km_md_wrd, col_md_wrd_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0])

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_md_1km_all = []
for n in range(smai_monthly_avg.shape[0]):
    sub_window_md_1km = Window(col_md_1km_ind[0], row_md_1km_ind[0], len(col_md_1km_ind), len(row_md_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_aus_ease_1km),
                      'height': len(lat_aus_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                       'transform': Affine(1000.89502334956, 0.0, 10902749.489346944, 0.0, -1000.89502334956, -1269134.927662937)}
    smap_sm_md_1km_output = sub_n_reproj(smai_monthly_avg[n, :, :], kwargs_1km_sub, sub_window_md_1km, output_crs)

    masked_ds_md_1km, mask_transform_ds_md_1km = mask(dataset=smap_sm_md_1km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_1km[np.where(masked_ds_md_1km == 0)] = np.nan
    masked_ds_md_1km = masked_ds_md_1km.squeeze()

    masked_ds_md_1km_all.append(masked_ds_md_1km)

masked_ds_md_1km_all = np.asarray(masked_ds_md_1km_all)


# Make the maps at watershed
feature_shp_md = ShapelyFeature(Reader(path_shp_md + '/' + shp_md_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_md = np.array(smap_sm_md_1km_output.bounds)
extent_md = extent_md[[0, 2, 1, 3]]

# xx_wrd, yy_wrd = np.meshgrid(lon_1km_md, lat_1km_md) # Create the map matrix
title_content = ['2015', '2016', '2017', '2018', '2019']
columns = 2
rows = 3
fig = plt.figure(figsize=(10, 12), facecolor='w', edgecolor='k')
for ipt in range(len(yearname)):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_1km_md[0], lon_1km_md[-1], lat_1km_md[-1], lat_1km_md[0]])
    ax.add_feature(feature_shp_md)
    img = ax.imshow(masked_ds_md_1km_all[ipt, :, :], origin='upper', vmin=-4, vmax=4, cmap='coolwarm_r',
               extent=extent_md)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=9)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(140.5, -25.5, title_content[ipt], fontsize=18, horizontalalignment='left', verticalalignment='top', weight='bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.ax.locator_params(nbins=4)
plt.show()
plt.savefig(path_results + '/smai_allyear_md.png')


# 3.9 Make the annually averaged SMAI maps in Australia

sm_smap_1km_all_read = rasterio.open(path_aus_soil + '/sm_smap_1km_all.tif').read()
sm_gldas_25km_all_read = rasterio.open(path_aus_soil + '/sm_gldas_25km_all.tif').read()
f = h5py.File(path_gldas + '/ds_gldas_ease_2019.hdf5', "r")
varname_list = ['sm_gldas_am_ease_2019', 'sm_gldas_pm_ease_2019']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Concatenate GLDAS 2019 data
sm_gldas_am_ease_2019_aus = sm_gldas_am_ease_2019[row_aus_ease_25km_ind[0]:row_aus_ease_25km_ind[-1] + 1,
                    col_aus_ease_25km_ind[0]:col_aus_ease_25km_ind[-1] + 1, :]
sm_gldas_am_ease_2019_aus = np.nanmean(sm_gldas_am_ease_2019_aus, axis=2)
sm_gldas_pm_ease_2019_aus = sm_gldas_pm_ease_2019[row_aus_ease_25km_ind[0]:row_aus_ease_25km_ind[-1] + 1,
                    col_aus_ease_25km_ind[0]:col_aus_ease_25km_ind[-1] + 1, :]
sm_gldas_pm_ease_2019_aus = np.nanmean(sm_gldas_pm_ease_2019_aus, axis=2)
sm_gldas_ease_2019_aus = np.nanmean(np.stack((sm_gldas_am_ease_2019_aus, sm_gldas_pm_ease_2019_aus), axis=2), axis=2)
sm_gldas_ease_2019_aus = np.expand_dims(sm_gldas_ease_2019_aus, axis=0)

sm_gldas_25km_all_read = np.concatenate((sm_gldas_25km_all_read, sm_gldas_ease_2019_aus), axis=0)

# Aggregate the SMAP data from 25 km to 1 km
interdist_ease_25km = 25067.525
size_world_ease_25km = np.array([584, 1388])
[row_aus_ease_25km_from_1km_ind, col_aus_ease_25km_from_1km_ind] = \
    find_easeind_lofrhi(lat_aus_ease_1km, lon_aus_ease_1km, interdist_ease_25km,
                        size_world_ease_25km[0], size_world_ease_25km[1], row_aus_ease_25km_ind, col_aus_ease_25km_ind)

sm_smap_1km_all_read_agg = []
for ily in range(5):
    sm_smap_1km_1day = sm_smap_1km_all_read[ily, :, :]
    sm_smap_1km_1day_agg = np.array \
        ([np.nanmean(sm_smap_1km_1day[row_aus_ease_25km_from_1km_ind[x], :], axis=0)
          for x in range(len(lat_aus_ease_25km))])
    sm_smap_1km_1day_agg = np.array \
        ([np.nanmean(sm_smap_1km_1day_agg[:, col_aus_ease_25km_from_1km_ind[y]], axis=1)
          for y in range(len(lon_aus_ease_25km))])
    sm_smap_1km_1day_agg = np.fliplr(np.rot90(sm_smap_1km_1day_agg, 3))
    sm_smap_1km_1day_agg[np.where(sm_smap_1km_1day_agg <= 0)] = np.nan

    sm_smap_1km_all_read_agg.append(sm_smap_1km_1day_agg)

sm_smap_1km_all_read_agg = np.stack(sm_smap_1km_all_read_agg)

sm_smap_delta = sm_smap_1km_all_read_agg - sm_gldas_25km_all_read
sm_smap_rel_bias = (sm_smap_1km_all_read_agg - sm_gldas_25km_all_read)/sm_gldas_25km_all_read

shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

# Map 1 (with aggregation)
xx_wrd, yy_wrd = np.meshgrid(lon_aus_ease_25km, lat_aus_ease_25km) # Create the map matrix
title_content = ['2015', '2016', '2017', '2018', '2019']
columns = 2
rows = 5
fig = plt.figure(figsize=(10, 15), facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.1, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.2)
for ipt in range(5):
    ax1 = fig.add_subplot(rows, columns, ipt*2+1, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_25km[0], lon_aus_ease_25km[-1], lat_aus_ease_25km[-1], lat_aus_ease_25km[0]])
    ax1.add_feature(shape_world, linewidth=0.5)
    img1 = ax1.pcolormesh(xx_wrd, yy_wrd, sm_smap_1km_all_read_agg[ipt, :, :], vmin=0, vmax=0.4, cmap='coolwarm_r')
    gl = ax1.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # ax1.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
    #         verticalalignment='top', weight='bold')

    ax2 = fig.add_subplot(rows, columns, ipt*2+2, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_25km[0], lon_aus_ease_25km[-1], lat_aus_ease_25km[-1], lat_aus_ease_25km[0]])
    ax2.add_feature(shape_world, linewidth=0.5)
    img2 = ax2.pcolormesh(xx_wrd, yy_wrd, sm_gldas_25km_all_read[ipt, :, :], vmin=0, vmax=0.4, cmap='coolwarm_r')
    gl = ax2.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # ax2.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
    #         verticalalignment='top', weight='bold')

fig.text(0.05, 0.85, '2015', ha='center', fontsize=16, fontweight='bold')
fig.text(0.05, 0.67, '2016', ha='center', fontsize=16, fontweight='bold')
fig.text(0.05, 0.49, '2017', ha='center', fontsize=16, fontweight='bold')
fig.text(0.05, 0.3, '2018', ha='center', fontsize=16, fontweight='bold')
fig.text(0.05, 0.12, '2019', ha='center', fontsize=16, fontweight='bold')
fig.text(0.28, 0.95, 'SMAP', ha='center', fontsize=18, fontweight='bold')
fig.text(0.71, 0.95, 'GLDAS', ha='center', fontsize=18, fontweight='bold')
cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
cbar = fig.colorbar(img1, cax=cbar_ax, extend='both')
cbar.ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=2)
# plt.show()
plt.savefig(path_results + '/sm_compare_1.png')
plt.close()


# Map 2
xx_wrd_1km, yy_wrd_1km = np.meshgrid(lon_aus_ease_1km, lat_aus_ease_1km) # Create the map matrix
xx_wrd_25km, yy_wrd_25km = np.meshgrid(lon_aus_ease_25km, lat_aus_ease_25km) # Create the map matrix

title_content = ['2015', '2016', '2017', '2018', '2019']
columns = 3
rows = 5
fig = plt.figure(figsize=(12, 15), facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.1, right=0.93, bottom=0.08, top=0.92, hspace=0.25, wspace=0.2)
for ipt in range(5):
    ax1 = fig.add_subplot(rows, columns, ipt*3+1, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_1km[0], lon_aus_ease_1km[-1], lat_aus_ease_1km[-1], lat_aus_ease_1km[0]])
    ax1.add_feature(shape_world, linewidth=0.5)
    img1 = ax1.pcolormesh(xx_wrd_1km, yy_wrd_1km, sm_smap_1km_all_read[ipt, :, :], vmin=0, vmax=0.4, cmap='coolwarm_r')
    gl = ax1.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # ax1.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
    #         verticalalignment='top', weight='bold')

    ax2 = fig.add_subplot(rows, columns, ipt*3+2, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_25km[0], lon_aus_ease_25km[-1], lat_aus_ease_25km[-1], lat_aus_ease_25km[0]])
    ax2.add_feature(shape_world, linewidth=0.5)
    img2 = ax2.pcolormesh(xx_wrd_25km, yy_wrd_25km, sm_gldas_25km_all_read[ipt, :, :], vmin=0, vmax=0.4, cmap='coolwarm_r')
    gl = ax2.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # ax2.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
    #         verticalalignment='top', weight='bold')

    ax3 = fig.add_subplot(rows, columns, ipt*3+3, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_25km[0], lon_aus_ease_25km[-1], lat_aus_ease_25km[-1], lat_aus_ease_25km[0]])
    ax3.add_feature(shape_world, linewidth=0.5)
    img3 = ax3.pcolormesh(xx_wrd_25km, yy_wrd_25km, sm_smap_rel_bias[ipt, :, :], vmin=-1, vmax=1, cmap='bwr_r')
    gl = ax3.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # ax2.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
    #         verticalalignment='top', weight='bold')

fig.text(0.05, 0.85, '2015', ha='center', fontsize=16, fontweight='bold')
fig.text(0.05, 0.67, '2016', ha='center', fontsize=16, fontweight='bold')
fig.text(0.05, 0.5, '2017', ha='center', fontsize=16, fontweight='bold')
fig.text(0.05, 0.32, '2018', ha='center', fontsize=16, fontweight='bold')
fig.text(0.05, 0.15, '2019', ha='center', fontsize=16, fontweight='bold')
fig.text(0.23, 0.95, 'SMAP', ha='center', fontsize=18, fontweight='bold')
fig.text(0.52, 0.95, 'GLDAS', ha='center', fontsize=18, fontweight='bold')
fig.text(0.81, 0.95, 'Relative Bias', ha='center', fontsize=18, fontweight='bold')
cbar_ax1 = fig.add_axes([0.13, 0.04, 0.2, 0.01])
cbar1 = fig.colorbar(img1, cax=cbar_ax1, extend='both', orientation='horizontal')
cbar1.ax.locator_params(nbins=5)
cbar1.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=2)
cbar_ax2 = fig.add_axes([0.42, 0.04, 0.2, 0.01])
cbar2 = fig.colorbar(img2, cax=cbar_ax2, extend='both', orientation='horizontal')
cbar2.ax.locator_params(nbins=5)
cbar2.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=2)
cbar_ax3 = fig.add_axes([0.71, 0.04, 0.2, 0.01])
cbar3 = fig.colorbar(img3, cax=cbar_ax3, extend='both', orientation='horizontal')
# cbar3.ax.locator_params(nbins=5)
cbar3.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar3.set_ticklabels(['-100', '-50', '0', '50', '100'])
cbar3.set_label('(%)', fontsize=10, x=0.95, labelpad=2)
plt.show()
plt.savefig(path_results + '/sm_compare_1_new.png', dpi=300)
plt.close()


########################################################################################################################
# 4. Make time series plots
# Weany Creek: -19.8820,146.5360, tropical savanna
# Gnangara: -31.3767,115.7132, Mediterranean
# Temora: -34.4046,147.5326, semi-arid
# Tumbarumba: -35.6560,148.1520, temperate maritime climate
# Tullochgorum: -41.6694,147.9117,  temperate climate
# Silver Springs: -35.27202, 147.42902, semi-arid
# Samarra: -35.2275, 147.4850, semi-arid
# Evergreen: -35.23887, 147.53330, semi-arid

# (Cheverelis: -35.00535, 146.30988, semi-arid)
# (Daly: -14.1592,131.3881, tropical)
# (Robsons Creek: -17.1200,145.6300, tropical)
# (Yanco: -35.0050,146.2992, semi-arid)

stn_lat_all = [-19.8820, -31.3497, -34.4046, -35.6560, -41.6694, -35.27202, -35.2275, -35.23887]
stn_lon_all = [146.5360, 115.9068, 147.5326, 148.1520, 147.9117, 147.42902, 147.4850, 147.53330]

# 4.1 Locate positions of in-situ stations using lat/lon tables of Australia

stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_aus_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_aus_ease_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    del(stn_row_1km_ind, stn_col_1km_ind)


# 4.2 Locate positions of in-situ stations using lat/lon tables of the world

stn_row_1km_ind_world_all = []
stn_col_1km_ind_world_all = []
for idt in range(len(stn_lat_all)):
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_1km)).item()
    stn_row_1km_ind_world_all.append(stn_row_1km_ind)
    stn_col_1km_ind_world_all.append(stn_col_1km_ind)
    del(stn_row_1km_ind, stn_col_1km_ind)


# 4.3.1 Extract SWDI data by position tables
for iyr in [4]:#range(len(yearname)):  # range(yearname):
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    swdi_stn_all = []
    for idt in range(len(tif_files)):
        swdi_tf = gdal.Open(tif_files[idt])
        swdi_arr = swdi_tf.ReadAsArray().astype(np.float32)
        swdi_stn = swdi_arr[stn_row_1km_ind_all, stn_col_1km_ind_all]
        swdi_stn_all.append(swdi_stn)
        print(tif_files[idt])

    swdi_stn_all = np.array(swdi_stn_all)

    # Divide the tif files by month
    tif_files_delta = [int(tif_files[x].split('.')[0][-3:]) - 1 for x in range(len(tif_files))]
    tif_files_month = [(datetime.date(yearname[iyr], 1, 1) + datetime.timedelta(tif_files_delta[x])).month for x in
                       range(len(tif_files_delta))]
    tif_files_month = np.array(tif_files_month)

    month_ind = [list(np.where(tif_files_month == x + 1))[0] for x in range(len(monthname))]
    swdi_stn_avg_all = [np.nanmean(swdi_stn_all[month_ind[x], :], axis=0) for x in range(len(monthname))]
    swdi_stn_avg_all = np.array(swdi_stn_avg_all)

# swdi_stn_all = np.copy(swdi_stn_avg_all)
# swdi_stn_all[6,:] = swdi_stn_all[5,:]

# 4.3.2 Extract SMAI data by position tables
smai_stn_all = []
for idt in range(len(monthname)):
    smai_stn = smai_all[stn_row_1km_ind_all, stn_col_1km_ind_all, idt]
    smai_stn_all.append(smai_stn)
smai_stn_all = np.array(smai_stn_all)



# 4.3.3 Load in the world SMAP 1 km SM and calculate monthly average
for iyr in [4]:
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    sm_stn_all = []
    for idt in range(len(tif_files)):
        sm_tf = gdal.Open(tif_files[idt])
        sm_arr = sm_tf.ReadAsArray().astype(np.float32)
        sm_stn = sm_arr[:, stn_row_1km_ind_world_all, stn_col_1km_ind_world_all]
        sm_stn = np.nanmean(sm_stn, axis=0)
        sm_stn_all.append(sm_stn)
        print(tif_files[idt])
    sm_stn_all = np.array(sm_stn_all)

# Save variable
# os.chdir(path_model)
var_name = ['sm_stn_all']
with h5py.File(path_aus_soil +'/sm_stn_all_2019.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()

# Read the SMAP SM
f_smap = h5py.File(path_aus_soil + "/sm_stn_all_2019.hdf5", "r")
varname_list = list(f_smap.keys())
sm_stn_all = f_smap[varname_list[0]][()]

sm_stn_avg_all = [np.nanmean(sm_stn_all[month_ind[x], :], axis=0) for x in range(len(monthname))]
sm_stn_avg_all = np.array(sm_stn_avg_all)


# 4.3.4 Extract GPM data
f_gpm = h5py.File(path_gpm + "/gpm_precip_2019.hdf5", "r")
varname_list_gpm = list(f_gpm.keys())

for x in range(len(varname_list_gpm)):
    var_obj = f_gpm[varname_list_gpm[x]][()]
    exec(varname_list_gpm[x] + '= var_obj')
    del(var_obj)
f_gpm.close()


# Locate the corresponding GPM 10 km data located by lat/lon of in-situ data
stn_row_10km_ind_all = []
stn_col_10km_ind_all = []

for idt in range(len(stn_lat_all)):
    stn_row_10km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_geo_10km)).item()
    stn_col_10km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_geo_10km)).item()
    stn_row_10km_ind_all.append(stn_row_10km_ind)
    stn_col_10km_ind_all.append(stn_col_10km_ind)
    del(stn_row_10km_ind, stn_col_10km_ind)

gpm_precip = []

linear_ind = np.ravel_multi_index([stn_row_10km_ind_all, stn_col_10km_ind_all],
                                  (gpm_precip_10km_2019.shape[0], gpm_precip_10km_2019.shape[1]))
gpm_precip_2d = np.reshape(gpm_precip_10km_2019, (gpm_precip_10km_2019.shape[0] * gpm_precip_10km_2019.shape[1],
                                                   gpm_precip_10km_2019.shape[2]))  # Convert from 3D to 2D
gpm_precip = gpm_precip_2d[linear_ind, :]
gpm_precip = np.transpose(gpm_precip, (1, 0))

# Divide the GPM data by month
# tif_files_delta = [int(tif_files[x].split('.')[0][-3:]) - 1 for x in range(len(tif_files))]
dates_1yr = list(range(365))
days_month = [(datetime.date(yearname[4], 1, 1) + datetime.timedelta(dates_1yr[x])).month for x in
                   range(len(dates_1yr))]
days_month = np.array(days_month)

month_ind_full = [list(np.where(days_month == x + 1))[0] for x in range(len(monthname))]
gpm_precip_sum = [np.nansum(gpm_precip[month_ind_full[x], :], axis=0) for x in range(len(monthname))]
gpm_precip_sum = np.array(gpm_precip_sum)


# 4.4 Make time-series plot
# stn_name_all = ['Weany Creek', 'Gnangara', 'Cheverelis', 'Temora', 'Tumbarumba', 'Tullochgorum']
stn_name_all = ['Weany Creek', 'Gnangara', 'Temora', 'Tumbarumba', 'Tullochgorum', 'Silver Springs', 'Samarra', 'Evergreen']

# 4.4.1 Together
fig = plt.figure(figsize=(14, 9))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for ist in range(6):

    x = sm_stn_avg_all[:, ist]*100
    y1 = swdi_stn_avg_all[:, ist]
    y2 = smai_stn_all[:, ist]
    z = gpm_precip_sum[:, ist]

    ax = fig.add_subplot(4, 2, ist+1)

    xmask = ~np.isnan(x)
    y1mask = ~np.isnan(y1)
    y2mask = ~np.isnan(y2)

    lns1 = ax.plot(x[xmask], c='k', marker='s', label='SM', markersize=5)
    lns2 = ax.plot(y1[y1mask], c='m', marker='s', label='SWDI', markersize=5)
    lns3 = ax.plot(y2[y2mask], c='b', marker='o', label='SMDI', markersize=5)

    plt.xlim(-1, len(y1))
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
    ax.grid(linestyle='--')
    # plt.ylim(-30, 50, 3)
    ax.set_ylim(-30, 30, 3)
    ax.set_yticks(np.arange(-30, 70, 20))
    ax.tick_params(axis='y', labelsize=12)
    ax.text(11.5, 40, stn_name_all[ist], fontsize=12, horizontalalignment='right')

    ax2 = ax.twinx()
    ax2.set_ylim(0, 80, 3)
    ax2.set_yticks(np.arange(0, 100, 20))
    ax2.invert_yaxis()
    lns4 = ax2.bar(np.arange(len(y1)), z, width=0.8, color='royalblue', label='P', alpha=0.5)
    ax2.tick_params(axis='y', labelsize=12)

# add all legends together
handles = lns1+lns2+lns3+[lns4]
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 3.7))

fig.text(0.51, 0.05, 'Months', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.42, 'Drought Indicators', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.95, 0.42, 'GPM Precip (mm)', rotation='vertical', fontsize=16, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=-0.15, top=0.95, hspace=0.4, wspace=0.2)

plt.savefig(path_results + '/t-series1' + '.png')
plt.close(fig)


# 4.4.2 Together (Separate precipitation and indices)
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for ist in range(8):

    # x = sm_stn_avg_all[:, ist]*100
    y1 = swdi_stn_avg_all[:, ist]
    y2 = smai_stn_all[:, ist]
    z = gpm_precip_sum[:, ist]

    ax = fig.add_subplot(4, 2, ist+1)

    # xmask = ~np.isnan(x)
    y1mask = ~np.isnan(y1)
    y2mask = ~np.isnan(y2)

    # lns1 = ax.plot(x[xmask], c='k', marker='s', label='SM', markersize=5)
    lns2 = ax.plot(y1[y1mask], c='m', marker='s', label='SWDI', markersize=5)
    # lns3 = ax.plot(y2[y2mask], c='b', marker='o', label='SMDI', markersize=5)

    plt.xlim(-0.5, len(y1)-0.5)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
    ax.grid(linestyle='--')
    ax.set_ylim(-30, 70, 3)
    ax.set_yticks(np.arange(-30, 50, 20))
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('SWDI', fontsize=12, fontweight='bold', verticalalignment='bottom', y=0.35, labelpad=0.1)
    ax.text(0, 20, stn_name_all[ist], fontsize=12, horizontalalignment='left')

    ax2 = ax.twinx()
    ax2.set_ylim(-6, 10, 3)
    ax2.set_yticks(np.arange(-6, 12, 4))
    lns3 = ax2.plot(y2[y2mask], c='b', marker='o', label='SMDI', markersize=5)
    # lns4 = ax2.bar(np.arange(len(y1)), z, width=0.8, color='royalblue', label='P', alpha=0.5)
    ax2.set_ylabel('SMDI', fontsize=12, fontweight='bold', labelpad=0.1)
    ax2.tick_params(axis='y', labelsize=12)

    divider = make_axes_locatable(ax2)
    axHistx = divider.append_axes("top", size=0.5, pad=0, sharex=ax)
    axHistx.set_ylim(0, 80, 4)
    axHistx.set_yticks(np.arange(0, 100, 25))
    axHistx.tick_params(axis='y', colors='royalblue')
    axHistx.set_ylabel('P', fontsize=12, fontweight='bold', labelpad=0.1)
    axHistx.yaxis.label.set_color('royalblue')
    axHistx.grid(linestyle='--')
    # axHistx.set_xticks([])
    axHistx.invert_yaxis()
    lns4 = axHistx.bar(np.arange(len(y1)), z, width=0.8, color='royalblue', label='Precipitation', alpha=0.5)

    plt.setp(axHistx.get_xticklabels(), visible=False)

# add all legends together
handles = lns2+lns3+[lns4]
labels = [l.get_label() for l in handles]

# handles, labels = ax.get_legend_handles_labels()
# plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 14))
plt.legend(handles, labels, loc=(-0.65, 14.8), mode="expand", borderaxespad=0, ncol=3, prop={"size": 13})

# fig.text(0.51, 0.05, 'Months', ha='center', fontsize=16, fontweight='bold')
# fig.text(0.04, 0.42, 'SWDI', rotation='vertical', fontsize=16, fontweight='bold')
# fig.text(0.95, 0.42, 'SMDI', rotation='vertical', fontsize=16, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, hspace=0.35, wspace=0.25)

plt.savefig(path_results + '/t-series1_new' + '.png')
plt.close(fig)


# 4.4.3 SWDI
fig = plt.figure(figsize=(14, 9))
# fig.subplots_adjust(hspace=0.2, wspace=0.2)
for ist in range(6):

    # x = sm_stn_avg_all[:, ist]*100
    y1 = swdi_stn_avg_all[:, ist]
    # y2 = smai_stn_all[:, ist]
    z = gpm_precip_sum[:, ist]

    ax = fig.add_subplot(4, 2, ist+1)

    # xmask = ~np.isnan(x)
    y1mask = ~np.isnan(y1)
    # y2mask = ~np.isnan(y2)

    # lns1 = ax.plot(x[xmask], c='k', marker='s', label='SM', markersize=5)
    lns2 = ax.plot(y1[y1mask], c='m', marker='s', label='SWDI', markersize=5)
    # lns3 = ax.plot(y2[y2mask], c='b', marker='o', label='SMDI', markersize=5)

    plt.xlim(-1, len(y1))
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
    ax.grid(linestyle='--')
    # plt.ylim(-30, 50, 3)
    ax.set_ylim(-30, 30, 3)
    ax.set_yticks(np.arange(-30, 45, 15))
    # plt.grid(linestyle='--')
    ax.tick_params(axis='y', labelsize=12)
    ax.text(11.5, 20, stn_name_all[ist], fontsize=12, horizontalalignment='right')

    ax2 = ax.twinx()
    ax2.set_ylim(0, 80, 3)
    ax2.set_yticks(np.arange(0, 100, 20))
    ax2.invert_yaxis()
    lns4 = ax2.bar(np.arange(len(y1)), z, width=0.8, color='royalblue', label='P', alpha=0.5)
    ax2.tick_params(axis='y', labelsize=12)

# add all legends together
# handles = lns1+lns2+lns3+[lns4]
handles = lns2+[lns4]
labels = [l.get_label() for l in handles]

# handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 3.7))

fig.text(0.51, 0.05, 'Months', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.42, 'SWDI', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.95, 0.42, 'GPM Precip (mm)', rotation='vertical', fontsize=16, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=-0.15, top=0.95, hspace=0.4, wspace=0.2)

plt.savefig(path_results + '/t-series1_swdi' + '.png')
plt.close(fig)

# 4.4.4 SMDI
fig = plt.figure(figsize=(14, 9))
# fig.subplots_adjust(hspace=0.2, wspace=0.2)
for ist in range(6):

    # x = sm_stn_avg_all[:, ist]*100
    # y1 = swdi_stn_avg_all[:, ist]
    y2 = smai_stn_all[:, ist]
    z = gpm_precip_sum[:, ist]

    ax = fig.add_subplot(4, 2, ist+1)

    # xmask = ~np.isnan(x)
    # y1mask = ~np.isnan(y1)
    y2mask = ~np.isnan(y2)

    # lns1 = ax.plot(x[xmask], c='k', marker='s', label='SM', markersize=5)
    # lns2 = ax.plot(y1[y1mask], c='m', marker='s', label='SWDI', markersize=5)
    lns3 = ax.plot(y2[y2mask], c='b', marker='s', label='SMDI', markersize=5)

    plt.xlim(-1, len(y2))
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
    ax.grid(linestyle='--')
    # plt.ylim(-30, 50, 3)
    ax.set_ylim(-6, 10, 3)
    ax.set_yticks(np.arange(-6, 12, 4))
    # plt.grid(linestyle='--')
    ax.tick_params(axis='y', labelsize=12)
    ax.text(11.5, 7, stn_name_all[ist], fontsize=12, horizontalalignment='right')

    ax2 = ax.twinx()
    ax2.set_ylim(0, 80, 3)
    ax2.set_yticks(np.arange(0, 100, 20))
    ax2.invert_yaxis()
    lns4 = ax2.bar(np.arange(len(y2)), z, width=0.8, color='royalblue', label='P', alpha=0.5)
    ax2.tick_params(axis='y', labelsize=12)

# add all legends together
# handles = lns1+lns2+lns3+[lns4]
handles = lns3+[lns4]
labels = [l.get_label() for l in handles]

# handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 3.7))

fig.text(0.51, 0.05, 'Months', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.42, 'SMDI', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.95, 0.42, 'GPM Precip (mm)', rotation='vertical', fontsize=16, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=-0.15, top=0.95, hspace=0.4, wspace=0.2)

plt.savefig(path_results + '/t-series1_smdi' + '.png')
plt.close(fig)


########################################################################################################################
# 5. Make box-plots for the selected locations

# 5.1 Weekly plots
# 5.1.1 Process the SWDI data
for iyr in [4]:#range(len(yearname)):
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    swdi_md_all = []
    for idt in range(len(tif_files)):
        swdi_tf = gdal.Open(tif_files[idt])
        swdi_arr = swdi_tf.ReadAsArray().astype(np.float32)
        swdi_md = swdi_arr[row_md_1km_ind[0]:row_md_1km_ind[-1]+1, col_md_1km_ind[0]:col_md_1km_ind[-1]+1]
        swdi_md = swdi_md.ravel()
        swdi_md_all.append(swdi_md)
        print(tif_files[idt])

    swdi_md_all = np.array(swdi_md_all)

swdi_days_ind = [int(tif_files[x].split('.')[0][-3:])-1 for x in range(len(tif_files))]

swdi_md_full = np.empty([365, swdi_md_all.shape[1]], dtype='float32')
swdi_md_full[:] = np.nan
swdi_md_full[swdi_days_ind, :] = swdi_md_all
swdi_md_weekly = [swdi_md_full[x*7:x*7+7, :].ravel() for x in range(365//7)]

swdi_md_weekly_full = []
for n in range(len(swdi_md_weekly)):
    swdi_md_array = swdi_md_weekly[n]
    swdi_md_array = swdi_md_array[~np.isnan(swdi_md_array)]
    swdi_md_weekly_full.append(swdi_md_array)
    del(swdi_md_array)

# Calculate Spatial SpDev of swdi
swdi_md_weekly_spdev = [np.sqrt(np.nanmean((swdi_md_all[x, :] - np.nanmean(swdi_md_all[x, :])) ** 2))
                        for x in range(len(swdi_md_all))]
swdi_md_weekly_spdev = np.array(swdi_md_weekly_spdev)
swdi_md_weekly_spdev_full = np.empty([365, ], dtype='float32')
swdi_md_weekly_spdev_full[:] = np.nan
swdi_md_weekly_spdev_full[swdi_days_ind, ] = swdi_md_weekly_spdev
swdi_md_weekly_spdev_group = [swdi_md_weekly_spdev_full[x*7:x*7+7, ].ravel() for x in range(365//7)]


# 5.1.2 Load in the world SMAP 1 km SM
for iyr in [4]:
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    sm_smap_all = []
    for idt in range(len(tif_files)):
        sm_tf = gdal.Open(tif_files[idt])
        sm_arr = sm_tf.ReadAsArray()[:, row_md_1km_world_ind[0]:row_md_1km_world_ind[-1]+1,
                 col_md_1km_world_ind[0]:col_md_1km_world_ind[-1]+1]
        sm_arr = np.nanmean(sm_arr, axis=0)
        sm_smap_all.append(sm_arr.ravel())
        print(tif_files[idt])
        del(sm_arr)
    sm_smap_all = np.array(sm_smap_all)

# Save the variable
os.chdir(path_model)
var_name = ['sm_smap_all']

with h5py.File('sm_smap_all_2019.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()

# Load the variable
f1 = h5py.File(path_aus_soil + "/sm_smap_all_2019.hdf5", "r")
varname_list = list(f1.keys())
sm_smap_all = f1[varname_list[0]][()]


smap_days_ind = [int(tif_files[x].split('.')[0][-3:])-1 for x in range(len(tif_files))]

smap_md_full = np.empty([365, sm_smap_all.shape[1]], dtype='float32')
smap_md_full[:] = np.nan
smap_md_full[smap_days_ind, :] = sm_smap_all
smap_md_weekly = [smap_md_full[x*7:x*7+7, :].ravel() for x in range(365//7)]

smap_md_weekly_full = []
for n in range(len(smap_md_weekly)):
    smap_md_array = smap_md_weekly[n]
    smap_md_array = smap_md_array[~np.isnan(smap_md_array)]
    smap_md_weekly_full.append(smap_md_array)
    del(smap_md_array)

# Calculate Spatial SpDev of SMAP SM
smap_md_weekly_spdev = [np.sqrt(np.nanmean((smap_md_full[x, :] - np.nanmean(smap_md_full[x, :])) ** 2))
                        for x in range(len(smap_md_full))]
smap_md_weekly_spdev = np.array(smap_md_weekly_spdev)
smap_md_weekly_spdev_group = [smap_md_weekly_spdev[x*7:x*7+7, ].ravel() for x in range(365//7)]


# 5.1.3 Extract GPM data
f_gpm = h5py.File(path_gpm + "/gpm_precip_2019.hdf5", "r")
varname_list_gpm = list(f_gpm.keys())

for x in range(len(varname_list_gpm)):
    var_obj = f_gpm[varname_list_gpm[x]][()]
    exec(varname_list_gpm[x] + '= var_obj')
    del(var_obj)
f_gpm.close()


gpm_precip = gpm_precip_10km_2019[row_md_10km_world_ind[0]:row_md_10km_world_ind[-1]+1,
             col_md_10km_world_ind[0]:col_md_10km_world_ind[-1]+1, :]
gpm_md_weekly_split = \
    [gpm_precip[:, :, x*7:x*7+7].reshape(gpm_precip.shape[0]*gpm_precip.shape[1], 7) for x in range(365//7)]
gpm_md_weekly = [np.nansum(gpm_precip[:, :, x*7:x*7+7], axis=2).ravel() for x in range(365//7)]

# gpm_precip = np.reshape(gpm_precip, (gpm_precip.shape[0]*gpm_precip.shape[1], gpm_precip.shape[2]))
# gpm_md_weekly = [gpm_precip[x*7:x*7+6, :].ravel() for x in range(365//7)]

gpm_md_weekly_full = []
for n in range(len(gpm_md_weekly)):
    gpm_md_weekly_mat = gpm_md_weekly[n]
    gpm_md_weekly_mat = gpm_md_weekly_mat[gpm_md_weekly_mat > 0]
    gpm_md_weekly_full.append(gpm_md_weekly_mat)
    del(gpm_md_weekly_mat)

# Calculate Spatial SpDev of GPM
# gpm_md_weekly_spdev = [np.sqrt(np.nanmean((gpm_precip[:, x] - np.nanmean(gpm_precip[:, x])) ** 2))
#                         for x in range(gpm_precip.shape[1])]
# gpm_md_weekly_spdev = np.array(gpm_md_weekly_spdev)
# gpm_md_weekly_spdev_group = [gpm_md_weekly_spdev[x*7:x*7+6, ].ravel() for x in range(365//7)]

gpm_md_weekly_spdev_group = \
    [np.sqrt(np.nanmean((gpm_md_weekly_split[x] - np.nanmean(gpm_md_weekly_split[x], axis=0)) ** 2, axis=0))
     for x in range(len(gpm_md_weekly_split))]


# 5.1.4.1 Make the boxplot (absolute values)
fig = plt.figure(figsize=(12, 12))
ax = fig.subplots(3, 1)
ax[0].boxplot(smap_md_weekly_full, 0, '')
ax[0].set_xticks(np.arange(1, 53, 5))
ax[0].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'], fontsize=13)
ax[0].tick_params(axis="y", labelsize=13)
ax[0].grid(linestyle='--')
# ax[0].set_title('SM', fontsize=16, position=(0.1, 0.85))
ax[1].boxplot(swdi_md_weekly_full, 0, '')
ax[1].set_xticks(np.arange(1, 53, 5))
ax[1].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'], fontsize=13)
ax[1].tick_params(axis="y", labelsize=13)
ax[1].grid(linestyle='--')
# ax[1].set_title('SWDI', fontsize=15, position=(0.1, 0.85))
ax[2].boxplot(gpm_md_weekly_full, 0, '')
ax[2].set_xticks(np.arange(1, 53, 5))
ax[2].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'], fontsize=13)
ax[2].tick_params(axis="y", labelsize=13)
ax[2].grid(linestyle='--')
# ax[2].set_title('Precipitation', fontsize=15, position=(0.1, 0.85))

fig.text(0.03, 0.1, 'Precipitation (mm)', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.5, 'SWDI', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.77, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.51, 0.01, 'Weeks', ha='center', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, hspace=0.25, wspace=0.25)
plt.show()
plt.savefig(path_results + '/boxplot_1' + '.png')
plt.close(fig)


# 5.1.4.2 Make the boxplot (spatial standard deviation)
fig = plt.figure(figsize=(12, 12))
ax = fig.subplots(3, 1)
ax[0].boxplot(smap_md_weekly_spdev_group, 0, '')
ax[0].set_xticks(np.arange(1, 53, 5))
ax[0].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'], fontsize=13)
ax[0].tick_params(axis="y", labelsize=13)
ax[0].grid(linestyle='--')
# ax[0].set_title('SM', fontsize=16, position=(0.1, 0.85))
ax[1].boxplot(swdi_md_weekly_spdev_group, 0, '')
ax[1].set_xticks(np.arange(1, 53, 5))
ax[1].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'], fontsize=13)
ax[1].tick_params(axis="y", labelsize=13)
ax[1].grid(linestyle='--')
# ax[1].set_title('SWDI', fontsize=15, position=(0.1, 0.85))
ax[2].boxplot(gpm_md_weekly_spdev_group, 0, '')
ax[2].set_xticks(np.arange(1, 53, 5))
ax[2].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'], fontsize=13)
ax[2].tick_params(axis="y", labelsize=13)
ax[2].grid(linestyle='--')
# ax[2].set_title('Precipitation', fontsize=15, position=(0.1, 0.85))

fig.text(0.03, 0.09, 'Precipitation($\sigma$)(mm)', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.48, 'SWDI($\sigma$)', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.03, 0.76, 'SM($\sigma$)($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.51, 0.01, 'Weeks', ha='center', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/boxplot_spdev' + '.png')
plt.close(fig)

########################################################################################################################

# 5.2.2 Load in the world SMAP 1 km SM
for iyr in [4]:
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    sm_smap_all = []
    for idt in range(len(tif_files)):
        sm_tf = gdal.Open(tif_files[idt])
        sm_arr = sm_tf.ReadAsArray()[:, row_md_1km_world_ind[0]:row_md_1km_world_ind[-1]+1,
                 col_md_1km_world_ind[0]:col_md_1km_world_ind[-1]+1]
        sm_arr = np.nanmean(sm_arr, axis=0)
        sm_smap_all.append(sm_arr.ravel())
        print(tif_files[idt])
        del(sm_arr)
    sm_smap_all = np.array(sm_smap_all)

# Save variable
os.chdir(path_model)
var_name = ['sm_smap_all']

with h5py.File('sm_smap_all_2019.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()

smap_days_ind = [int(tif_files[x].split('.')[0][-3:])-1 for x in range(len(tif_files))]

smap_md_full = np.empty([365, sm_smap_all.shape[1]], dtype='float32')
smap_md_full[:] = np.nan
smap_md_full[smap_days_ind, :] = sm_smap_all
smap_md_weekly = [smap_md_full[x*7:x*7+6, :].ravel() for x in range(365//7)]

smap_md_weekly_full = []
for n in range(len(smap_md_weekly)):
    smap_md_array = smap_md_weekly[n]
    smap_md_array = smap_md_array[~np.isnan(smap_md_array)]
    smap_md_weekly_full.append(smap_md_array)
    del(smap_md_array)

# Calculate Spatial SpDev of SMAP SM
smap_md_weekly_spdev = [np.sqrt(np.nanmean((smap_md_full[x, :] - np.nanmean(smap_md_full[x, :])) ** 2))
                        for x in range(len(smap_md_full))]
smap_md_weekly_spdev = np.array(smap_md_weekly_spdev)
smap_md_weekly_spdev_group = [smap_md_weekly_spdev[x*7:x*7+6, ].ravel() for x in range(365//7)]




# 5.2.4.1 Make the boxplot (absolute values)

fig = plt.figure(figsize=(12, 8))
ax = fig.subplots(3, 1)
ax[0].boxplot(smap_md_weekly_full, 0, '')
ax[0].set_xticks(np.arange(1, 53, 5))
ax[0].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'])
# ax[0].set_title('SM', fontsize=16, position=(0.1, 0.85))
ax[1].boxplot(swdi_md_weekly_full, 0, '')
ax[1].set_xticks(np.arange(1, 53, 5))
ax[1].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'])
# ax[1].set_title('SWDI', fontsize=15, position=(0.1, 0.85))
ax[2].boxplot(gpm_md_weekly_full, 0, '')
ax[2].set_xticks(np.arange(1, 53, 5))
ax[2].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'])
# ax[2].set_title('Precipitation', fontsize=15, position=(0.1, 0.85))

fig.text(0.04, 0.12, 'Precipitation', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.04, 0.5, 'SWDI', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.04, 0.78, 'SM', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.51, 0.01, 'Weeks', ha='center', fontsize=16, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.92, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/boxplot_1' + '.png')
plt.close(fig)


# 5.2.4.2 Make the boxplot (spatial standard deviation)

fig = plt.figure(figsize=(12, 8))
ax = fig.subplots(3, 1)
ax[0].boxplot(smap_md_weekly_spdev_group, 0, '')
ax[0].set_xticks(np.arange(1, 53, 5))
ax[0].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'])
# ax[0].set_title('SM', fontsize=16, position=(0.1, 0.85))
ax[1].boxplot(swdi_md_weekly_spdev_group, 0, '')
ax[1].set_xticks(np.arange(1, 53, 5))
ax[1].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'])
# ax[1].set_title('SWDI', fontsize=15, position=(0.1, 0.85))
ax[2].boxplot(gpm_md_weekly_spdev_group, 0, '')
ax[2].set_xticks(np.arange(1, 53, 5))
ax[2].set_xticklabels(['1', '6', '11', '16', '21', '26', '31', '36', '41', '46', '51'])
# ax[2].set_title('Precipitation', fontsize=15, position=(0.1, 0.85))

fig.text(0.04, 0.1, 'Precipitation($\sigma$)', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.04, 0.47, 'SWDI($\sigma$)', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.04, 0.76, 'SM($\sigma$)', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.51, 0.01, 'Weeks', ha='center', fontsize=16, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.92, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/boxplot_spdev' + '.png')
plt.close(fig)

########################################################################################################################
# 6. Validation (Monthly/daily plots)

# 6.1. Compare monthly 1 km SM, in-situ SM, SWDI, SMAI at 19 ISMN sites.
# 6.1.0. Load the site lat/lon from excel files and Locate the SM positions by lat/lon of in-situ data
ismn_list = sorted(glob.glob(path_ismn + '/[A-Z]*.xlsx'))
coords_all = []
df_table_am_all = []
df_table_pm_all = []
for ife in [0]:
    df_table_am = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='AM')
    df_table_pm = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='PM')

    netname = os.path.basename(ismn_list[ife]).split('_')[1]
    netname = [netname] * df_table_am.shape[0]
    coords = df_table_am[['lat', 'lon']]
    coords_all.append(coords)

    df_table_am_value = df_table_am.iloc[:, :]
    df_table_am_value.insert(0, 'network', netname)
    df_table_pm_value = df_table_pm.iloc[:, :]
    df_table_pm_value.insert(0, 'network', netname)
    df_table_am_all.append(df_table_am_value)
    df_table_pm_all.append(df_table_pm_value)
    del(df_table_am, df_table_pm, df_table_am_value, df_table_pm_value, coords, netname)
    print(ife)

df_coords = pd.concat(coords_all)
df_table_am_all = pd.concat(df_table_am_all)
df_table_pm_all = pd.concat(df_table_pm_all)

# Locate the SM pixel positions
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

# Locate positions of in-situ stations using lat/lon tables of Australia
# 1 km
stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_aus_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_aus_ease_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    del(stn_row_1km_ind, stn_col_1km_ind)

# Locate positions of in-situ stations using lat/lon tables of the world

stn_row_1km_ind_world_all = []
stn_col_1km_ind_world_all = []
for idt in range(len(stn_lat_all)):
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_1km)).item()
    stn_row_1km_ind_world_all.append(stn_row_1km_ind)
    stn_col_1km_ind_world_all.append(stn_col_1km_ind)
    del(stn_row_1km_ind, stn_col_1km_ind)

# 25 km
stn_row_25km_ind_all = []
stn_col_25km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_25km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_aus_ease_25km)).item()
    stn_col_25km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_aus_ease_25km)).item()
    stn_row_25km_ind_all.append(stn_row_25km_ind)
    stn_col_25km_ind_all.append(stn_col_25km_ind)
    del(stn_row_25km_ind, stn_col_25km_ind)


# 6.1.1. Calculate the monthly averaged in-situ SM
monthly_seq = np.reshape(daysofmonth_seq, (1, -1), order='F')
monthly_seq = monthly_seq[:, 3:] # Remove the first 3 months in 2015

ismn_sm_am_all = df_table_am_all.iloc[:, 3:]
ismn_sm_pm_all = df_table_pm_all.iloc[:, 3:]
ismn_sm_all = np.stack((ismn_sm_am_all, ismn_sm_pm_all), axis=2)
ismn_sm_all = np.nanmean(ismn_sm_all, axis=2)
monthly_seq_cumsum = np.cumsum(monthly_seq)
ismn_sm_all_split = np.hsplit(ismn_sm_all, monthly_seq_cumsum) # split by each month
ismn_sm_monthly = [np.nanmean(ismn_sm_all_split[x], axis=1) for x in range(len(ismn_sm_all_split))]
ismn_sm_monthly = np.stack(ismn_sm_monthly, axis=1)
ismn_sm_monthly = ismn_sm_monthly[:, :-1]
ismn_sm_monthly = np.transpose(ismn_sm_monthly, (1, 0))

# 6.1.2. Extract 1 km SMAP by lat/lon (daily)
smap_ext_allyear = []
for iyr in range(len(yearname)):
    os.chdir(path_aus_soil + '/smap_1km/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    tif_files_month = np.asarray([(datetime.datetime(yearname[iyr], 1, 1) +
                                   datetime.timedelta(int(os.path.basename(tif_files[x]).split('.')[0][-3:]) - 1)).month - 1
                                  for x in range(len(tif_files))])
    tif_files_month_ind = [np.where(tif_files_month == x)[0] for x in range(12)]

    smap_ext_1year = []
    for imo in range(len(tif_files_month_ind)):
        smap_ext_1month = []
        if len(tif_files_month_ind[imo]) != 0:
            for idt in range(len(tif_files_month_ind[imo])):
                smap_file = gdal.Open(tif_files[tif_files_month_ind[imo][idt]])
                smap_file = smap_file.ReadAsArray().astype(np.float32)
                smap_ext = smap_file[stn_row_1km_ind_all, stn_col_1km_ind_all]
                smap_ext_1month.append(smap_ext)
                print(tif_files[tif_files_month_ind[imo][idt]])

            smap_ext_1year.append(smap_ext_1month)
            del (smap_ext_1month)
        else:
            pass

    smap_ext_allyear.append(smap_ext_1year)

    del(smap_ext_1year)

smap_ext_allyear = list(chain.from_iterable(smap_ext_allyear))
for ils in range(len(smap_ext_allyear)):
    smap_ext_allyear[ils] = np.stack(smap_ext_allyear[ils], axis=0)
    smap_ext_allyear[ils] = np.transpose(smap_ext_allyear[ils], (1, 0))

ismn_sm_all_split.remove(ismn_sm_all_split[-1])


# Average ISMN/SMAP SM to monthly
ismn_sm_slc = []
smap_sm_slc = []
for imo in range(len(smap_ext_allyear)):
    ismn_sm_1month = []
    smap_sm_1month = []

    for ist in range(19):
        ismn_ind = np.where(~np.isnan(ismn_sm_all_split[imo][ist, :]))
        smap_ind = np.where(~np.isnan(smap_ext_allyear[imo][ist, :]))
        if len(ismn_ind) != 0 and len(smap_ind) != 0:
            # nonnan_ind = np.where(~np.isnan(ismn_sm_all_split[imo][ist, :]) & ~np.isnan(smap_ext_allyear[imo][ist, :]))[
            #     0]
            nonnan_ind = np.intersect1d(ismn_ind, smap_ind)
            ismn_sm_pt = np.nanmean(ismn_sm_all_split[imo][ist, nonnan_ind])
            smap_sm_pt = np.nanmean(smap_ext_allyear[imo][ist, nonnan_ind])
        else:
            ismn_sm_pt = np.nan
            smap_sm_pt = np.nan

        ismn_sm_1month.append(ismn_sm_pt)
        smap_sm_1month.append(smap_sm_pt)
        del(ismn_ind, smap_ind, nonnan_ind, ismn_sm_pt, smap_sm_pt)

    ismn_sm_1month = np.stack(ismn_sm_1month)
    smap_sm_1month = np.stack(smap_sm_1month)

    ismn_sm_slc.append(ismn_sm_1month)
    smap_sm_slc.append(smap_sm_1month)
    del(ismn_sm_1month, smap_sm_1month)

ismn_sm_slc = np.stack(ismn_sm_slc, axis=1)
smap_sm_slc = np.stack(smap_sm_slc, axis=1)


# 6.1.3. Extract GLDAS data by lat/lon
gldas_sm_monthly = gdal.Open(path_aus_soil + '/sm_gldas_25km_all_monthly.tif')
gldas_sm_monthly = gldas_sm_monthly.ReadAsArray()

gldas_ext_all = []
for imo in range(gldas_sm_monthly.shape[0]):
    # gldas_ext_1month = []
    # smap_file = gdal.Open(tif_files[tif_files_month_ind[imo][idt]])
    # smap_file = smap_file.ReadAsArray().astype(np.float32)
    gldas_ext = gldas_sm_monthly[imo, stn_row_25km_ind_all, stn_col_25km_ind_all]
    # gldas_ext_1month.append(gldas_ext)
    gldas_ext_all.append(gldas_ext)

    # print(imo)
    del(gldas_ext)

# Reorganize the extracted GLDAS data by station
gldas_sm_slc = np.empty([df_coords.shape[0], len(gldas_ext_all)+12], dtype='float32')
gldas_sm_slc[:] = np.nan

for ist in range(gldas_sm_slc.shape[0]):
    for imo in range(gldas_sm_slc.shape[1]-12):
        gldas_sm_slc[ist, imo] = gldas_ext_all[imo][ist]

gldas_sm_slc = gldas_sm_slc[:, 3:]

# Save ISMN and matched SMAP SM data
# ismn_sm_md = np.transpose(ismn_sm_monthly, (1, 0))
# smap_sm_md = np.transpose(smap_ismn_allyear, (1, 0))
# ismn_sm_md = ismn_sm_md[:, 1:]

columns = [str(yearname[x]) + '_' + monthname[y] for x in range(len(yearname)) for y in range(len(monthname))]
columns = columns[3:]

df_ismn_sm_md = pd.DataFrame(ismn_sm_slc, columns=columns, index=list(df_coords.index))
df_smap_sm_md = pd.DataFrame(smap_sm_slc, columns=columns, index=list(df_coords.index))
writer = pd.ExcelWriter('/Users/binfang/Downloads/smap_ismn_sm.xlsx')
df_ismn_sm_md.to_excel(writer, sheet_name='ISMN')
df_smap_sm_md.to_excel(writer, sheet_name='SMAP')
writer.save()

df_gldas_sm_md = pd.DataFrame(gldas_sm_slc, columns=columns, index=list(df_coords.index))
writer_gldas = pd.ExcelWriter('/Users/binfang/Downloads/gldas_sm.xlsx')
df_gldas_sm_md.to_excel(writer_gldas, sheet_name='GLDAS')
writer_gldas.save()


# 6.1.4. Extract 1 km SMAP by lat/lon and average to monthly
smap_ext_allyear = []
smap_md_allyear = []
smap_md_spdev_allyear = []
for iyr in range(len(yearname)):
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    tif_files_month = np.asarray([(datetime.datetime(yearname[iyr], 1, 1) +
                                   datetime.timedelta(int(os.path.basename(tif_files[x]).split('.')[0][-3:]) - 1)).month - 1
                                  for x in range(len(tif_files))])
    tif_files_month_ind = [np.where(tif_files_month == x)[0] for x in range(12)]

    smap_ext_1year = []
    smap_md_1year = []
    smap_md_spdev_1year = []
    for imo in range(len(tif_files_month_ind)):
        smap_ext_1month = []
        smap_md_1month = []
        smap_md_spdev_1month = []
        if len(tif_files_month_ind[imo]) != 0:
            for idt in range(len(tif_files_month_ind[imo])):
                smap_file = gdal.Open(tif_files[tif_files_month_ind[imo][idt]])
                smap_file = smap_file.ReadAsArray().astype(np.float32)
                smap_ext = smap_file[:, stn_row_1km_ind_world_all, stn_col_1km_ind_world_all]
                smap_ext = np.nanmean(smap_ext, axis=0)
                smap_md = smap_file[:, row_md_wrd_1km_ind[0]:row_md_wrd_1km_ind[-1]+1,
                          col_md_wrd_1km_ind[0]:col_md_wrd_1km_ind[-1]+1]
                smap_md = np.nanmean(smap_md, axis=0)
                smap_md = smap_md.ravel()
                smap_ext_1month.append(smap_ext)
                smap_md_1month.append(smap_md)
                print(tif_files[tif_files_month_ind[imo][idt]])

            smap_ext_1month = np.stack(smap_ext_1month, axis=1)
            smap_ext_1month = np.nanmean(smap_ext_1month, axis=1)
            smap_md_1month_avg = np.stack(smap_md_1month, axis=0)
            smap_md_1month_avg = np.nanmean(smap_md_1month_avg, axis=0)
            smap_md_1month_spdev = \
                np.array([np.sqrt(np.nanmean((smap_md_1month[x] - np.nanmean(smap_md_1month[x])) ** 2))
                 for x in range(len(smap_md_1month))])


            smap_ext_1year.append(smap_ext_1month)
            smap_md_1year.append(smap_md_1month_avg)
            smap_md_spdev_1year.append(smap_md_1month_spdev)
            del (smap_ext_1month, smap_md_1month, smap_md_1month_avg, smap_md_1month_spdev)
        else:
            pass
            # smap_ext_1month = smap_ext_empty
            # smap_md_1month = smap_md_empty
            # smap_md_1month_spdev = smap_md_spdev_empty

    smap_ext_allyear.append(smap_ext_1year)
    smap_md_allyear.append(smap_md_1year)
    smap_md_spdev_allyear.append(smap_md_spdev_1year)
    del(smap_ext_1year, smap_md_1year, smap_md_spdev_1year)

smap_ext_allyear = list(chain.from_iterable(smap_ext_allyear))
smap_ext_allyear = np.array(smap_ext_allyear)
smap_md_allyear = list(chain.from_iterable(smap_md_allyear))
smap_md_allyear = np.array(smap_md_allyear)
smap_md_spdev_allyear = list(chain.from_iterable(smap_md_spdev_allyear))
smap_md_spdev_allyear = np.array(smap_md_spdev_allyear)

os.chdir('/Volumes/MyPassport/SMAP_Project/Datasets/Australia')
var_name = ['smap_sm_spdev_allyear']
dt = h5py.special_dtype(vlen=np.float32)
with h5py.File('smap_1km_md_spdev.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x), dtype=dt)
f.close()

os.chdir('/Volumes/MyPassport/SMAP_Project/Datasets/Australia')
var_name = ['smap_ismn_allyear', 'smap_sm_allyear']
with h5py.File('smap_1km_md.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()




# 6.1.5. Extract the SWDI data / SMAI data
f = h5py.File("/Volumes/MyPassport/SMAP_Project/Datasets/Australia/smap_1km_md.hdf5", "r")
varname_list = ['smap_ismn_allyear', 'smap_sm_allyear']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Save ISMN and matched SMAP SM data
ismn_sm_md = np.transpose(ismn_sm_monthly, (1, 0))
smap_sm_md = np.transpose(smap_ismn_allyear, (1, 0))
ismn_sm_md = ismn_sm_md[:, 1:]

columns = [str(yearname[x]) + '_' + monthname[y] for x in range(len(yearname)) for y in range(len(monthname))]
columns = columns[4:]

df_ismn_sm_md = pd.DataFrame(ismn_sm_md, columns=columns, index=list(df_coords.index))
df_smap_sm_md = pd.DataFrame(smap_sm_md, columns=columns, index=list(df_coords.index))
writer = pd.ExcelWriter('/Users/binfang/Downloads/smap_ismn_sm.xlsx')
df_ismn_sm_md.to_excel(writer, sheet_name='ISMN')
df_smap_sm_md.to_excel(writer, sheet_name='SMAP')
writer.save()

# Load in SWDI and SMAI monthly data
swdi_tf = gdal.Open('/Volumes/MyPassport/SMAP_Project/Datasets/Australia/allyear_swdi_monthly.tif')
swdi_monthly = swdi_tf.ReadAsArray().astype(np.float32)
swdi_monthly = swdi_monthly[:, stn_row_1km_ind_all, stn_col_1km_ind_all]

smai_tf = gdal.Open('/Volumes/MyPassport/SMAP_Project/Datasets/Australia/allyear_smai_monthly.tif')
smai_monthly = smai_tf.ReadAsArray().astype(np.float32)
smai_monthly = smai_monthly[:, stn_row_1km_ind_all, stn_col_1km_ind_all]

# Add three lines to the ISMN table
ismn_sm_monthly = np.concatenate((swdi_monthly[:3, :], ismn_sm_monthly), axis=0)


# 6.2. Make time-series plot
stn_name_all = list(df_coords.index)

# 6.2.1
# Figure 1 (Together)
fig = plt.figure(figsize=(14, 12))
for ist in range(10):
    # x = ismn_sm_monthly[:, ist]*100
    y1 = swdi_monthly[:, ist]
    y2 = smai_monthly[:, ist]
    # z = ismn_sm_monthly[:, ist]

    ax = fig.add_subplot(5, 2, ist+1)

    # lns1 = ax.plot(x, c='k', marker='s', label='SM', markersize=4)
    lns2 = ax.plot(y1, c='m', marker='s', label='SWDI', markersize=4)
    lns3 = ax.plot(y2, c='b', marker='o', label='SMDI', markersize=4)

    plt.xlim(0, len(y1))
    ax.set_xticks(np.arange(0, 60, 12)+12)
    ax.set_xticklabels([])
    labels = ['2015', '2016', '2017', '2018', '2019']
    mticks = ax.get_xticks()
    ax.set_xticks(mticks-6, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
    ax.set_xticklabels(labels, minor=True)
    plt.ylim(-30, 50)
    ax.set_yticks(np.arange(-30, 70, 20))
    # plt.grid(linestyle='--')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(linestyle='--')
    ax.text(58, 35, stn_name_all[ist].replace('_', ' '), fontsize=16, horizontalalignment='right')

    # ax2 = ax.twinx()
    # ax2.set_ylim(0, 0.5, 5)
    # ax2.invert_yaxis()
    # lns4 = ax2.bar(np.arange(len(x)), z, width = 0.8, color='royalblue', label='Precip', alpha=0.5)
    # ax2.tick_params(axis='y', labelsize=10)

# add all legends together
handles = lns2+lns3
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 5.9))

fig.text(0.51, 0.01, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.4, 'Drought Indicators', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series1_monthly' + '.png')
plt.close(fig)


# Figure 2 (Together)
fig = plt.figure(figsize=(14, 12))
for ist in range(10, 19):
    y1 = swdi_monthly[:, ist]

    ax = fig.add_subplot(5, 2, ist-9)

    # lns1 = ax.plot(x, c='k', marker='s', label='SM', markersize=4)
    lns2 = ax.plot(y1, c='m', marker='s', label='SWDI', markersize=4)
    lns3 = ax.plot(y2, c='b', marker='o', label='SMDI', markersize=4)

    plt.xlim(0, len(y1))
    ax.set_xticks(np.arange(0, 60, 12)+12)
    ax.set_xticklabels([])
    labels = ['2015', '2016', '2017', '2018', '2019']
    mticks = ax.get_xticks()
    ax.set_xticks(mticks-6, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
    ax.set_xticklabels(labels, minor=True)
    plt.ylim(-30, 50)
    ax.set_yticks(np.arange(-30, 70, 20))
    # plt.grid(linestyle='--')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(linestyle='--')
    ax.text(58, 35, stn_name_all[ist].replace('_', ' '), fontsize=16, horizontalalignment='right')

    # ax2 = ax.twinx()
    # ax2.set_ylim(0, 0.5, 5)
    # ax2.invert_yaxis()
    # lns4 = ax2.bar(np.arange(len(x)), z, width = 0.8, color='royalblue', label='Precip', alpha=0.5)
    # ax2.tick_params(axis='y', labelsize=10)

# add all legends together
handles = lns2+lns3
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(2.25, 5.9))

fig.text(0.51, 0.01, 'Years', ha='center', fontsize=16, fontweight='bold')
fig.text(0.01, 0.4, 'Drought Indicators', rotation='vertical', fontsize=16, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series2_monthly' + '.png')
plt.close(fig)


# 6.2.2
site_name = df_coords.index

# Figure 1 (Separate)
fig = plt.figure(figsize=(12, 8))

y1 = swdi_monthly[:, 0]
y2 = swdi_monthly[:, 4]
y3 = swdi_monthly[:, 7]
y4 = swdi_monthly[:, 8]
y5 = swdi_monthly[:, 10]
y6 = swdi_monthly[:, 11]
y7 = swdi_monthly[:, 15]

ax = fig.add_subplot(1, 1, 1)

lns1 = ax.plot(y1, c='k', marker='s', label=site_name[0].replace('_', ' '), markersize=4)
lns2 = ax.plot(y2, c='m', marker='s', label=site_name[4].replace('_', ' '), markersize=4)
lns3 = ax.plot(y3, c='b', marker='s', label=site_name[7].replace('_', ' '), markersize=4)
lns4 = ax.plot(y4, c='g', marker='s', label=site_name[8].replace('_', ' '), markersize=4)
lns5 = ax.plot(y5, c='r', marker='s', label=site_name[10].replace('_', ' '), markersize=4)
lns6 = ax.plot(y6, c='c', marker='s', label=site_name[11].replace('_', ' '), markersize=4)
lns7 = ax.plot(y7, c='y', marker='s', label=site_name[15].replace('_', ' '), markersize=4)

plt.xlim(0, len(y1))
ax.set_xticks(np.arange(0, 60, 12)+12)
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax.get_xticks()
ax.set_xticks(mticks-6, minor=True)
ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
ax.set_xticklabels(labels, minor=True)
plt.ylim(-30, 50)
ax.set_yticks(np.arange(-30, 70, 20))
ax.tick_params(axis='y', labelsize=14)
ax.grid(linestyle='--')

# add all legends together
handles = lns1+lns2+lns3+lns4+lns5+lns6+lns7
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='upper right')
# plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 5.9))

fig.text(0.51, 0.01, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.47, 'SWDI', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series1_monthly_new1' + '.png')
plt.close(fig)


# Figure 2 (Separate)
fig = plt.figure(figsize=(12, 8))

y1 = swdi_monthly[:, 1]
y2 = swdi_monthly[:, 2]
y3 = swdi_monthly[:, 3]
y4 = swdi_monthly[:, 5]
y5 = swdi_monthly[:, 6]
y6 = swdi_monthly[:, 9]
y7 = swdi_monthly[:, 12]
y8 = swdi_monthly[:, 13]
y9 = swdi_monthly[:, 14]
y10 = swdi_monthly[:, 16]
y11 = swdi_monthly[:, 17]
y12 = swdi_monthly[:, 18]

ax = fig.add_subplot(1, 1, 1)

lns1 = ax.plot(y1, c='k', marker='s', label=site_name[1].replace('_', ' '), markersize=4)
lns2 = ax.plot(y2, c='m', marker='s', label=site_name[2].replace('_', ' '), markersize=4)
lns3 = ax.plot(y3, c='b', marker='s', label=site_name[3].replace('_', ' '), markersize=4)
lns4 = ax.plot(y4, c='g', marker='s', label=site_name[5].replace('_', ' '), markersize=4)
lns5 = ax.plot(y5, c='r', marker='s', label=site_name[6].replace('_', ' '), markersize=4)
lns6 = ax.plot(y6, c='c', marker='s', label=site_name[9].replace('_', ' '), markersize=4)
lns7 = ax.plot(y7, c='y', marker='s', label=site_name[12].replace('_', ' '), markersize=4)
lns8 = ax.plot(y8, c='tab:brown', marker='s', label=site_name[13].replace('_', ' '), markersize=4)
lns9 = ax.plot(y9, c='tab:purple', marker='s', label=site_name[14].replace('_', ' '), markersize=4)
lns10 = ax.plot(y10, c='tab:orange', marker='s', label=site_name[16].replace('_', ' '), markersize=4)
lns11 = ax.plot(y11, c='slategray', marker='s', label=site_name[17].replace('_', ' '), markersize=4)
lns12 = ax.plot(y12, c='gold', marker='s', label=site_name[18].replace('_', ' '), markersize=4)

plt.xlim(0, len(y1))
ax.set_xticks(np.arange(0, 60, 12)+12)
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax.get_xticks()
ax.set_xticks(mticks-6, minor=True)
ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
ax.set_xticklabels(labels, minor=True)
plt.ylim(-30, 50)
ax.set_yticks(np.arange(-30, 70, 20))
ax.tick_params(axis='y', labelsize=14)
ax.grid(linestyle='--')

# add all legends together
handles = lns1+lns2+lns3+lns4+lns5+lns6+lns7+lns8+lns9+lns10+lns11+lns12
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='upper right')
# plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 5.9))

fig.text(0.51, 0.01, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.47, 'SWDI', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series1_monthly_new2' + '.png')
plt.close(fig)


# Figure 3 (Separate)
fig = plt.figure(figsize=(12, 8))

y1 = smai_monthly[:, 0]
y2 = smai_monthly[:, 4]
y3 = smai_monthly[:, 7]
y4 = smai_monthly[:, 8]
y5 = smai_monthly[:, 10]
y6 = smai_monthly[:, 11]
y7 = smai_monthly[:, 15]

ax = fig.add_subplot(1, 1, 1)

lns1 = ax.plot(y1, c='k', marker='s', label=site_name[0].replace('_', ' '), markersize=4)
lns2 = ax.plot(y2, c='m', marker='s', label=site_name[4].replace('_', ' '), markersize=4)
lns3 = ax.plot(y3, c='b', marker='s', label=site_name[7].replace('_', ' '), markersize=4)
lns4 = ax.plot(y4, c='g', marker='s', label=site_name[8].replace('_', ' '), markersize=4)
lns5 = ax.plot(y5, c='r', marker='s', label=site_name[10].replace('_', ' '), markersize=4)
lns6 = ax.plot(y6, c='c', marker='s', label=site_name[11].replace('_', ' '), markersize=4)
lns7 = ax.plot(y7, c='y', marker='s', label=site_name[15].replace('_', ' '), markersize=4)

plt.xlim(0, len(y1))
ax.set_xticks(np.arange(0, 60, 12)+12)
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax.get_xticks()
ax.set_xticks(mticks-6, minor=True)
ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
ax.set_xticklabels(labels, minor=True)
plt.ylim(-6, 10)
ax.set_yticks(np.arange(-6, 14, 4))
ax.tick_params(axis='y', labelsize=14)
ax.grid(linestyle='--')

# add all legends together
handles = lns1+lns2+lns3+lns4+lns5+lns6+lns7
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='upper right')
# plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 5.9))

fig.text(0.51, 0.01, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.47, 'SMDI', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series2_monthly_new1' + '.png')
plt.close(fig)


# Figure 4 (Separate)
fig = plt.figure(figsize=(12, 8))

y1 = smai_monthly[:, 1]
y2 = smai_monthly[:, 2]
y3 = smai_monthly[:, 3]
y4 = smai_monthly[:, 5]
y5 = smai_monthly[:, 6]
y6 = smai_monthly[:, 9]
y7 = smai_monthly[:, 12]
y8 = smai_monthly[:, 13]
y9 = smai_monthly[:, 14]
y10 = smai_monthly[:, 16]
y11 = smai_monthly[:, 17]
y12 = smai_monthly[:, 18]

ax = fig.add_subplot(1, 1, 1)

lns1 = ax.plot(y1, c='k', marker='s', label=site_name[1].replace('_', ' '), markersize=4)
lns2 = ax.plot(y2, c='m', marker='s', label=site_name[2].replace('_', ' '), markersize=4)
lns3 = ax.plot(y3, c='b', marker='s', label=site_name[3].replace('_', ' '), markersize=4)
lns4 = ax.plot(y4, c='g', marker='s', label=site_name[5].replace('_', ' '), markersize=4)
lns5 = ax.plot(y5, c='r', marker='s', label=site_name[6].replace('_', ' '), markersize=4)
lns6 = ax.plot(y6, c='c', marker='s', label=site_name[9].replace('_', ' '), markersize=4)
lns7 = ax.plot(y7, c='y', marker='s', label=site_name[12].replace('_', ' '), markersize=4)
lns8 = ax.plot(y8, c='tab:brown', marker='s', label=site_name[13].replace('_', ' '), markersize=4)
lns9 = ax.plot(y9, c='tab:purple', marker='s', label=site_name[14].replace('_', ' '), markersize=4)
lns10 = ax.plot(y10, c='tab:orange', marker='s', label=site_name[16].replace('_', ' '), markersize=4)
lns11 = ax.plot(y11, c='slategray', marker='s', label=site_name[17].replace('_', ' '), markersize=4)
lns12 = ax.plot(y12, c='gold', marker='s', label=site_name[18].replace('_', ' '), markersize=4)

plt.xlim(0, len(y1))
ax.set_xticks(np.arange(0, 60, 12)+12)
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax.get_xticks()
ax.set_xticks(mticks-6, minor=True)
ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
ax.set_xticklabels(labels, minor=True)
plt.ylim(-6, 10)
ax.set_yticks(np.arange(-6, 14, 4))
ax.tick_params(axis='y', labelsize=14)
ax.grid(linestyle='--')

# add all legends together
handles = lns1+lns2+lns3+lns4+lns5+lns6+lns7+lns8+lns9+lns10+lns11+lns12
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='upper right')
# plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 5.9))

fig.text(0.51, 0.01, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.47, 'SMDI', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series2_monthly_new2' + '.png')
plt.close(fig)


########################################################################################################################
# 6.4 Extract data in the Murray-Darling Basin

f1 = h5py.File("/Volumes/MyPassport/SMAP_Project/Datasets/Australia/smap_1km_md_spdev.hdf5", "r")
varname_list = list(f1.keys())
for x in range(len(varname_list)):
    var_obj = f1[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f1.close()

smap_sm_spdev_allyear_ls = \
    [smap_sm_spdev_allyear[x][~np.isnan(smap_sm_spdev_allyear[x])] for x in range(len(smap_sm_spdev_allyear))]
smap_sm_spdev_allyear_ls = [[]] * 4 + smap_sm_spdev_allyear_ls

f2 = h5py.File("/Volumes/MyPassport/SMAP_Project/Datasets/Australia/smap_1km_md.hdf5", "r")
varname_list = list(f2.keys())
for x in range(len(varname_list)):
    var_obj = f2[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f2.close()

# smap_ismn_allyear_ls = \
#     [smap_ismn_allyear[x][~np.isnan(smap_ismn_allyear[x])] for x in range(len(smap_ismn_allyear))]
smap_sm_allyear_ls = \
    [smap_sm_allyear[x][~np.isnan(smap_sm_allyear[x])] for x in range(len(smap_sm_allyear))]
smap_sm_allyear_ls = [[]] * 4 + smap_sm_allyear_ls

# 6.4.1 Process the SWDI data and extract the Murray-Darling Basin
# # swdi_md_empty = np.empty([len(row_md_1km_ind)*len(col_md_1km_ind)], dtype='float32')
# # swdi_md_empty[:] = np.nan
swdi_md_allyear = []
swdi_md_spdev_allyear = []
for iyr in range(len(yearname)):
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    tif_files_month = np.asarray([(datetime.datetime(yearname[iyr], 1, 1) +
                                   datetime.timedelta(int(os.path.basename(tif_files[x]).split('.')[0][-3:]) - 1)).month - 1
                                  for x in range(len(tif_files))])
    tif_files_month_ind = [np.where(tif_files_month == x)[0] for x in range(12)]

    swdi_md_1year = []
    swdi_md_spdev_1year = []
    for imo in range(len(tif_files_month_ind)):
        swdi_md_1month = []
        swdi_md_1month_spdev = []
        if len(tif_files_month_ind[imo]) != 0:
            for idt in range(len(tif_files_month_ind[imo])):
                swdi_tf = gdal.Open(tif_files[tif_files_month_ind[imo][idt]])
                swdi_md = swdi_tf.ReadAsArray().astype(np.float32)
                swdi_md = swdi_md[row_md_1km_ind[0]:row_md_1km_ind[-1]+1, col_md_1km_ind[0]:col_md_1km_ind[-1]+1]
                swdi_md = swdi_md.ravel()
                swdi_md_1month.append(swdi_md)
                print(tif_files[tif_files_month_ind[imo][idt]])
            swdi_md_1month = np.stack(swdi_md_1month, axis=0)
            swdi_md_1month_avg = np.nanmean(swdi_md_1month, axis=0)
            swdi_md_1month_spdev = \
                np.sqrt(np.nanmean((swdi_md_1month - np.nanmean(swdi_md_1month, axis=1).reshape(-1, 1)) ** 2, axis=1))

            # swdi_md_1month = np.stack(swdi_md_1month, axis=2)
            # swdi_md_1month = np.nanmean(swdi_md_1month, axis=2)
            swdi_md_1year.append(swdi_md_1month_avg)
            swdi_md_spdev_1year.append(swdi_md_1month_spdev)
            del (swdi_md_1month, swdi_md_1month_avg, swdi_md_1month_spdev)

        else:
            pass
            # swdi_md_1month_avg = swdi_md_empty
            # swdi_md_1month_spdev = swdi_md_empty

    swdi_md_allyear.append(swdi_md_1year)
    swdi_md_spdev_allyear.append(swdi_md_spdev_1year)
    del(swdi_md_1year, swdi_md_spdev_1year)

swdi_md_allyear = list(chain.from_iterable(swdi_md_allyear))
swdi_md_allyear_ls = \
    [swdi_md_allyear[x][~np.isnan(swdi_md_allyear[x])] for x in range(len(swdi_md_allyear))]
swdi_md_allyear_ls = [[]] * 4 + swdi_md_allyear_ls

swdi_md_spdev_allyear = list(chain.from_iterable(swdi_md_spdev_allyear))
swdi_md_spdev_allyear_ls = \
    [swdi_md_spdev_allyear[x][~np.isnan(swdi_md_spdev_allyear[x])] for x in range(len(swdi_md_spdev_allyear))]
swdi_md_spdev_allyear_ls = [[]] * 4 + swdi_md_spdev_allyear_ls



# 6.4.2 Extract GPM data
os.chdir(path_gpm)
gpm_files = sorted(glob.glob('*.hdf5'))

gpm_precip_md_allyear = []
gpm_precip_md_allyear_spdev = []
for iyr in range(len(yearname)):
    f_gpm = h5py.File(gpm_files[iyr], 'r')
    varname_list_gpm = list(f_gpm.keys())
    gpm_precip_10km = f_gpm[varname_list_gpm[0]][()]
    f_gpm.close()

    month_seq = daysofmonth_seq[:, iyr]
    if iyr==0:
        month_seq = month_seq[3:, ]
    else:
        pass

    month_seq_cumsum = np.cumsum(month_seq)

    gpm_precip_md = gpm_precip_10km[row_md_10km_world_ind[0]:row_md_10km_world_ind[-1]+1,
                 col_md_10km_world_ind[0]:col_md_10km_world_ind[-1]+1, :]
    gpm_precip_md = \
        gpm_precip_md.reshape(gpm_precip_md.shape[0]*gpm_precip_md.shape[1], gpm_precip_md.shape[2])
    gpm_precip_md_split = np.hsplit(gpm_precip_md, month_seq_cumsum)  # split by each month
    gpm_precip_md_split = gpm_precip_md_split[:-1]
    gpm_precip_md_monthly = [np.nansum(gpm_precip_md_split[x], axis=1) for x in range(len(gpm_precip_md_split))]
    # gpm_precip_md_monthly = np.stack(gpm_precip_md_monthly, axis=1)
    # gpm_precip_md_monthly = np.transpose(gpm_precip_md_monthly, (1, 0))
    gpm_precip_md_monthly_spdev = \
        [np.sqrt(np.nanmean((gpm_precip_md_split[x] - np.nanmean(gpm_precip_md_split[x], axis=0)) ** 2, axis=0))
         for x in range(len(gpm_precip_md_split))]

    gpm_precip_md_allyear.append(gpm_precip_md_monthly)
    gpm_precip_md_allyear_spdev.append(gpm_precip_md_monthly_spdev)

    print(gpm_files[iyr])
    del(gpm_precip_md_monthly, gpm_precip_md_monthly_spdev, varname_list_gpm, gpm_precip_10km, month_seq,
        month_seq_cumsum, gpm_precip_md, gpm_precip_md_split)

gpm_precip_md_allyear = list(chain.from_iterable(gpm_precip_md_allyear))
gpm_precip_md_allyear_spdev = list(chain.from_iterable(gpm_precip_md_allyear_spdev))

gpm_precip_md_allyear = [[]] * 4 + gpm_precip_md_allyear[1:]
gpm_precip_md_allyear_spdev = [[]] * 4 + gpm_precip_md_allyear_spdev[1:]


# 6.5.1 Make the boxplot (absolute values)

fig = plt.figure(figsize=(13, 12))
ax = fig.subplots(3, 1)
ax[0].boxplot(smap_sm_allyear_ls, 0, '')
ax[0].set_xticks(np.arange(0, 60, 12)+12)
ax[0].set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax[0].get_xticks()
ax[0].set_xticks(mticks - 6, minor=True)
ax[0].tick_params(axis='x', which='minor', length=0, labelsize=14)
ax[0].tick_params(axis='y', labelsize=14)
ax[0].set_xticklabels(labels, minor=True)
ax[0].grid(linestyle='--')
# ax[0].set_title('SM', fontsize=16, position=(0.1, 0.85))
ax[1].boxplot(swdi_md_allyear_ls, 0, '')
ax[1].set_xticks(np.arange(0, 60, 12)+12)
ax[1].set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax[1].get_xticks()
ax[1].set_xticks(mticks - 6, minor=True)
ax[1].tick_params(axis='x', which='minor', length=0, labelsize=14)
ax[1].tick_params(axis='y', labelsize=14)
ax[1].set_xticklabels(labels, minor=True)
ax[1].grid(linestyle='--')
# ax[1].set_title('SWDI', fontsize=15, position=(0.1, 0.85))
ax[2].boxplot(gpm_precip_md_allyear, 0, '')
ax[2].set_xticks(np.arange(0, 60, 12)+12)
ax[2].set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax[1].get_xticks()
ax[2].set_xticks(mticks - 6, minor=True)
ax[2].tick_params(axis='x', which='minor', length=0, labelsize=14)
ax[2].tick_params(axis='y', labelsize=14)
ax[2].set_xticklabels(labels, minor=True)
ax[2].grid(linestyle='--')
# ax[2].set_title('Precipitation', fontsize=15, position=(0.1, 0.85))

fig.text(0.02, 0.1, 'Precipitation (mm)', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.02, 0.5, 'SWDI', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.02, 0.78, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.52, 0.01, 'Years', ha='center', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.96, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/boxplot_allyear_1' + '.png')
plt.close(fig)



# 6.5.2 Make the boxplot (spatial standard deviation)

fig = plt.figure(figsize=(13, 12))
ax = fig.subplots(3, 1)
ax[0].boxplot(smap_sm_spdev_allyear_ls, 0, '')
ax[0].set_xticks(np.arange(0, 60, 12)+12)
ax[0].set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax[0].get_xticks()
ax[0].set_xticks(mticks - 6, minor=True)
ax[0].tick_params(axis='x', which='minor', length=0, labelsize=14)
ax[0].tick_params(axis='y', labelsize=14)
ax[0].set_xticklabels(labels, minor=True)
ax[0].grid(linestyle='--')
# ax[0].set_title('SM', fontsize=16, position=(0.1, 0.85))
ax[1].boxplot(swdi_md_spdev_allyear_ls, 0, '')
ax[1].set_xticks(np.arange(0, 60, 12)+12)
ax[1].set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax[1].get_xticks()
ax[1].set_xticks(mticks - 6, minor=True)
ax[1].tick_params(axis='x', which='minor', length=0, labelsize=14)
ax[1].tick_params(axis='y', labelsize=14)
ax[1].set_xticklabels(labels, minor=True)
ax[1].grid(linestyle='--')
# ax[1].set_title('SWDI', fontsize=15, position=(0.1, 0.85))
ax[2].boxplot(gpm_precip_md_allyear_spdev, 0, '')
ax[2].set_xticks(np.arange(0, 60, 12)+12)
ax[2].set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019']
mticks = ax[1].get_xticks()
ax[2].set_xticks(mticks - 6, minor=True)
ax[2].tick_params(axis='x', which='minor', length=0, labelsize=14)
ax[2].tick_params(axis='y', labelsize=14)
ax[2].set_xticklabels(labels, minor=True)
ax[2].grid(linestyle='--')
# ax[2].set_title('Precipitation', fontsize=15, position=(0.1, 0.85))

fig.text(0.02, 0.09, 'Precipitation($\sigma$)(mm)', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.02, 0.5, 'SWDI($\sigma$)', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.02, 0.77, 'SM($\sigma$)($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.52, 0.01, 'Years', ha='center', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.96, bottom=0.08, top=0.96, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/boxplot_allyear_2' + '.png')
plt.close(fig)

########################################################################################################################
# 6.6.1 Make time series plot between 1 km SMAP SM and ISMN SM

sm_mat_init = np.empty([19, 48], dtype='float32')
sm_mat_init[:] = np.nan

ismn_sm = pd.read_excel(path_processed_2+'/smap_ismn_sm.xlsx', sheet_name='ISMN', index_col=0)
smap_sm_1km = pd.read_excel(path_processed_2+'/smap_ismn_sm.xlsx', sheet_name='SMAP', index_col=0)

ismn_sm_arr = np.copy(sm_mat_init)
ismn_sm_arr[:, 3:44] = np.array(ismn_sm)[:, 0:41]
smap_sm_1km_arr = np.copy(sm_mat_init)
smap_sm_1km_arr[:, 3:44] = np.array(smap_sm_1km)[:, 0:41]

stn_name_all = list(ismn_sm.index)


# Figure 1
fig = plt.figure(figsize=(14, 12))
for ist in range(10):
    x = ismn_sm_arr[ist, :]
    y = smap_sm_1km_arr[ist, :]

    ax = fig.add_subplot(5, 2, ist+1)
    lns1 = ax.plot(x, c='k', marker='s', label='ISMN', markersize=4)
    lns2 = ax.plot(y, c='m', marker='s', label='SMAP', markersize=4)

    plt.xlim(0, len(x))
    ax.set_xticks(np.arange(0, 48, 12))
    ax.set_xticklabels([])
    labels = ['2015', '2016', '2017', '2018']
    mticks = ax.get_xticks()
    ax.set_xticks(mticks+6, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
    ax.set_xticklabels(labels, minor=True)
    plt.ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.8, 0.2))
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(linestyle='--')
    ax.text(48, 0.5, stn_name_all[ist].replace('_', ' '), fontsize=16, horizontalalignment='right')

# add all legends together
handles = lns1+lns2
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 5.9))

fig.text(0.51, 0.02, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.5, 'SM', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series1_sm' + '.png')
plt.close(fig)

# Figure 2
fig = plt.figure(figsize=(14, 12))
for ist in range(10, 19):
    x = ismn_sm_arr[ist, :]
    y = smap_sm_1km_arr[ist, :]

    ax = fig.add_subplot(5, 2, ist-9)
    lns1 = ax.plot(x, c='k', marker='s', label='ISMN', markersize=4)
    lns2 = ax.plot(y, c='m', marker='s', label='SMAP', markersize=4)

    plt.xlim(0, len(x))
    ax.set_xticks(np.arange(0, 48, 12))
    ax.set_xticklabels([])
    labels = ['2015', '2016', '2017', '2018']
    mticks = ax.get_xticks()
    ax.set_xticks(mticks+6, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
    ax.set_xticklabels(labels, minor=True)
    plt.ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.8, 0.2))
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(linestyle='--')
    ax.text(48, 0.5, stn_name_all[ist].replace('_', ' '), fontsize=16, horizontalalignment='right')

# add all legends together
handles = lns1+lns2
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(2.25, 5.9))

fig.text(0.51, 0.02, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.5, 'SM', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series2_sm' + '.png')
plt.close(fig)


# 6.6.2 Make time series plot between 1 km SMAP SM, ISMN SM and GLDAS SM

sm_mat_init = np.empty([19, 48], dtype='float32')
sm_mat_init[:] = np.nan

ismn_sm = pd.read_excel(path_processed_2+'/smap_ismn_gldas_sm.xlsx', sheet_name='ISMN', index_col=0)
smap_sm_1km = pd.read_excel(path_processed_2+'/smap_ismn_gldas_sm.xlsx', sheet_name='SMAP', index_col=0)
gldas_sm = pd.read_excel(path_processed_2+'/smap_ismn_gldas_sm.xlsx', sheet_name='GLDAS', index_col=0)

ismn_sm_arr = np.copy(sm_mat_init)
ismn_sm_arr[:, 3:44] = np.array(ismn_sm)[:, 0:41]
smap_sm_1km_arr = np.copy(sm_mat_init)
smap_sm_1km_arr[:, 3:44] = np.array(smap_sm_1km)[:, 0:41]
gldas_sm_arr = np.copy(sm_mat_init)
gldas_sm_arr[:, 3:44] = np.array(gldas_sm)[:, 0:41]

stn_name_all = list(ismn_sm.index)


# Figure 1
fig = plt.figure(figsize=(14, 12))
for ist in range(10):
    x = ismn_sm_arr[ist, :]
    y = smap_sm_1km_arr[ist, :]
    z = gldas_sm_arr[ist, :]

    ax = fig.add_subplot(5, 2, ist+1)
    lns1 = ax.plot(x, c='k', marker='s', label='ISMN', markersize=4)
    lns2 = ax.plot(y, c='m', marker='s', label='SMAP', markersize=4)
    lns3 = ax.plot(z, c='b', marker='s', label='GLDAS', markersize=4)

    plt.xlim(0, len(x))
    ax.set_xticks(np.arange(0, 48, 12))
    ax.set_xticklabels([])
    labels = ['2015', '2016', '2017', '2018']
    mticks = ax.get_xticks()
    ax.set_xticks(mticks+6, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
    ax.set_xticklabels(labels, minor=True)
    plt.ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.8, 0.2))
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(linestyle='--')
    ax.text(48, 0.5, stn_name_all[ist].replace('_', ' '), fontsize=16, horizontalalignment='right')

# add all legends together
handles = lns1+lns2+lns3
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc=(-0.6, 6.1), mode="expand", borderaxespad=0, ncol=3, prop={"size": 13})
# plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 5.9))

fig.text(0.51, 0.02, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.5, 'SM', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series1_sm_new' + '.png')
plt.close(fig)


# Figure 2
fig = plt.figure(figsize=(14, 12))
for ist in range(10, 19):
    x = ismn_sm_arr[ist, :]
    y = smap_sm_1km_arr[ist, :]
    z = gldas_sm_arr[ist, :]

    ax = fig.add_subplot(5, 2, ist-9)
    lns1 = ax.plot(x, c='k', marker='s', label='ISMN', markersize=4)
    lns2 = ax.plot(y, c='m', marker='s', label='SMAP', markersize=4)
    lns3 = ax.plot(z, c='b', marker='s', label='GLDAS', markersize=4)

    plt.xlim(0, len(x))
    ax.set_xticks(np.arange(0, 48, 12))
    ax.set_xticklabels([])
    labels = ['2015', '2016', '2017', '2018']
    mticks = ax.get_xticks()
    ax.set_xticks(mticks+6, minor=True)
    ax.tick_params(axis='x', which='minor', length=0, labelsize=14)
    ax.set_xticklabels(labels, minor=True)
    plt.ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.8, 0.2))
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(linestyle='--')
    ax.text(48, 0.5, stn_name_all[ist].replace('_', ' '), fontsize=16, horizontalalignment='right')

# add all legends together
handles = lns1+lns2+lns3
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc=(0.65, 6.1), mode="expand", borderaxespad=0, ncol=3, prop={"size": 13})
# plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(2.25, 5.9))

fig.text(0.51, 0.02, 'Years', ha='center', fontsize=18, fontweight='bold')
fig.text(0.01, 0.5, 'SM', rotation='vertical', fontsize=18, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series2_sm_new' + '.png')
plt.close(fig)

