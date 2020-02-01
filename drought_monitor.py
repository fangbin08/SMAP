import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import matplotlib.ticker as mticker
import numpy as np
import glob
import h5py
import gdal
import fiona
import rasterio
import calendar
import datetime
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
path_smap_sm_ds = '/Users/binfang/Downloads/Processing/Downscale'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200126'
# Path of preview
path_preview = '/Users/binfang/Documents/SMAP_Project/results/results_191202/preview'
# Path of swdi data
path_swdi = '/Users/binfang/Downloads/Processing/Australia/swdi'

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
# 1 Extract the geographic information of the Australian soil data

src_tf = gdal.Open('/Volumes/MyPassport/SMAP_Project/Datasets/australian_soil_data/CLY_000_005_EV_N_P_AU_NAT_C_20140801.tif')
src_tf_arr = src_tf.ReadAsArray().astype(np.float32)

cellsize_aussoil = src_tf.GetGeoTransform()[1]
size_aus = src_tf_arr.shape
lat_aus_max = src_tf.GetGeoTransform()[3] - cellsize_aussoil/2
lon_aus_min = src_tf.GetGeoTransform()[0] - cellsize_aussoil/2
lat_aus_min = lat_aus_max - cellsize_aussoil*(size_aus[0]-1)
lon_aus_max = lon_aus_min + cellsize_aussoil*(size_aus[1]-1)

lat_aus_90m = np.linspace(lat_aus_max, lat_aus_min, size_aus[0])
lon_aus_90m = np.linspace(lon_aus_min, lon_aus_max, size_aus[1])
del(src_tf_arr)

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

os.chdir('/Users/binfang/Downloads/australia_soil_data')
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
out_ds_tiff = gdal.GetDriverByName('GTiff').Create('aussoil_data.tif',
     len(lon_aus_ease_1km), len(lat_aus_ease_1km), len(aussoil_files),  # Number of bands
     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])

# Loop write each band to Geotiff file
for idl in range(len(aussoil_files)):
    out_ds_tiff.GetRasterBand(idl + 1).WriteArray(aussoil_data[:, :, idl])
    out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
    out_ds_tiff.GetRasterBand(idl + 1).SetDescription(aussoil_files[idl].split('_')[0])
out_ds_tiff = None  # close dataset to write to disc

del(aussoil_data)




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


aussoil_tf = gdal.Open('/Users/binfang/Downloads/australia_soil_data/aussoil_data.tif')
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
smap_sm_aus_1km_data = gdal.Open('/Users/binfang/Downloads/Processing/Downscale/2019/smap_sm_1km_ds_2019001.tif')
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

for iyr in [4]:#range(len(yearname)):
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    for idt in range(261, len(tif_files)):
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



# 2.2 Generate the seasonally SWDI maps

# Extract the days for each season
for iyr in [4]:  # range(len(yearname)):
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    names_daily = np.array([int(os.path.splitext(tif_files[idt])[0].split('_')[-1][-3:]) for idt in range(len(tif_files))])

    # Divide by seasons
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    seasons_div = np.array([0, 90, 181, 273, 365])

    ind_season_all = []
    for i in range(len(seasons_div)-1):
        ind_season = np.where((names_daily > seasons_div[i]) & (names_daily <= seasons_div[i+1]))[0]
        ind_season_all.append(ind_season)
        del(ind_season)


# Average the SWDI data by seasons and map
swdi_arr_avg_all = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), 4], dtype='float32')
swdi_arr_avg_all[:] = np.nan
for iyr in [4]:#range(len(yearname)):  # range(yearname):
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    swdi_arr_avg_all = []
    for ise in range(len(ind_season_all)):
        season_list = ind_season_all[ise]

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


# 2.3 Make the seasonally averaged SWDI maps in Australia
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
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.25, wspace=0.2)
plt.show()
plt.savefig(path_results + '/swdi_aus.png')


# 2.4 Make the seasonally averaged SWDI maps in Murray-Darling River basin

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
fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
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
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.25, wspace=0.2)
plt.show()
plt.savefig(path_results + '/swdi_md.png')


del(swdi_arr_avg_all, masked_ds_md_1km, masked_ds_md_1km_all)


########################################################################################################################
# 3. Calculate Soil Moisture Deficit Index (SMDI)

row_aus_ind = np.where((row_lmask_ease_25km_ind >= row_aus_ease_25km_ind[0]) & (row_lmask_ease_25km_ind <= row_aus_ease_25km_ind[-1]))
col_aus_ind = np.where((col_lmask_ease_25km_ind >= col_aus_ease_25km_ind[0]) & (col_lmask_ease_25km_ind <= col_aus_ease_25km_ind[-1]))

lmask_ease_25km_aus_ind = np.intersect1d(row_aus_ind, col_aus_ind)

col_aus_ease_1km_ind_mat, row_aus_ease_1km_ind_mat = np.meshgrid(col_aus_ease_1km_ind, row_aus_ease_1km_ind)
aus_ease_1km_ind_mat = np.array([row_aus_ease_1km_ind_mat.flatten(), col_aus_ease_1km_ind_mat.flatten()])
aus_ease_1km_ind = np.ravel_multi_index(aus_ease_1km_ind_mat, (len(lat_world_ease_1km), len(lon_world_ease_1km)))

# 3.1 Extract the GLDAS grids in Australia and disaggregate to 1 km resolution,
# and calculate the maximun/minimum/median of each grid of each month

os.chdir(path_procdata)
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
        mat_output_25km[row_lmask_ease_25km_ind[lmask_ease_25km_aus_ind], col_lmask_ease_25km_ind[lmask_ease_25km_aus_ind]] = sm_gldas_stats[:, n]

        sm_1km_output = np.copy(sm_1km_init)
        sm_1km_output_1d = np.array([mat_output_25km[row_meshgrid_from_25km[0, aus_ease_1km_ind[x]],
                                                   col_meshgrid_from_25km[0, aus_ease_1km_ind[x]]]
                                   for x in range(len(aus_ease_1km_ind))])
        sm_1km_output[0, aus_ease_1km_ind] = sm_1km_output_1d
        sm_1km_output = sm_1km_output.reshape(len(lat_world_ease_1km), len(lon_world_ease_1km))
        sm_1km_output = sm_1km_output[row_aus_ease_1km_ind[0]:row_aus_ease_1km_ind[-1]+1, col_aus_ease_1km_ind[0]:col_aus_ease_1km_ind[-1]+1]

        sm_1km_aus[:, :, imo*3+n] = sm_1km_output

        del(mat_output_25km, sm_1km_output, sm_1km_output_1d)
        print(imo*3+n)

    del(sm_gldas_stats)


# Save the raster of monthly stats
out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_procdata + '/aus_stats.tif',
     len(lon_aus_ease_1km), len(lat_aus_ease_1km), sm_1km_aus.shape[2],  # Number of bands
     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])

# Loop write each band to Geotiff file
for idl in range(sm_1km_aus.shape[2]):
    out_ds_tiff.GetRasterBand(idl + 1).WriteArray(sm_1km_aus[:, :, idl])
    out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
out_ds_tiff = None  # close dataset to write to disc
del(out_ds_tiff)


########################################################################################################################
# 3.2 Calculate the SMDI by each month

# Load in SMAP 1 km SM and calculate monthly average
for iyr in [4]:
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Divide the tif files by month
    tif_files_delta = [int(tif_files[x].split('.')[0][-3:]) - 1 for x in range(len(tif_files))]
    tif_files_month = [(datetime.date(yearname[iyr], 1, 1) + datetime.timedelta(tif_files_delta[x])).month for x in range(len(tif_files_delta))]
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
            sm_arr = sm_arr[:, row_aus_ease_1km_ind[0]:row_aus_ease_1km_ind[-1]+1, col_aus_ease_1km_ind[0]:col_aus_ease_1km_ind[-1]+1]
            sm_arr = np.nanmean(sm_arr, axis=0)
            sm_arr_month[:, :, idt] = sm_arr
            print(tif_files[month_ind[idt]])
            del(sm_arr)

        sm_arr_month_avg = np.nanmean(sm_arr_month, axis=2)
        sm_arr_month_avg_all[:, :, imo] = sm_arr_month_avg
        del(sm_arr_month, month_ind, sm_arr_month_avg)


    # Save the raster of monthly SM of one year
    out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_procdata + '/smap_monthly_sm_' + str(yearname[iyr]) + '.tif',
         len(lon_aus_ease_1km), len(lat_aus_ease_1km), len(monthname),  # Number of bands
         gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])

    # Loop write each band to Geotiff file
    for idl in range(sm_arr_month_avg_all.shape[2]):
        out_ds_tiff.GetRasterBand(idl + 1).WriteArray(sm_arr_month_avg_all[:, :, idl])
        out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
    out_ds_tiff = None  # close dataset to write to disc
    del(out_ds_tiff)



# Load in the SMAP monthly average data and Australian statistical data

aus_smap_sm_monthly = gdal.Open(path_procdata + '/smap_monthly_sm_' + str(yearname[iyr]) + '.tif')
aus_smap_sm_monthly = aus_smap_sm_monthly.ReadAsArray().astype(np.float32)

aus_stats = gdal.Open(path_procdata + '/aus_stats.tif')
aus_stats = aus_stats.ReadAsArray().astype(np.float32)

sm_ind_max = np.arange(0, 36, 3)
sm_ind_min = np.arange(1, 36, 3)
sm_ind_median = np.arange(2, 36, 3)
aus_sm_max = aus_stats[sm_ind_max, :, :]
aus_sm_min = aus_stats[sm_ind_min, :, :]
aus_sm_median = aus_stats[sm_ind_median, :, :]

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

sm_defi_mat_all[:, :, 6] = sm_defi_mat_all[:, :, 5]

# Calculate the monthly SMDI
smai_all = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), len(monthname)], dtype='float32')
smai_all[:] = np.nan
smai_all[:, :, 0] = sm_defi_mat_all[:, :, 0]/50 # SMAI for the first month

for imo in range(1, len(monthname)):
    smai_all[:, :, imo] = 0.5 * smai_all[:, :, imo-1] + sm_defi_mat_all[:, :, imo]/50

# Normalize to the range of [-4, 4]
# smai_all_norm = smai_all * (4 / np.nanmax(np.abs(smai_all)))


########################################################################################################################
# 3.3 Make the seasonally averaged SMAI maps in Australia
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
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.locator_params(nbins=4)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.25, wspace=0.2)
plt.show()
plt.savefig(path_results + '/smai_aus.png')


# 3.4 Make the seasonally averaged SMAI maps in Murray-Darling River basin

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
fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
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
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.25, wspace=0.2)
plt.show()
plt.savefig(path_results + '/smai_md.png')



########################################################################################################################
# 4. Make time series plots

stn_lat_all = [-14.1592, -31.3497, -17.1200, -19.8820, -35.0050, -34.4046, -35.6560, -41.6694]
stn_lon_all = [131.3881, 115.9068, 145.6300, 146.5360, 146.2992, 147.5326, 148.1520, 147.9117]

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
os.chdir(path_procdata)
var_name = ['sm_stn_all']

with h5py.File('sm_stn_all_2019.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()

sm_stn_avg_all = [np.nanmean(sm_stn_all[month_ind[x], :], axis=0) for x in range(len(monthname))]
sm_stn_avg_all = np.array(sm_stn_avg_all)


# 4.3.4 Extract GPM data
f_gpm = h5py.File(path_procdata + "/gpm_precip_2019.hdf5", "r")
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
days_month = [(datetime.date(yearname[iyr], 1, 1) + datetime.timedelta(dates_1yr[x])).month for x in
                   range(len(dates_1yr))]
days_month = np.array(days_month)

month_ind_full = [list(np.where(days_month == x + 1))[0] for x in range(len(monthname))]
gpm_precip_sum = [np.nansum(gpm_precip[month_ind_full[x], :], axis=0) for x in range(len(monthname))]
gpm_precip_sum = np.array(gpm_precip_sum)


# 4.4 Make time-series plot
stn_name_all = ['Daly', 'Gnangara', 'Robsons Creek', 'Weany Creek', 'Yanco', 'Temora', 'Tumbarumba', 'Tullochgorum']

fig = plt.figure(figsize=(14, 8))
# fig.subplots_adjust(hspace=0.2, wspace=0.2)

for ist in range(8):

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
    lns3 = ax.plot(y2[y2mask], c='b', marker='o', label='SMAI', markersize=5)

    plt.xlim(0, len(y1))
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylim(-30, 50)
    # ax.set_yticks(np.arange(0, 0.6, 0.1))
    # plt.grid(linestyle='--')
    ax.tick_params(axis='y', labelsize=10)
    ax.text(10, 40, stn_name_all[ist], fontsize=12)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 100, 4)
    ax2.invert_yaxis()
    lns4 = ax2.bar(np.arange(len(x)), z, width = 0.8, color='royalblue', label='Precip', alpha=0.5)
    ax2.tick_params(axis='y', labelsize=10)

# add all legends together
handles = lns1+lns2+lns3+[lns4]
labels = [l.get_label() for l in handles]

handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 4.5))

fig.text(0.51, 0.01, 'Months', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.4, 'Drought Indicators', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.95, 0.4, 'GPM Precip (mm)', rotation='vertical', fontsize=16, fontweight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.92, hspace=0.25, wspace=0.25)
# plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/t-series1' + '.png')
plt.close(fig)



########################################################################################################################
# 5. Make box-plots for the selected locations

# 5.1 Process the SWDI data
for iyr in [4]:#range(len(yearname)):  # range(yearname):
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
swdi_md_weekly = [swdi_md_full[x*7:x*7+6, :].ravel() for x in range(365//7)]

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
swdi_md_weekly_spdev_group = [swdi_md_weekly_spdev_full[x*7:x*7+6, ].ravel() for x in range(365//7)]


# 5.2 Load in the world SMAP 1 km SM
for iyr in [4]:
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    sm_smap_all = []
    for idt in range(len(tif_files)):
        sm_tf = gdal.Open(tif_files[idt])
        sm_arr = sm_tf.ReadAsArray()[:, row_md_1km_world_ind[0]:row_md_1km_world_ind[-1]+1, col_md_1km_world_ind[0]:col_md_1km_world_ind[-1]+1]
        sm_arr = np.nanmean(sm_arr, axis=0)
        sm_smap_all.append(sm_arr.ravel())
        print(tif_files[idt])
        del(sm_arr)
    sm_smap_all = np.array(sm_smap_all)

# Save variable
os.chdir(path_procdata)
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


# 5.3 Extract GPM data
f_gpm = h5py.File(path_procdata + "/gpm_precip_2019.hdf5", "r")
varname_list_gpm = list(f_gpm.keys())

for x in range(len(varname_list_gpm)):
    var_obj = f_gpm[varname_list_gpm[x]][()]
    exec(varname_list_gpm[x] + '= var_obj')
    del(var_obj)
f_gpm.close()


gpm_precip = gpm_precip_10km_2019[row_md_10km_world_ind[0]:row_md_10km_world_ind[-1]+1,
             col_md_10km_world_ind[0]:col_md_10km_world_ind[-1]+1, :]
gpm_precip = np.reshape(gpm_precip, (gpm_precip.shape[0]*gpm_precip.shape[1], gpm_precip.shape[2]))
gpm_md_weekly = [gpm_precip[x*7:x*7+6, :].ravel() for x in range(365//7)]

gpm_md_weekly_full = []
for n in range(len(gpm_md_weekly)):
    gpm_md_weekly_mat = gpm_md_weekly[n]
    gpm_md_weekly_mat = gpm_md_weekly_mat[gpm_md_weekly_mat > 0]
    gpm_md_weekly_full.append(gpm_md_weekly_mat)
    del(gpm_md_weekly_mat)

# Calculate Spatial SpDev of GPM
gpm_md_weekly_spdev = [np.sqrt(np.nanmean((gpm_precip[:, x] - np.nanmean(gpm_precip[:, x])) ** 2))
                        for x in range(gpm_precip.shape[1])]
gpm_md_weekly_spdev = np.array(gpm_md_weekly_spdev)
gpm_md_weekly_spdev_group = [gpm_md_weekly_spdev[x*7:x*7+6, ].ravel() for x in range(365//7)]


# 5.4.1 Make the boxplot (absolute values)

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


# 5.4.2 Make the boxplot (spatial standard deviation)

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

