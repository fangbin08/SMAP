import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import matplotlib.ticker as mticker
import numpy as np
import glob
import h5py
import gdal
from osgeo import ogr
import fiona
import rasterio
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
import datetime

# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
# Function 2. Subset and reproject the Geotiff data to WGS84 projection

def sub_n_reproj(input_mat, kwargs_input, sub_window, output_crs, band_n):
    # Get the georeference and bounding parameters of subset image
    kwargs_sub = kwargs_input.copy()
    kwargs_sub.update({
        'height': sub_window.height,
        'width': sub_window.width,
        'transform': rasterio.windows.transform(sub_window, kwargs_sub['transform']),
        'count': band_n
    })

    # Generate subset rasterio dataset
    input_ds_subset = MemoryFile().open(**kwargs_sub)
    if band_n == 1:
        input_mat = np.expand_dims(input_mat, axis=0)
    else:
        pass

    for i in range(band_n):
        input_ds_subset.write(
            input_mat[i, sub_window.row_off:sub_window.row_off+sub_window.height, sub_window.col_off:sub_window.col_off+sub_window.width], i+1)

    # Reproject a dataset
    transform_reproj, width_reproj, height_reproj = \
        calculate_default_transform(input_ds_subset.crs, output_crs,
                                    input_ds_subset.width, input_ds_subset.height, *input_ds_subset.bounds)
    kwargs_reproj = input_ds_subset.meta.copy()
    kwargs_reproj.update({
            'crs': output_crs,
            'transform': transform_reproj,
            'width': width_reproj,
            'height': height_reproj,
            'count': band_n
        })

    output_ds = MemoryFile().open(**kwargs_reproj)
    for i in range(band_n):
        reproject(source=rasterio.band(input_ds_subset, i+1), destination=rasterio.band(output_ds, i+1),
                  src_transform=input_ds_subset.transform, src_crs=input_ds_subset.crs,
                  dst_transform=transform_reproj, dst_crs=output_crs, resampling=Resampling.nearest)

    return output_ds


########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of Land mask
path_lmask = '/Users/binfang/Downloads/Processing/processed_data'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data'
# Path of downscaled SM
path_smap_sm_ds = '/Users/binfang/Downloads/Processing/Downscale'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200810'
# Path of SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'

folder_400m = '/400m/'
folder_1km = '/1km/'
folder_9km = '/9km/'
yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]
output_crs = 'EPSG:4326'

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_conus_max', 'lat_conus_min', 'lon_conus_max', 'lon_conus_min', 'cellsize_400m', 'cellsize_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_conus_ease_1km', 'lon_conus_ease_1km',
                'lat_conus_ease_9km', 'lon_conus_ease_9km', 'lat_conus_ease_400m', 'lon_conus_ease_400m',
                'lat_conus_ease_12_5km', 'lon_conus_ease_12_5km', 'row_conus_ease_9km_ind', 'col_conus_ease_9km_ind',
                'row_conus_ease_1km_from_9km_ind', 'col_conus_ease_1km_from_9km_ind']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


########################################################################################################################
# 1 SMAP SM maps (CONUS)

# 1.1.0 Composite the data of the first 16 days of one specific month
year_plt = [2019]
month_plt = [8]
days_begin = 1
days_end = 24
days_n = days_end - days_begin + 1
plot_n = days_n//3

# 1.1.1 Load in SMAP 9 km SM
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + folder_9km + 'smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        smap_9km_stack = []
        for idt in range(days_begin-1, days_end):
            smap_9km_am = f_read_smap_9km[varname_list_smap_9km[0]][row_conus_ease_9km_ind[0]:row_conus_ease_9km_ind[-1]+1,
                          col_conus_ease_9km_ind[0]:col_conus_ease_9km_ind[-1]+1, idt] # AM
            smap_9km_pm = f_read_smap_9km[varname_list_smap_9km[1]][row_conus_ease_9km_ind[0]:row_conus_ease_9km_ind[-1]+1,
                          col_conus_ease_9km_ind[0]:col_conus_ease_9km_ind[-1]+1, idt] # PM
            smap_9km = np.nanmean(np.stack((smap_9km_am, smap_9km_pm), axis=2), axis=2)
            smap_9km_stack.append(smap_9km)
            del(smap_9km_am, smap_9km_pm, smap_9km)

        smap_9km_mean_1 = np.nanmean(np.stack(smap_9km_stack[:plot_n], axis=2), axis=2)
        smap_9km_mean_2 = np.nanmean(np.stack(smap_9km_stack[plot_n:plot_n*2], axis=2), axis=2)
        smap_9km_mean_3 = np.nanmean(np.stack(smap_9km_stack[plot_n*2:], axis=2), axis=2)
        smap_9km_mean_all = np.stack((smap_9km_mean_1, smap_9km_mean_2, smap_9km_mean_3), axis=0)
        del(smap_9km_stack, smap_9km_mean_1, smap_9km_mean_2, smap_9km_mean_3)

        f_read_smap_9km.close()


# 1.1.2 Load in SMAP 1km SM (AM)
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):

        smap_1km_stack = []
        for idt in range(days_n):  # 16 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + folder_1km + 'nldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray().astype(np.float32)
            src_tf_arr = np.nanmean(src_tf_arr, axis=0)
            smap_1km_stack.append(src_tf_arr)
            print(tif_file_smap_1km)
            del(tif_file_smap_1km, src_tf, src_tf_arr)

        smap_1km_mean_1 = np.nanmean(np.stack(smap_1km_stack[:plot_n], axis=2), axis=2)
        smap_1km_mean_2 = np.nanmean(np.stack(smap_1km_stack[plot_n:plot_n*2], axis=2), axis=2)
        smap_1km_mean_3 = np.nanmean(np.stack(smap_1km_stack[plot_n*2:], axis=2), axis=2)
        smap_1km_mean_all = np.stack((smap_1km_mean_1, smap_1km_mean_2, smap_1km_mean_3), axis=0)
        del(smap_1km_stack, smap_1km_mean_1, smap_1km_mean_2, smap_1km_mean_3)


# 1.1.3 Load in SMAP 400m SM (AM)
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):

        smap_400m_stack = []
        for idt in range(days_n):  # 16 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_400m = path_smap + folder_400m + str(iyr) + '/smap_sm_400m_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_400m)
            src_tf_arr = src_tf.ReadAsArray().astype(np.float32)
            src_tf_arr = np.nanmean(src_tf_arr, axis=0)
            smap_400m_stack.append(src_tf_arr)
            print(tif_file_smap_400m)
            del(tif_file_smap_400m, src_tf, src_tf_arr)

        smap_400m_mean_1 = np.nanmean(np.stack(smap_400m_stack[:plot_n], axis=2), axis=2)
        smap_400m_mean_2 = np.nanmean(np.stack(smap_400m_stack[plot_n:plot_n*2], axis=2), axis=2)
        smap_400m_mean_3 = np.nanmean(np.stack(smap_400m_stack[plot_n*2:], axis=2), axis=2)
        smap_400m_mean_all = np.stack((smap_400m_mean_1, smap_400m_mean_2, smap_400m_mean_3), axis=0)
        del(smap_400m_stack, smap_400m_mean_1, smap_400m_mean_2, smap_400m_mean_3)

# Disaggregate the 9 km SMAP SM by 1 km lat/lon tables
smap_9km_mean_all_dis = []
for idt in range(len(smap_9km_mean_all)):
    smap_9km_mean_all_1day = smap_9km_mean_all[idt, :, :]
    smap_9km_mean_all_1day_dis = \
        np.array([smap_9km_mean_all_1day[row_conus_ease_1km_from_9km_ind[x], :] for x in range(len(lat_conus_ease_1km))])
    smap_9km_mean_all_1day_dis = \
        np.array([smap_9km_mean_all_1day_dis[:, col_conus_ease_1km_from_9km_ind[y]] for y in range(len(lon_conus_ease_1km))])
    smap_9km_mean_all_1day_dis = np.fliplr(np.rot90(smap_9km_mean_all_1day_dis, 3))
    smap_9km_mean_all_dis.append(smap_9km_mean_all_1day_dis)

    del(smap_9km_mean_all_1day, smap_9km_mean_all_1day_dis)

smap_9km_mean_all_dis = np.array(smap_9km_mean_all_dis)


########################################################################################################################
# 1.2 Maps of August, 2019
# Define map references
kwargs_400m = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_400m), 'height': len(lat_conus_ease_400m),
          'count': 3, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(400.358009339824, 0.0, -12060785.03136207, 0.0, -400.358009339824, 5854236.10968041)
          }
kwargs_1km = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_1km), 'height': len(lat_conus_ease_1km),
          'count': 3, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(1000.89502334956, 0.0, -12060785.031554235, 0.0, -1000.89502334956, 5854234.991137885)
          }
kwargs_9km = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_9km), 'height': len(lat_conus_ease_9km),
          'count': 3, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(9009.093602916, 0.0, -12061786.445962137, 0.0, -9009.093602916, 5855236.405530798)
          }
# kwargs_9km_dis = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
#           'width': len(lon_conus_ease_1km), 'height': len(lat_conus_ease_1km),
#           'count': 3, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
#           'transform': Affine(1000.89502334956, 0.0, -12060785.031554235, 0.0, -1000.89502334956, 5854234.991137885)
#           }

# 1.2.1 (CONUS)
window_400m = Window(0, 0, len(lon_conus_ease_400m), len(lat_conus_ease_400m))
window_9km = Window(0, 0, len(lon_conus_ease_9km), len(lat_conus_ease_9km))
window_1km = Window(0, 0, len(lon_conus_ease_1km), len(lat_conus_ease_1km))

smap_400m_mean_all_reprj = sub_n_reproj(smap_400m_mean_all, kwargs_400m, window_400m, output_crs, 3)
smap_400m_mean_all_reprj = smap_400m_mean_all_reprj.read()
smap_1km_mean_all_reprj = sub_n_reproj(smap_1km_mean_all, kwargs_1km, window_1km, output_crs, 3)
smap_1km_mean_all_reprj = smap_1km_mean_all_reprj.read()
smap_9km_mean_all_reprj = sub_n_reproj(smap_9km_mean_all, kwargs_9km, window_9km, output_crs, 3)
smap_9km_mean_all_reprj = smap_9km_mean_all_reprj.read()


# title_content = ['400 m (08/01 - 08/08)', '9 km (08/01 - 08/08)', '400 m (08/09 - 08/16)', '9 km (08/09 - 08/16)',
#               '400 m (08/17 - 08/24)', '9 km (08/17 - 08/24)']
shape_conus = ShapelyFeature(Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

fig = plt.figure(figsize=(12, 9), dpi=100, facecolor='w', edgecolor='k')
for ipt in range(2):
    # 400 m
    ax = fig.add_subplot(3, 2, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_conus)
    # ax.set_title(title_content[ipt], pad=15, fontsize=13, fontweight='bold')
    img = ax.imshow(smap_400m_mean_all_reprj[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=[-125, -67, 25, 53])
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=15)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 1 km
    ax = fig.add_subplot(3, 2, ipt+3, projection=ccrs.PlateCarree())
    ax.add_feature(shape_conus)
    # ax.set_title(title_content[ipt*2+1], pad=15, fontsize=13, fontweight='bold')
    img = ax.imshow(smap_1km_mean_all_reprj[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=[-125, -67, 25, 53])
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=15)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 9 km
    ax = fig.add_subplot(3, 2, ipt+5, projection=ccrs.PlateCarree())
    ax.add_feature(shape_conus)
    # ax.set_title(title_content[ipt*2+1], pad=15, fontsize=13, fontweight='bold')
    img = ax.imshow(smap_9km_mean_all_reprj[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=[-125, -67, 25, 53])
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=15)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

plt.subplots_adjust(left=0.09, right=0.88, bottom=0.05, top=0.92, hspace=0.2, wspace=0.2)
cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
fig.text(0.2, 0.95, '08/01 - 08/08', fontsize=15, fontweight='bold')
fig.text(0.63, 0.95, '08/09 - 08/16', fontsize=15, fontweight='bold')
fig.text(0.02, 0.75, '400 m', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.02, 0.45, '1 km', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.02, 0.15, '9 km', fontsize=15, fontweight='bold', rotation=90)
plt.show()

plt.savefig(path_results + '/sm_comp_conus.png')
plt.close()


# 1.2.2 (Upper San Pedro)
path_shapefile_usp = path_gis_data + '/watershed_boundary/uppersanpedro.shp'
shapefile_usp = fiona.open(path_shapefile_usp, 'r')
crop_shape_usp = [feature["geometry"] for feature in shapefile_usp]
shp_usp_extent = list(shapefile_usp.bounds)

#Subset the region of watershed
[lat_400m_usp, row_usp_400m_ind, lon_400m_usp, col_usp_400m_ind] = \
    coordtable_subset(lat_conus_ease_400m, lon_conus_ease_400m, shp_usp_extent[3], shp_usp_extent[1], shp_usp_extent[2], shp_usp_extent[0])
[lat_1km_usp, row_usp_1km_ind, lon_1km_usp, col_usp_1km_ind] = \
    coordtable_subset(lat_conus_ease_1km, lon_conus_ease_1km, shp_usp_extent[3], shp_usp_extent[1], shp_usp_extent[2], shp_usp_extent[0])
# [lat_9km_usp, row_usp_9km_ind, lon_9km_usp, col_usp_9km_ind] = \
#     coordtable_subset(lat_conus_ease_9km, lon_conus_ease_9km, shp_usp_extent[3], shp_usp_extent[1], shp_usp_extent[2], shp_usp_extent[0])

window_400m = Window(col_usp_400m_ind[0], row_usp_400m_ind[0], len(col_usp_400m_ind), len(row_usp_400m_ind))
window_1km = Window(col_usp_1km_ind[0], row_usp_1km_ind[0], len(col_usp_1km_ind), len(row_usp_1km_ind))
# window_9km = Window(col_usp_9km_ind[0], row_usp_9km_ind[0], len(col_usp_9km_ind), len(row_usp_9km_ind))

smap_400m_mean_all_usp = sub_n_reproj(smap_400m_mean_all, kwargs_400m, window_400m, output_crs, 3)
smap_1km_mean_all_usp = sub_n_reproj(smap_1km_mean_all, kwargs_1km, window_1km, output_crs, 3)
smap_9km_mean_all_usp = sub_n_reproj(smap_9km_mean_all_dis, kwargs_1km, window_1km, output_crs, 3)

masked_smap_400m_mean_usp, mask_transform_smap_400m_mean_usp = \
    mask(dataset=smap_400m_mean_all_usp, shapes=crop_shape_usp, crop=True)
masked_smap_400m_mean_usp[np.where(masked_smap_400m_mean_usp == 0)] = np.nan
masked_smap_1km_mean_usp, mask_transform_smap_1km_mean_usp = \
    mask(dataset=smap_1km_mean_all_usp, shapes=crop_shape_usp, crop=True)
masked_smap_1km_mean_usp[np.where(masked_smap_1km_mean_usp == 0)] = np.nan
masked_smap_9km_mean_usp, mask_transform_smap_9km_mean_usp = \
    mask(dataset=smap_9km_mean_all_usp, shapes=crop_shape_usp, crop=True)
masked_smap_9km_mean_usp[np.where(masked_smap_9km_mean_usp == 0)] = np.nan

# Make the maps (Upper San Pedro)
feature_shp_usp = ShapelyFeature(Reader(path_shapefile_usp).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_usp = np.array(smap_400m_mean_all_usp.bounds)
extent_usp = extent_usp[[0, 2, 1, 3]]

fig = plt.figure(figsize=(12, 12), dpi=100, facecolor='w', edgecolor='k')
for ipt in range(3):
    # 400 m
    ax = fig.add_subplot(3, 3, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_usp)
    img = ax.imshow(masked_smap_400m_mean_usp[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_usp)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 1 km
    ax = fig.add_subplot(3, 3, ipt+4, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_usp)
    img = ax.imshow(masked_smap_1km_mean_usp[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_usp)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 9 km
    ax = fig.add_subplot(3, 3, ipt+7, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_usp)
    img = ax.imshow(masked_smap_9km_mean_usp[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_usp)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

plt.subplots_adjust(left=0.07, right=0.88, bottom=0.05, top=0.92, hspace=0.2, wspace=0.2)
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=12, x=0.95)
cbar.ax.tick_params(labelsize=11)
cbar_ax.locator_params(nbins=6)
fig.text(0.12, 0.95, '08/01 - 08/08', fontsize=15, fontweight='bold')
fig.text(0.41, 0.95, '08/09 - 08/16', fontsize=15, fontweight='bold')
fig.text(0.69, 0.95, '08/17 - 08/24', fontsize=15, fontweight='bold')
fig.text(0.02, 0.75, '400 m', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.02, 0.45, '1 km', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.02, 0.15, '9 km', fontsize=15, fontweight='bold', rotation=90)
plt.show()

plt.savefig(path_results + '/sm_comp_usp.png')
plt.close()


# 1.2.3 (Middle-Snake watershed)
path_shapefile_ms = path_gis_data + '/watershed_boundary/midsnake.shp'
shapefile_ms = fiona.open(path_shapefile_ms, 'r')
crop_shape_ms = [feature["geometry"] for feature in shapefile_ms]
shp_ms_extent = list(shapefile_ms.bounds)

#Subset the region of watershed
[lat_400m_ms, row_ms_400m_ind, lon_400m_ms, col_ms_400m_ind] = \
    coordtable_subset(lat_conus_ease_400m, lon_conus_ease_400m, shp_ms_extent[3], shp_ms_extent[1], shp_ms_extent[2], shp_ms_extent[0])
[lat_1km_ms, row_ms_1km_ind, lon_1km_ms, col_ms_1km_ind] = \
    coordtable_subset(lat_conus_ease_1km, lon_conus_ease_1km, shp_ms_extent[3], shp_ms_extent[1], shp_ms_extent[2], shp_ms_extent[0])
# [lat_9km_ms, row_ms_9km_ind, lon_9km_ms, col_ms_9km_ind] = \
#     coordtable_subset(lat_conus_ease_9km, lon_conus_ease_9km, shp_ms_extent[3], shp_ms_extent[1], shp_ms_extent[2], shp_ms_extent[0])

window_400m = Window(col_ms_400m_ind[0], row_ms_400m_ind[0], len(col_ms_400m_ind), len(row_ms_400m_ind))
window_1km = Window(col_ms_1km_ind[0], row_ms_1km_ind[0], len(col_ms_1km_ind), len(row_ms_1km_ind))
# window_9km = Window(col_ms_9km_ind[0], row_ms_9km_ind[0], len(col_ms_9km_ind), len(row_ms_9km_ind))

smap_400m_mean_all_ms = sub_n_reproj(smap_400m_mean_all, kwargs_400m, window_400m, output_crs, 3)
smap_1km_mean_all_ms = sub_n_reproj(smap_1km_mean_all, kwargs_1km, window_1km, output_crs, 3)
smap_9km_mean_all_ms = sub_n_reproj(smap_9km_mean_all_dis, kwargs_1km, window_1km, output_crs, 3)

masked_smap_400m_mean_ms, mask_transform_smap_400m_mean_ms = \
    mask(dataset=smap_400m_mean_all_ms, shapes=crop_shape_ms, crop=True)
masked_smap_400m_mean_ms[np.where(masked_smap_400m_mean_ms == 0)] = np.nan
masked_smap_1km_mean_ms, mask_transform_smap_1km_mean_ms = \
    mask(dataset=smap_1km_mean_all_ms, shapes=crop_shape_ms, crop=True)
masked_smap_1km_mean_ms[np.where(masked_smap_1km_mean_ms == 0)] = np.nan
masked_smap_9km_mean_ms, mask_transform_smap_9km_mean_ms = \
    mask(dataset=smap_9km_mean_all_ms, shapes=crop_shape_ms, crop=True)
masked_smap_9km_mean_ms[np.where(masked_smap_9km_mean_ms == 0)] = np.nan

# Make the maps (Middle Snake)
feature_shp_ms = ShapelyFeature(Reader(path_shapefile_ms).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_ms = np.array(smap_400m_mean_all_ms.bounds)
extent_ms = extent_ms[[0, 2, 1, 3]]

fig = plt.figure(figsize=(15, 11), dpi=100, facecolor='w', edgecolor='k')
for ipt in range(3):
    # 400 m
    ax = fig.add_subplot(3, 3, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ms)
    img = ax.imshow(masked_smap_400m_mean_ms[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_ms)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 1 km
    ax = fig.add_subplot(3, 3, ipt+4, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ms)
    img = ax.imshow(masked_smap_1km_mean_ms[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_ms)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 9 km
    ax = fig.add_subplot(3, 3, ipt+7, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ms)
    img = ax.imshow(masked_smap_9km_mean_ms[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_ms)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

plt.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.92, hspace=0.2, wspace=0.2)
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=12, x=0.95)
cbar.ax.tick_params(labelsize=11)
cbar_ax.locator_params(nbins=6)
fig.text(0.12, 0.95, '08/01 - 08/08', fontsize=15, fontweight='bold')
fig.text(0.42, 0.95, '08/09 - 08/16', fontsize=15, fontweight='bold')
fig.text(0.71, 0.95, '08/17 - 08/24', fontsize=15, fontweight='bold')
fig.text(0.01, 0.75, '400 m', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.01, 0.45, '1 km', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.01, 0.15, '9 km', fontsize=15, fontweight='bold', rotation=90)
plt.show()

plt.savefig(path_results + '/sm_comp_ms.png')
plt.close()




# 1.2.4 (Upper Washita watershed)
path_shapefile_uw = path_gis_data + '/watershed_boundary/upper_washita.shp'
shapefile_uw = fiona.open(path_shapefile_uw, 'r')
crop_shape_uw = [feature["geometry"] for feature in shapefile_uw]
shp_uw_extent = list(shapefile_uw.bounds)

#Subset the region of watershed
[lat_400m_uw, row_uw_400m_ind, lon_400m_uw, col_uw_400m_ind] = \
    coordtable_subset(lat_conus_ease_400m, lon_conus_ease_400m, shp_uw_extent[3], shp_uw_extent[1], shp_uw_extent[2], shp_uw_extent[0])
[lat_1km_uw, row_uw_1km_ind, lon_1km_uw, col_uw_1km_ind] = \
    coordtable_subset(lat_conus_ease_1km, lon_conus_ease_1km, shp_uw_extent[3], shp_uw_extent[1], shp_uw_extent[2], shp_uw_extent[0])
# [lat_9km_uw, row_uw_9km_ind, lon_9km_uw, col_uw_9km_ind] = \
#     coordtable_subset(lat_conus_ease_9km, lon_conus_ease_9km, shp_uw_extent[3], shp_uw_extent[1], shp_uw_extent[2], shp_uw_extent[0])

window_400m = Window(col_uw_400m_ind[0], row_uw_400m_ind[0], len(col_uw_400m_ind), len(row_uw_400m_ind))
window_1km = Window(col_uw_1km_ind[0], row_uw_1km_ind[0], len(col_uw_1km_ind), len(row_uw_1km_ind))
# window_9km = Window(col_uw_9km_ind[0], row_uw_9km_ind[0], len(col_uw_9km_ind), len(row_uw_9km_ind))

smap_400m_mean_all_uw = sub_n_reproj(smap_400m_mean_all, kwargs_400m, window_400m, output_crs, 3)
smap_1km_mean_all_uw = sub_n_reproj(smap_1km_mean_all, kwargs_1km, window_1km, output_crs, 3)
smap_9km_mean_all_uw = sub_n_reproj(smap_9km_mean_all_dis, kwargs_1km, window_1km, output_crs, 3)

masked_smap_400m_mean_uw, mask_transform_smap_400m_mean_uw = \
    mask(dataset=smap_400m_mean_all_uw, shapes=crop_shape_uw, crop=True)
masked_smap_400m_mean_uw[np.where(masked_smap_400m_mean_uw == 0)] = np.nan
masked_smap_1km_mean_uw, mask_transform_smap_1km_mean_uw = \
    mask(dataset=smap_1km_mean_all_uw, shapes=crop_shape_uw, crop=True)
masked_smap_1km_mean_uw[np.where(masked_smap_1km_mean_uw == 0)] = np.nan
masked_smap_9km_mean_uw, mask_transform_smap_9km_mean_uw = \
    mask(dataset=smap_9km_mean_all_uw, shapes=crop_shape_uw, crop=True)
masked_smap_9km_mean_uw[np.where(masked_smap_9km_mean_uw == 0)] = np.nan

# Make the maps (Upper Washita)
feature_shp_uw = ShapelyFeature(Reader(path_shapefile_uw).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_uw = np.array(smap_400m_mean_all_uw.bounds)
extent_uw = extent_uw[[0, 2, 1, 3]]

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
for ipt in range(3):
    # 400 m
    ax = fig.add_subplot(3, 3, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_uw)
    img = ax.imshow(masked_smap_400m_mean_uw[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_uw)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 1 km
    ax = fig.add_subplot(3, 3, ipt+4, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_uw)
    img = ax.imshow(masked_smap_1km_mean_uw[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_uw)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 9 km
    ax = fig.add_subplot(3, 3, ipt+7, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_uw)
    img = ax.imshow(masked_smap_9km_mean_uw[ipt, :, :], origin='upper', vmin=0, vmax=0.25, cmap='Spectral',
               extent=extent_uw)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

plt.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=12, x=0.95)
cbar.ax.tick_params(labelsize=11)
cbar_ax.locator_params(nbins=6)
fig.text(0.12, 0.95, '08/01 - 08/08', fontsize=15, fontweight='bold')
fig.text(0.41, 0.95, '08/09 - 08/16', fontsize=15, fontweight='bold')
fig.text(0.71, 0.95, '08/17 - 08/24', fontsize=15, fontweight='bold')
fig.text(0.005, 0.75, '400 m', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.005, 0.45, '1 km', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.005, 0.15, '9 km', fontsize=15, fontweight='bold', rotation=90)
plt.show()

plt.savefig(path_results + '/sm_comp_uw.png')
plt.close()




# 1.2.5 (Little River)
path_shapefile_lr = path_gis_data + '/watershed_boundary/little_river.shp'
shapefile_lr = fiona.open(path_shapefile_lr, 'r')
crop_shape_lr = [feature["geometry"] for feature in shapefile_lr]
shp_lr_extent = list(shapefile_lr.bounds)

#Subset the region of watershed
[lat_400m_lr, row_lr_400m_ind, lon_400m_lr, col_lr_400m_ind] = \
    coordtable_subset(lat_conus_ease_400m, lon_conus_ease_400m, shp_lr_extent[3], shp_lr_extent[1], shp_lr_extent[2], shp_lr_extent[0])
[lat_1km_lr, row_lr_1km_ind, lon_1km_lr, col_lr_1km_ind] = \
    coordtable_subset(lat_conus_ease_1km, lon_conus_ease_1km, shp_lr_extent[3], shp_lr_extent[1], shp_lr_extent[2], shp_lr_extent[0])
# [lat_9km_lr, row_lr_9km_ind, lon_9km_lr, col_lr_9km_ind] = \
#     coordtable_subset(lat_conus_ease_9km, lon_conus_ease_9km, shp_lr_extent[3], shp_lr_extent[1], shp_lr_extent[2], shp_lr_extent[0])

window_400m = Window(col_lr_400m_ind[0], row_lr_400m_ind[0], len(col_lr_400m_ind), len(row_lr_400m_ind))
window_1km = Window(col_lr_1km_ind[0], row_lr_1km_ind[0], len(col_lr_1km_ind), len(row_lr_1km_ind))
# window_9km = Window(col_lr_9km_ind[0], row_lr_9km_ind[0], len(col_lr_9km_ind), len(row_lr_9km_ind))

smap_400m_mean_all_lr = sub_n_reproj(smap_400m_mean_all, kwargs_400m, window_400m, output_crs, 3)
smap_1km_mean_all_lr = sub_n_reproj(smap_1km_mean_all, kwargs_1km, window_1km, output_crs, 3)
smap_9km_mean_all_lr = sub_n_reproj(smap_9km_mean_all_dis, kwargs_1km, window_1km, output_crs, 3)

masked_smap_400m_mean_lr, mask_transform_smap_400m_mean_lr = \
    mask(dataset=smap_400m_mean_all_lr, shapes=crop_shape_lr, crop=True)
masked_smap_400m_mean_lr[np.where(masked_smap_400m_mean_lr == 0)] = np.nan
masked_smap_1km_mean_lr, mask_transform_smap_1km_mean_lr = \
    mask(dataset=smap_1km_mean_all_lr, shapes=crop_shape_lr, crop=True)
masked_smap_1km_mean_lr[np.where(masked_smap_1km_mean_lr == 0)] = np.nan
masked_smap_9km_mean_lr, mask_transform_smap_9km_mean_lr = \
    mask(dataset=smap_9km_mean_all_lr, shapes=crop_shape_lr, crop=True)
masked_smap_9km_mean_lr[np.where(masked_smap_9km_mean_lr == 0)] = np.nan

# Make the maps (Little River)
feature_shp_lr = ShapelyFeature(Reader(path_shapefile_lr).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_lr = np.array(smap_400m_mean_all_lr.bounds)
extent_lr = extent_lr[[0, 2, 1, 3]]

fig = plt.figure(figsize=(11, 12), dpi=100, facecolor='w', edgecolor='k')
for ipt in range(3):
    # 400 m
    ax = fig.add_subplot(3, 3, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_lr)
    img = ax.imshow(masked_smap_400m_mean_lr[ipt, :, :], origin='upper', vmin=0.1, vmax=0.35, cmap='Spectral',
               extent=extent_lr)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 1 km
    ax = fig.add_subplot(3, 3, ipt+4, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_lr)
    img = ax.imshow(masked_smap_1km_mean_lr[ipt, :, :], origin='upper', vmin=0.1, vmax=0.35, cmap='Spectral',
               extent=extent_lr)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 9 km
    ax = fig.add_subplot(3, 3, ipt+7, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_lr)
    img = ax.imshow(masked_smap_9km_mean_lr[ipt, :, :], origin='upper', vmin=0.1, vmax=0.35, cmap='Spectral',
               extent=extent_lr)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

plt.subplots_adjust(left=0.07, right=0.88, bottom=0.05, top=0.92, hspace=0.2, wspace=0.2)
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=12, x=0.95)
cbar.ax.tick_params(labelsize=11)
cbar_ax.locator_params(nbins=6)
fig.text(0.12, 0.95, '08/01 - 08/08', fontsize=15, fontweight='bold')
fig.text(0.4, 0.95, '08/09 - 08/16', fontsize=15, fontweight='bold')
fig.text(0.69, 0.95, '08/17 - 08/24', fontsize=15, fontweight='bold')
fig.text(0.02, 0.75, '400 m', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.02, 0.45, '1 km', fontsize=15, fontweight='bold', rotation=90)
fig.text(0.02, 0.15, '9 km', fontsize=15, fontweight='bold', rotation=90)
plt.show()

plt.savefig(path_results + '/sm_comp_lr.png')
plt.close()


