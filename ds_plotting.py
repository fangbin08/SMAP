import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
path_lmask = '/Users/binfang/Downloads/Processing/processed_data'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data'
# Path of EASE projection lat/lon tables
path_ease_coord_table = '/Volumes/MyPassport/SMAP_Project/Datasets/geolocation'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_201204'
# Path of preview
path_model_evaluation = '/Users/binfang/Documents/SMAP_Project/results/results_201204/model_evaluation'
# Path of SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
yearname = np.linspace(2015, 2020, 6, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'cellsize_1km', 'cellsize_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_ease_25km', 'lon_world_ease_25km', 'row_world_ease_9km_from_1km_ind',
                'col_world_ease_9km_from_1km_ind']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Generate land/water mask provided by GLDAS/NASA
lmask_file = open(path_ease_coord_table + '/EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin', 'r')
lmask_ease_25km = np.fromfile(lmask_file, dtype=np.dtype('uint8'))
lmask_ease_25km = np.reshape(lmask_ease_25km, [len(lat_world_ease_25km), len(lon_world_ease_25km)]).astype(float)
lmask_ease_25km[np.where(lmask_ease_25km != 0)] = np.nan
lmask_ease_25km[np.where(lmask_ease_25km == 0)] = 1
# lmask_ease_25km[np.where((lmask_ease_25km == 101) | (lmask_ease_25km == 255))] = 0
lmask_file.close()

# Find the indices of land pixels by the 25-km resolution land-ocean mask
[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind] = np.where(~np.isnan(lmask_ease_25km))


########################################################################################################################
# 1. GLDAS Model Maps

matsize_1km_init = np.empty((len(lat_world_ease_1km), len(lon_world_ease_1km)), dtype='float32')
matsize_1km_init[:] = np.nan
matsize_9km_init = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km)), dtype='float32')
matsize_9km_init[:] = np.nan
matsize_25km_init = np.empty((len(lat_world_ease_25km), len(lon_world_ease_25km)), dtype='float32')
matsize_25km_init[:] = np.nan

# 1.1 Import data for plotting GLDAS model maps
hdf_file = path_model + '/gldas/ds_model_coef_nofill.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list = list(f_read.keys())

matsize_25km_monthly_init = np.repeat(matsize_25km_init[:, :, np.newaxis], len(monthname)*2, axis=2)
r2_mat_monthly = np.copy(matsize_25km_monthly_init)
rmse_mat_monthly = np.copy(matsize_25km_monthly_init)
slope_mat_monthly = np.copy(matsize_25km_monthly_init)

for imo in range(len(monthname)*2):
    r2_mat = f_read[varname_list[24+imo]][:, :, 0]
    rmse_mat = f_read[varname_list[24+imo]][:, :, 1]
    slope_mat = f_read[varname_list[imo]][:, :, 0]
    slope_mat_monthly[:, :, imo] = slope_mat
    r2_mat_monthly[:, :, imo] = r2_mat
    rmse_mat_monthly[:, :, imo] = rmse_mat
    del(r2_mat, rmse_mat, slope_mat)


# 1.2 Single maps
# os.chdir(path_results)

xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
fig = plt.figure(num=None, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
ax.add_feature(shape_world)
img = ax.pcolormesh(xx_wrd, yy_wrd, r2_mat_monthly[:, :, 6], transform=ccrs.PlateCarree(), vmin=0, vmax=0.6, cmap='viridis')
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
plt.suptitle('SMAP SM 9 km', y=0.96, fontsize=15)
plt.show()


# 1.3 Subplot maps
# 1.3.1 R^2 of AM
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
fig = plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(2, 2, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    ax.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img = ax.pcolormesh(xx_wrd, yy_wrd, np.nanmax(r2_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                        transform=ccrs.PlateCarree(), vmin=0, vmax=0.6, cmap='hot_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(-170, -45, title_content[ipt], fontsize=14, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=6)
plt.suptitle('$\mathregular{R^2}$ (a.m.)', fontsize=16, weight='bold')
plt.savefig(path_model_evaluation + '/r2_world_am.png')
plt.close()

# 1.3.2 RMSE of AM
fig = plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(2, 2, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    ax.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img = ax.pcolormesh(xx_wrd, yy_wrd, np.nanmin(rmse_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                        transform=ccrs.PlateCarree(), vmin=0, vmax=0.08, cmap='Reds')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(-170, -45, title_content[ipt], fontsize=14, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=8)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=1.05, y=0.05, labelpad=-15)
plt.suptitle('Slope (a.m.)', fontsize=16, weight='bold')
plt.show()
plt.savefig(path_model_evaluation + '/rmse_world_am.png')
plt.close()


# 1.3.3 Slope of AM
fig = plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(2, 2, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    ax.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img = ax.pcolormesh(xx_wrd, yy_wrd, np.nanmean(slope_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                        transform=ccrs.PlateCarree(), vmin=-0.015, vmax=0.005, cmap='hot_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(-170, -45, title_content[ipt], fontsize=14, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=4)
plt.suptitle('Slope (a.m.)', fontsize=16, weight='bold')
plt.show()
plt.savefig(path_model_evaluation + '/slope_am.png')
plt.close()


# 1.3.4 Difference of metrics of AM
difference_data1 = r2_mat_monthly[:, :, 6] - r2_mat_monthly[:, :, 0]
difference_data2 = r2_mat_monthly[:, :, 18] - r2_mat_monthly[:, :, 12]
difference_data3 = rmse_mat_monthly[:, :, 6] - rmse_mat_monthly[:, :, 0]
difference_data4 = rmse_mat_monthly[:, :, 18] - rmse_mat_monthly[:, :, 12]
difference_data = np.stack((difference_data1, difference_data2, difference_data3, difference_data4))

# title_content = ['$\Delta \mathregular{R^2}$ (a.m.)', '$\Delta \mathregular{R^2}$ (p.m.)',
#                  '$\Delta$RMSE (a.m.)', '$\Delta$RMSE (p.m.)']
title_content = ['$\Delta \mathregular{R^2}$ (a.m.)', '$\Delta$RMSE (a.m.)']
fig = plt.figure(figsize=(8, 10), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85, hspace=0.25, wspace=0.2)
for ipt in range(1):
    # Delta R^2
    ax1 = fig.add_subplot(2, 1, ipt+1, projection=ccrs.PlateCarree())
    ax1.add_feature(shape_world, linewidth=0.5)
    ax1.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img1 = ax1.pcolormesh(xx_wrd, yy_wrd, difference_data[ipt, :, :], transform=ccrs.PlateCarree(),
                        vmin=-0.4, vmax=0.4, cmap='bwr')
    gl = ax1.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    cbar1 = plt.colorbar(img1, extend='both', orientation='horizontal', aspect=50, pad=0.1, shrink=0.75)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.ax.locator_params(nbins=4)
    ax1.set_title(title_content[ipt], pad=20, fontsize=16, weight='bold')

    # Delta RMSE
    ax2 = fig.add_subplot(2, 1, ipt+2, projection=ccrs.PlateCarree())
    ax2.add_feature(shape_world, linewidth=0.5)
    ax2.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img2 = ax2.pcolormesh(xx_wrd, yy_wrd, difference_data[ipt+2, :, :], transform=ccrs.PlateCarree(),
                        vmin=-0.06, vmax=0.06, cmap='bwr')
    gl = ax2.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    cbar2 = plt.colorbar(img2, extend='both', orientation='horizontal', aspect=50, pad=0.1, shrink=0.75)
    cbar2.set_label('$\mathregular{(m^3/m^3)}$', fontsize=12, x=0.95)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.ax.locator_params(nbins=6)
    ax2.set_title(title_content[ipt+1], pad=20, fontsize=16, weight='bold')

plt.suptitle('Difference (July - January)', fontsize=20, weight='bold')
plt.show()
plt.savefig(path_model_evaluation + '/delta.png')
plt.close()


########################################################################################################################
# 2 SMAP SM maps (Worldwide)

# 2.1 Composite the data of the first 16 days of one specific month
# Load in SMAP 9 km SM
year_plt = [2020]
month_plt = [1, 4, 7]
days_begin = 1
days_end = 30
days_n = days_end - days_begin + 1

matsize_9km = [len(month_plt), len(lat_world_ease_9km), len(lon_world_ease_9km)]
smap_9km_mean_1_all = np.empty(matsize_9km, dtype='float32')
smap_9km_mean_1_all[:] = np.nan
smap_9km_mean_2_all = np.copy(smap_9km_mean_1_all)

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        smap_9km_load = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), days_n*2))
        smap_9km_load[:] = np.nan
        for idt in range(days_begin-1, days_end):
            smap_9km_load[:, :, 2*idt+0] = f_read_smap_9km[varname_list_smap_9km[0]][:, :, idt] # AM
            smap_9km_load[:, :, 2*idt+1] = f_read_smap_9km[varname_list_smap_9km[1]][:, :, idt] # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(smap_9km_load[:, :, :days_n], axis=2)
        smap_9km_mean_2 = np.nanmean(smap_9km_load[:, :, days_n:], axis=2)
        del(smap_9km_load)

        smap_9km_mean_1_all[imo, :, :] = smap_9km_mean_1
        smap_9km_mean_2_all[imo, :, :] = smap_9km_mean_2
        del(smap_9km_mean_1, smap_9km_mean_2)
        print(imo)


# Load in SMAP 1 km SM
smap_1km_agg_stack = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), days_n*2))
smap_1km_agg_stack[:] = np.nan
smap_1km_mean_1_all = np.empty(matsize_9km, dtype='float32')
smap_1km_mean_1_all[:] = np.nan
smap_1km_mean_2_all = np.copy(smap_1km_mean_1_all)

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        for idt in range(days_n):  # 30 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray().astype(np.float32)

            # Aggregate to 9 km
            for ilr in range(2):
                src_tf_arr_1layer = src_tf_arr[ilr, :, :]
                smap_sm_1km_agg = np.array\
                    ([np.nanmean(src_tf_arr_1layer[row_world_ease_9km_from_1km_ind[x], :], axis=0)
                        for x in range(len(lat_world_ease_9km))])
                smap_sm_1km_agg = np.array\
                    ([np.nanmean(smap_sm_1km_agg[:, col_world_ease_9km_from_1km_ind[y]], axis=1)
                        for y in range(len(lon_world_ease_9km))])
                smap_sm_1km_agg = np.fliplr(np.rot90(smap_sm_1km_agg, 3))
                smap_1km_agg_stack[:, :, 2*idt+ilr] = smap_sm_1km_agg
                del(smap_sm_1km_agg, src_tf_arr_1layer)

            print(str_date)
            del(src_tf_arr)

        smap_1km_mean_1 = np.nanmean(smap_1km_agg_stack[:, :, :days_n], axis=2)
        smap_1km_mean_2 = np.nanmean(smap_1km_agg_stack[:, :, days_n:], axis=2)

        smap_1km_mean_1_all[imo, :, :] = smap_1km_mean_1
        smap_1km_mean_2_all[imo, :, :] = smap_1km_mean_2
        del(smap_1km_mean_1, smap_1km_mean_2)

smap_data_stack = np.stack((smap_1km_mean_1_all, smap_9km_mean_1_all, smap_1km_mean_2_all, smap_9km_mean_2_all))

# Save and load the data
with h5py.File(path_model_evaluation + '/smap_data_stack.hdf5', 'w') as f:
    f.create_dataset('smap_data_stack', data=smap_data_stack)
f.close()

# f_read = h5py.File(path_model_evaluation + '/smap_data_stack.hdf5', "r")
# smap_data_stack = f_read['smap_data_stack'][()]
# f_read.close()

# 2.2 Maps of the world
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_9km, lat_world_ease_9km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
title_content = ['1 km (Jan 2020)', '9 km (Jan 2020)', '1 km (Apr 2020)', '9 km (Apr 2020)', '1 km (Jul 2020)', '9 km (Jul 2020)']

fig = plt.figure(figsize=(12, 9), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.2, wspace=0.2)
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(3, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smap_data_stack[0, ipt, :, :], vmin=0, vmax=0.5, cmap='Spectral')
    ax.set_extent([180, -180, -70, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    # ax.set_title(title_content[ipt], pad=12, fontsize=17, weight='bold')
    ax.text(-175, -55, title_content[ipt*2], fontsize=11, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 9 km
    ax = fig.add_subplot(3, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smap_data_stack[1, ipt, :, :], vmin=0, vmax=0.5, cmap='Spectral')
    ax.set_extent([180, -180, -70, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    # ax.set_title(title_content[ipt], pad=12, fontsize=17, weight='bold')
    ax.text(-175, -55, title_content[ipt*2+1], fontsize=11, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=1.05, y=0.05, labelpad=-15)
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/sm_comp_20.png')
plt.close()

########################################################################################################################
# 3. River Basin maps
# 3.1 Sacramento-San Joaquin RB

path_shp_ssj = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_SACRAMENTO RIVER - SAN JOAQUIN RIVER'
os.chdir(path_shp_ssj)
shp_ssj_file = "Aqueduct_river_basins_SACRAMENTO RIVER - SAN JOAQUIN RIVER.shp"
shp_ssj_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_ssj_file, 0)
shp_ssj_extent = list(shp_ssj_ds.GetLayer().GetExtent())

#Load and subset the region of Sacramento-San Joaquin RB (SMAP 9 km)
[lat_9km_ssj, row_ssj_9km_ind, lon_9km_ssj, col_ssj_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_ssj_extent[3], shp_ssj_extent[2], shp_ssj_extent[1], shp_ssj_extent[0])

# Load and subset SMAP 9 km SM of Sacramento-San Joaquin RB
year_plt = [2020]
month_plt = [1, 4, 7]
days_begin = 1
days_end = 30
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
smap_9km_mean_2_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        # smap_9km_load = smap_9km_init
        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_ssj_9km_ind[0]:row_ssj_9km_ind[-1] + 1,
                                           col_ssj_9km_ind[0]:col_ssj_9km_ind[-1] + 1, :]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_ssj_9km_ind[0]:row_ssj_9km_ind[-1] + 1,
                                           col_ssj_9km_ind[0]:col_ssj_9km_ind[-1] + 1, :]  # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(smap_9km_load_1, axis=2)
        smap_9km_mean_2 = np.nanmean(smap_9km_load_2, axis=2)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)
        smap_9km_mean_2_allyear.append(smap_9km_mean_2)

        del(smap_9km_load_1, smap_9km_load_2, smap_9km_mean_1, smap_9km_mean_2)
        print(monthname[month_plt[imo]-1])

smap_9km_mean_1_allyear = np.stack(smap_9km_mean_1_allyear, axis=2)
smap_9km_mean_2_allyear = np.stack(smap_9km_mean_2_allyear, axis=2)
smap_9km_data_stack_ssj = np.concatenate([smap_9km_mean_1_allyear, smap_9km_mean_2_allyear], axis=2)
del(smap_9km_mean_1_allyear, smap_9km_mean_2_allyear)


#Load and subset the region of Sacramento-San Joaquin RB (SMAP 1 km)
[lat_1km_ssj, row_ssj_1km_ind, lon_1km_ssj, col_ssj_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_ssj_extent[3], shp_ssj_extent[2], shp_ssj_extent[1], shp_ssj_extent[0])

smap_1km_mean_1_allyear = []
smap_1km_mean_2_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        smap_1km_load_2_stack = []
        for idt in range(days_n): # 30 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_ssj_1km_ind[0]:row_ssj_1km_ind[-1]+1,
                         col_ssj_1km_ind[0]:col_ssj_1km_ind[-1]+1]
            smap_1km_load_1 = src_tf_arr[0, :, :]
            smap_1km_load_2 = src_tf_arr[1, :, :]
            smap_1km_load_1_stack.append(smap_1km_load_1)
            smap_1km_load_2_stack.append(smap_1km_load_2)
            # src_tf_arr = np.transpose(src_tf_arr, (1, 2, 0))
            # smap_1km_load[:, :, 2*idt+0] = src_tf_arr[:, :, 0]
            # smap_1km_load[:, :, 2*idt+1] = src_tf_arr[:, :, 1]

            print(str_date)
            del(src_tf_arr, smap_1km_load_1, smap_1km_load_2)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_load_2_stack = np.stack(smap_1km_load_2_stack)
        smap_1km_mean_1 = np.nanmean(smap_1km_load_1_stack, axis=0)
        smap_1km_mean_2 = np.nanmean(smap_1km_load_2_stack, axis=0)
        # smap_1km_mean_3 = np.nanmean(smap_1km_load[:, :, days_n//3*4:], axis=2)
        smap_1km_mean_1_allyear.append(smap_1km_mean_1)
        smap_1km_mean_2_allyear.append(smap_1km_mean_2)
        # smap_1km_data_stack = np.stack((smap_1km_mean_1, smap_1km_mean_2))
        # smap_1km_data_stack = np.float32(smap_1km_data_stack)
        del(smap_1km_mean_1, smap_1km_mean_2, smap_1km_load_1_stack, smap_1km_load_2_stack)

smap_1km_mean_1_allyear = np.stack(smap_1km_mean_1_allyear, axis=2)
smap_1km_mean_2_allyear = np.stack(smap_1km_mean_2_allyear, axis=2)
smap_1km_data_stack_ssj = np.concatenate([smap_1km_mean_1_allyear, smap_1km_mean_2_allyear], axis=2)
del(smap_1km_mean_1_allyear, smap_1km_mean_2_allyear)

with h5py.File(path_model_evaluation + '/smap_ssj_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_ssj', data=smap_9km_data_stack_ssj)
    f.create_dataset('smap_1km_data_stack_ssj', data=smap_1km_data_stack_ssj)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_ssj_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_ssj', 'smap_1km_data_stack_ssj']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

smap_1km_data_stack_ssj = np.transpose(smap_1km_data_stack_ssj, (2, 0, 1))
smap_9km_data_stack_ssj = np.transpose(smap_9km_data_stack_ssj, (2, 0, 1))

# Subplot maps
output_crs = 'EPSG:4326'

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_ssj_1km_all = []
for n in range(smap_1km_data_stack_ssj.shape[0]):
    sub_window_ssj_1km = Window(col_ssj_1km_ind[0], row_ssj_1km_ind[0], len(col_ssj_1km_ind), len(row_ssj_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_ssj_1km_output = sub_n_reproj(smap_1km_data_stack_ssj[n, :, :], kwargs_1km_sub, sub_window_ssj_1km, output_crs)

    masked_ds_ssj_1km, mask_transform_ds_ssj_1km = mask(dataset=smap_sm_ssj_1km_output, shapes=crop_shape_ssj, crop=True)
    masked_ds_ssj_1km[np.where(masked_ds_ssj_1km == 0)] = np.nan
    masked_ds_ssj_1km = masked_ds_ssj_1km.squeeze()

    masked_ds_ssj_1km_all.append(masked_ds_ssj_1km)

masked_ds_ssj_1km_all = np.asarray(masked_ds_ssj_1km_all)


# 9 km
masked_ds_ssj_9km_all = []
for n in range(smap_9km_data_stack_ssj.shape[0]):
    sub_window_ssj_9km = Window(col_ssj_9km_ind[0], row_ssj_9km_ind[0], len(col_ssj_9km_ind), len(row_ssj_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    smap_sm_ssj_9km_output = sub_n_reproj(smap_9km_data_stack_ssj[n, :, :], kwargs_9km_sub, sub_window_ssj_9km, output_crs)

    masked_ds_ssj_9km, mask_transform_ds_ssj_9km = mask(dataset=smap_sm_ssj_9km_output, shapes=crop_shape_ssj, crop=True)
    masked_ds_ssj_9km[np.where(masked_ds_ssj_9km == 0)] = np.nan
    masked_ds_ssj_9km = masked_ds_ssj_9km.squeeze()

    masked_ds_ssj_9km_all.append(masked_ds_ssj_9km)

masked_ds_ssj_9km_all = np.asarray(masked_ds_ssj_9km_all)
masked_ds_ssj_9km_all[masked_ds_ssj_9km_all >= 0.5] = np.nan

# Make the subplot maps
title_content = ['1 km\n(Jan 2020)', '9 km\n(Jan 2020)', '1 km\n(Apr 2020)', '9 km\n(Apr 2020)', '1 km\n(Jul 2020)', '9 km\n(Jul 2020)']
feature_shp_ssj = ShapelyFeature(Reader(path_shp_ssj + '/' + shp_ssj_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_ssj = np.array(smap_sm_ssj_1km_output.bounds)
extent_ssj = extent_ssj[[0, 2, 1, 3]]

fig = plt.figure(figsize=(7, 6), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.2)
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(2, 3, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ssj)
    # ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_ssj_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_ssj)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=2)
    gl.ylocator = mticker.MultipleLocator(base=2)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-123, 36.7, title_content[ipt*2], fontsize=8, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 9 km
    ax = fig.add_subplot(2, 3, ipt+4, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ssj)
    # ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_ssj_9km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_ssj)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=2)
    gl.ylocator = mticker.MultipleLocator(base=2)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-123, 36.7, title_content[ipt*2+1], fontsize=8, horizontalalignment='left',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='horizontal', pad=0.1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_ssj_1.png')
plt.close()


# 3.2 Ganga-Brahmaputra RB
path_shp_gb = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_GANGES - BRAHMAPUTRA'
os.chdir(path_shp_gb)
shp_gb_file = "Aqueduct_river_basins_GANGES - BRAHMAPUTRA.shp"
shp_gb_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_gb_file, 0)
shp_gb_extent = list(shp_gb_ds.GetLayer().GetExtent())

#Load and subset the region of Ganga-Brahmaputra RB (SMAP 9 km)
[lat_9km_gb, row_gb_9km_ind, lon_9km_gb, col_gb_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_gb_extent[3], shp_gb_extent[2], shp_gb_extent[1], shp_gb_extent[0])

# Load and subset SMAP 9 km SM of Ganga-Brahmaputra RB
year_plt = [2020]
month_plt = [1, 4, 7]
days_begin = 1
days_end = 30
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
smap_9km_mean_2_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        # smap_9km_load = smap_9km_init
        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_gb_9km_ind[0]:row_gb_9km_ind[-1] + 1,
                                           col_gb_9km_ind[0]:col_gb_9km_ind[-1] + 1, :]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_gb_9km_ind[0]:row_gb_9km_ind[-1] + 1,
                                           col_gb_9km_ind[0]:col_gb_9km_ind[-1] + 1, :]  # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(smap_9km_load_1, axis=2)
        smap_9km_mean_2 = np.nanmean(smap_9km_load_2, axis=2)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)
        smap_9km_mean_2_allyear.append(smap_9km_mean_2)

        del(smap_9km_load_1, smap_9km_load_2, smap_9km_mean_1, smap_9km_mean_2)
        print(monthname[month_plt[imo]-1])

smap_9km_mean_1_allyear = np.stack(smap_9km_mean_1_allyear, axis=2)
smap_9km_mean_2_allyear = np.stack(smap_9km_mean_2_allyear, axis=2)
smap_9km_data_stack_gb = np.concatenate([smap_9km_mean_1_allyear, smap_9km_mean_2_allyear], axis=2)
del(smap_9km_mean_1_allyear, smap_9km_mean_2_allyear)


#Load and subset the region of Ganga-Brahmaputra RB (SMAP 1 km)
[lat_1km_gb, row_gb_1km_ind, lon_1km_gb, col_gb_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_gb_extent[3], shp_gb_extent[2], shp_gb_extent[1], shp_gb_extent[0])

smap_1km_mean_1_allyear = []
smap_1km_mean_2_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        smap_1km_load_2_stack = []
        for idt in range(days_n): # 30 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_gb_1km_ind[0]:row_gb_1km_ind[-1]+1,
                         col_gb_1km_ind[0]:col_gb_1km_ind[-1]+1]
            smap_1km_load_1 = src_tf_arr[0, :, :]
            smap_1km_load_2 = src_tf_arr[1, :, :]
            smap_1km_load_1_stack.append(smap_1km_load_1)
            smap_1km_load_2_stack.append(smap_1km_load_2)
            # src_tf_arr = np.transpose(src_tf_arr, (1, 2, 0))
            # smap_1km_load[:, :, 2*idt+0] = src_tf_arr[:, :, 0]
            # smap_1km_load[:, :, 2*idt+1] = src_tf_arr[:, :, 1]

            print(str_date)
            del(src_tf_arr, smap_1km_load_1, smap_1km_load_2)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_load_2_stack = np.stack(smap_1km_load_2_stack)
        smap_1km_mean_1 = np.nanmean(smap_1km_load_1_stack, axis=0)
        smap_1km_mean_2 = np.nanmean(smap_1km_load_2_stack, axis=0)
        # smap_1km_mean_3 = np.nanmean(smap_1km_load[:, :, days_n//3*4:], axis=2)
        smap_1km_mean_1_allyear.append(smap_1km_mean_1)
        smap_1km_mean_2_allyear.append(smap_1km_mean_2)
        # smap_1km_data_stack = np.stack((smap_1km_mean_1, smap_1km_mean_2))
        # smap_1km_data_stack = np.float32(smap_1km_data_stack)
        del(smap_1km_mean_1, smap_1km_mean_2, smap_1km_load_1_stack, smap_1km_load_2_stack)

smap_1km_mean_1_allyear = np.stack(smap_1km_mean_1_allyear, axis=2)
smap_1km_mean_2_allyear = np.stack(smap_1km_mean_2_allyear, axis=2)
smap_1km_data_stack_gb = np.concatenate([smap_1km_mean_1_allyear, smap_1km_mean_2_allyear], axis=2)
del(smap_1km_mean_1_allyear, smap_1km_mean_2_allyear)

with h5py.File(path_model_evaluation + '/smap_gb_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_gb', data=smap_9km_data_stack_gb)
    f.create_dataset('smap_1km_data_stack_gb', data=smap_1km_data_stack_gb)
f.close()


# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_gb_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_gb', 'smap_1km_data_stack_gb']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

smap_1km_data_stack_gb = np.transpose(smap_1km_data_stack_gb, (2, 0, 1))
smap_9km_data_stack_gb = np.transpose(smap_9km_data_stack_gb, (2, 0, 1))

# Subplot maps
shapefile_gb = fiona.open(path_shp_gb + '/' + shp_gb_file, 'r')
crop_shape_gb = [feature["geometry"] for feature in shapefile_gb]
# shp_gb_extent = list(shapefile_gb.bounds)
output_crs = 'EPSG:4326'

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_gb_1km_all = []
for n in range(smap_1km_data_stack_gb.shape[0]):
    sub_window_gb_1km = Window(col_gb_1km_ind[0], row_gb_1km_ind[0], len(col_gb_1km_ind), len(row_gb_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_gb_1km_output = sub_n_reproj(smap_1km_data_stack_gb[n, :, :], kwargs_1km_sub, sub_window_gb_1km, output_crs)

    masked_ds_gb_1km, mask_transform_ds_gb_1km = mask(dataset=smap_sm_gb_1km_output, shapes=crop_shape_gb, crop=True)
    masked_ds_gb_1km[np.where(masked_ds_gb_1km == 0)] = np.nan
    masked_ds_gb_1km = masked_ds_gb_1km.squeeze()

    masked_ds_gb_1km_all.append(masked_ds_gb_1km)

masked_ds_gb_1km_all = np.asarray(masked_ds_gb_1km_all)


# 9 km
masked_ds_gb_9km_all = []
for n in range(smap_9km_data_stack_gb.shape[0]):
    sub_window_gb_9km = Window(col_gb_9km_ind[0], row_gb_9km_ind[0], len(col_gb_9km_ind), len(row_gb_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    smap_sm_gb_9km_output = sub_n_reproj(smap_9km_data_stack_gb[n, :, :], kwargs_9km_sub, sub_window_gb_9km, output_crs)

    masked_ds_gb_9km, mask_transform_ds_gb_9km = mask(dataset=smap_sm_gb_9km_output, shapes=crop_shape_gb, crop=True)
    masked_ds_gb_9km[np.where(masked_ds_gb_9km == 0)] = np.nan
    masked_ds_gb_9km = masked_ds_gb_9km.squeeze()

    masked_ds_gb_9km_all.append(masked_ds_gb_9km)

masked_ds_gb_9km_all = np.asarray(masked_ds_gb_9km_all)
# masked_ds_gb_9km_all[masked_ds_gb_9km_all >= 0.5] = np.nan


# Make the subplot maps
title_content = ['1 km\n(Jan 2020)', '9 km\n(Jan 2020)', '1 km\n(Apr 2020)', '9 km\n(Apr 2020)', '1 km\n(Jul 2020)', '9 km\n(Jul 2020)']
feature_shp_gb = ShapelyFeature(Reader(path_shp_gb + '/' + shp_gb_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_gb = np.array(smap_sm_gb_1km_output.bounds)
extent_gb = extent_gb[[0, 2, 1, 3]]

fig = plt.figure(figsize=(7, 5), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.2)
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(3, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_gb)
    # ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_gb_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_gb)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(95.4, 24.4, title_content[ipt*2], fontsize=7, horizontalalignment='center',
            verticalalignment='top', weight='bold')
    # 9 km
    ax = fig.add_subplot(3, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_gb)
    # ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_gb_9km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_gb)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(95.4, 24.4, title_content[ipt*2+1], fontsize=7, horizontalalignment='center',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='horizontal', pad=0.1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_gb_1.png')
plt.close()



# 3.3 Murray-Darling RB
path_shp_md = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_MURRAY - DARLING'
os.chdir(path_shp_md)
shp_md_file = "Aqueduct_river_basins_MURRAY - DARLING.shp"
shp_md_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_md_file, 0)
shp_md_extent = list(shp_md_ds.GetLayer().GetExtent())

# Load and subset SMAP 9 km SM of GB RB
year_plt = [2020]
month_plt = [1, 4, 7]
days_begin = 1
days_end = 30
days_n = days_end - days_begin + 1

#Load and subset the region of Murray-Darling RB (SMAP 9 km)
[lat_9km_md, row_md_9km_ind, lon_9km_md, col_md_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_md_extent[3], shp_md_extent[2], shp_md_extent[1], shp_md_extent[0])

# Load and subset SMAP 9 km SM of Murray-Darling RB
year_plt = [2020]
month_plt = [1, 4, 7]
days_begin = 1
days_end = 30
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
smap_9km_mean_2_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        # smap_9km_load = smap_9km_init
        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_md_9km_ind[0]:row_md_9km_ind[-1] + 1,
                                           col_md_9km_ind[0]:col_md_9km_ind[-1] + 1, :]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_md_9km_ind[0]:row_md_9km_ind[-1] + 1,
                                           col_md_9km_ind[0]:col_md_9km_ind[-1] + 1, :]  # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(smap_9km_load_1, axis=2)
        smap_9km_mean_2 = np.nanmean(smap_9km_load_2, axis=2)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)
        smap_9km_mean_2_allyear.append(smap_9km_mean_2)

        del(smap_9km_load_1, smap_9km_load_2, smap_9km_mean_1, smap_9km_mean_2)
        print(monthname[month_plt[imo]-1])

smap_9km_mean_1_allyear = np.stack(smap_9km_mean_1_allyear, axis=2)
smap_9km_mean_2_allyear = np.stack(smap_9km_mean_2_allyear, axis=2)
smap_9km_data_stack_md = np.concatenate([smap_9km_mean_1_allyear, smap_9km_mean_2_allyear], axis=2)
del(smap_9km_mean_1_allyear, smap_9km_mean_2_allyear)


#Load and subset the region of Murray-Darling RB (SMAP 1 km)
[lat_1km_md, row_md_1km_ind, lon_1km_md, col_md_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_md_extent[3], shp_md_extent[2], shp_md_extent[1], shp_md_extent[0])

smap_1km_mean_1_allyear = []
smap_1km_mean_2_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        smap_1km_load_2_stack = []
        for idt in range(days_n): # 30 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_md_1km_ind[0]:row_md_1km_ind[-1]+1,
                         col_md_1km_ind[0]:col_md_1km_ind[-1]+1]
            smap_1km_load_1 = src_tf_arr[0, :, :]
            smap_1km_load_2 = src_tf_arr[1, :, :]
            smap_1km_load_1_stack.append(smap_1km_load_1)
            smap_1km_load_2_stack.append(smap_1km_load_2)
            # src_tf_arr = np.transpose(src_tf_arr, (1, 2, 0))
            # smap_1km_load[:, :, 2*idt+0] = src_tf_arr[:, :, 0]
            # smap_1km_load[:, :, 2*idt+1] = src_tf_arr[:, :, 1]

            print(str_date)
            del(src_tf_arr, smap_1km_load_1, smap_1km_load_2)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_load_2_stack = np.stack(smap_1km_load_2_stack)
        smap_1km_mean_1 = np.nanmean(smap_1km_load_1_stack, axis=0)
        smap_1km_mean_2 = np.nanmean(smap_1km_load_2_stack, axis=0)
        # smap_1km_mean_3 = np.nanmean(smap_1km_load[:, :, days_n//3*4:], axis=2)
        smap_1km_mean_1_allyear.append(smap_1km_mean_1)
        smap_1km_mean_2_allyear.append(smap_1km_mean_2)
        # smap_1km_data_stack = np.stack((smap_1km_mean_1, smap_1km_mean_2))
        # smap_1km_data_stack = np.float32(smap_1km_data_stack)
        del(smap_1km_mean_1, smap_1km_mean_2, smap_1km_load_1_stack, smap_1km_load_2_stack)

smap_1km_mean_1_allyear = np.stack(smap_1km_mean_1_allyear, axis=2)
smap_1km_mean_2_allyear = np.stack(smap_1km_mean_2_allyear, axis=2)
smap_1km_data_stack_md = np.concatenate([smap_1km_mean_1_allyear, smap_1km_mean_2_allyear], axis=2)
del(smap_1km_mean_1_allyear, smap_1km_mean_2_allyear)

with h5py.File(path_model_evaluation + '/smap_md_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_md', data=smap_9km_data_stack_md)
    f.create_dataset('smap_1km_data_stack_md', data=smap_1km_data_stack_md)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_md_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_md', 'smap_1km_data_stack_md']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

smap_1km_data_stack_md = np.transpose(smap_1km_data_stack_md, (2, 0, 1))
smap_9km_data_stack_md = np.transpose(smap_9km_data_stack_md, (2, 0, 1))

# Subplot maps
# Load in watershed shapefile boundaries
shapefile_md = fiona.open(path_shp_md + '/' + shp_md_file, 'r')
crop_shape_md = [feature["geometry"] for feature in shapefile_md]
# shp_md_extent = list(shapefile_md.bounds)

output_crs = 'EPSG:4326'

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_md_1km_all = []
for n in range(smap_1km_data_stack_md.shape[0]):
    sub_window_md_1km = Window(col_md_1km_ind[0], row_md_1km_ind[0], len(col_md_1km_ind), len(row_md_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_md_1km_output = sub_n_reproj(smap_1km_data_stack_md[n, :, :], kwargs_1km_sub, sub_window_md_1km, output_crs)

    masked_ds_md_1km, mask_transform_ds_md_1km = mask(dataset=smap_sm_md_1km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_1km[np.where(masked_ds_md_1km == 0)] = np.nan
    masked_ds_md_1km = masked_ds_md_1km.squeeze()

    masked_ds_md_1km_all.append(masked_ds_md_1km)

masked_ds_md_1km_all = np.asarray(masked_ds_md_1km_all)


# 9 km
masked_ds_md_9km_all = []
for n in range(smap_9km_data_stack_md.shape[0]):
    sub_window_md_9km = Window(col_md_9km_ind[0], row_md_9km_ind[0], len(col_md_9km_ind), len(row_md_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    smap_sm_md_9km_output = sub_n_reproj(smap_9km_data_stack_md[n, :, :], kwargs_9km_sub, sub_window_md_9km, output_crs)

    masked_ds_md_9km, mask_transform_ds_md_9km = mask(dataset=smap_sm_md_9km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_9km[np.where(masked_ds_md_9km == 0)] = np.nan
    masked_ds_md_9km = masked_ds_md_9km.squeeze()

    masked_ds_md_9km_all.append(masked_ds_md_9km)

masked_ds_md_9km_all = np.asarray(masked_ds_md_9km_all)
# masked_ds_md_9km_all[masked_ds_md_9km_all >= 0.5] = np.nan


# Make the subplot maps
title_content = ['1 km\n(Jan 2020)', '9 km\n(Jan 2020)', '1 km\n(Apr 2020)', '9 km\n(Apr 2020)', '1 km\n(Jul 2020)', '9 km\n(Jul 2020)']
feature_shp_md = ShapelyFeature(Reader(path_shp_md + '/' + shp_md_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_md = np.array(smap_sm_md_1km_output.bounds)
extent_md = extent_md[[0, 2, 1, 3]]

fig = plt.figure(figsize=(5, 5), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.03, right=0.88, bottom=0.05, top=0.95, hspace=0.2, wspace=0.2)
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(3, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_md)
    # ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_md_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_md)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(142, -25, title_content[ipt*2], fontsize=7, horizontalalignment='center',
            verticalalignment='top', weight='bold')
    # 9 km
    ax = fig.add_subplot(3, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_md)
    # ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_md_9km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_md)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(142, -25, title_content[ipt*2+1], fontsize=7, horizontalalignment='center',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='vertical', pad=0.1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=1.05, labelpad=-15, rotation=0)
plt.savefig(path_results + '/sm_comp_md_1.png')
plt.close()


# 3.4 Danube RB
# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_dan_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_dan', 'smap_1km_data_stack_dan']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

smap_1km_data_stack_dan = np.transpose(smap_1km_data_stack_dan, (2, 0, 1))
smap_9km_data_stack_dan = np.transpose(smap_9km_data_stack_dan, (2, 0, 1))

path_shp_dan = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_DANUBE'
os.chdir(path_shp_dan)
shp_dan_file = "Aqueduct_river_basins_DANUBE.shp"
shp_dan_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_dan_file, 0)
shp_dan_extent = list(shp_dan_ds.GetLayer().GetExtent())


# Load and subset SMAP 9 km SM of Danube RB
year_plt = [2020]
month_plt = [1, 4, 7]
days_begin = 1
days_end = 30
days_n = days_end - days_begin + 1

#Load and subset the region of Danube RB (SMAP 9 km)
[lat_9km_dan, row_dan_9km_ind, lon_9km_dan, col_dan_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

# Load and subset SMAP 9 km SM of Danube RB
year_plt = [2020]
month_plt = [1, 4, 7]
days_begin = 1
days_end = 30
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
smap_9km_mean_2_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        # smap_9km_load = smap_9km_init
        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_dan_9km_ind[0]:row_dan_9km_ind[-1] + 1,
                                           col_dan_9km_ind[0]:col_dan_9km_ind[-1] + 1, :]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_dan_9km_ind[0]:row_dan_9km_ind[-1] + 1,
                                           col_dan_9km_ind[0]:col_dan_9km_ind[-1] + 1, :]  # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(smap_9km_load_1, axis=2)
        smap_9km_mean_2 = np.nanmean(smap_9km_load_2, axis=2)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)
        smap_9km_mean_2_allyear.append(smap_9km_mean_2)

        del(smap_9km_load_1, smap_9km_load_2, smap_9km_mean_1, smap_9km_mean_2)
        print(monthname[month_plt[imo]-1])

smap_9km_mean_1_allyear = np.stack(smap_9km_mean_1_allyear, axis=2)
smap_9km_mean_2_allyear = np.stack(smap_9km_mean_2_allyear, axis=2)
smap_9km_data_stack_dan = np.concatenate([smap_9km_mean_1_allyear, smap_9km_mean_2_allyear], axis=2)
del(smap_9km_mean_1_allyear, smap_9km_mean_2_allyear)


#Load and subset the region of Danube RB (SMAP 1 km)
[lat_1km_dan, row_dan_1km_ind, lon_1km_dan, col_dan_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

smap_1km_mean_1_allyear = []
smap_1km_mean_2_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        smap_1km_load_2_stack = []
        for idt in range(days_n): # 30 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_dan_1km_ind[0]:row_dan_1km_ind[-1]+1,
                         col_dan_1km_ind[0]:col_dan_1km_ind[-1]+1]
            smap_1km_load_1 = src_tf_arr[0, :, :]
            smap_1km_load_2 = src_tf_arr[1, :, :]
            smap_1km_load_1_stack.append(smap_1km_load_1)
            smap_1km_load_2_stack.append(smap_1km_load_2)
            # src_tf_arr = np.transpose(src_tf_arr, (1, 2, 0))
            # smap_1km_load[:, :, 2*idt+0] = src_tf_arr[:, :, 0]
            # smap_1km_load[:, :, 2*idt+1] = src_tf_arr[:, :, 1]

            print(str_date)
            del(src_tf_arr, smap_1km_load_1, smap_1km_load_2)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_load_2_stack = np.stack(smap_1km_load_2_stack)
        smap_1km_mean_1 = np.nanmean(smap_1km_load_1_stack, axis=0)
        smap_1km_mean_2 = np.nanmean(smap_1km_load_2_stack, axis=0)
        # smap_1km_mean_3 = np.nanmean(smap_1km_load[:, :, days_n//3*4:], axis=2)
        smap_1km_mean_1_allyear.append(smap_1km_mean_1)
        smap_1km_mean_2_allyear.append(smap_1km_mean_2)
        # smap_1km_data_stack = np.stack((smap_1km_mean_1, smap_1km_mean_2))
        # smap_1km_data_stack = np.float32(smap_1km_data_stack)
        del(smap_1km_mean_1, smap_1km_mean_2, smap_1km_load_1_stack, smap_1km_load_2_stack)

smap_1km_mean_1_allyear = np.stack(smap_1km_mean_1_allyear, axis=2)
smap_1km_mean_2_allyear = np.stack(smap_1km_mean_2_allyear, axis=2)
smap_1km_data_stack_dan = np.concatenate([smap_1km_mean_1_allyear, smap_1km_mean_2_allyear], axis=2)
del(smap_1km_mean_1_allyear, smap_1km_mean_2_allyear)

with h5py.File(path_model_evaluation + '/smap_dan_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_dan', data=smap_9km_data_stack_dan)
    f.create_dataset('smap_1km_data_stack_dan', data=smap_1km_data_stack_dan)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_dan_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_dan', 'smap_1km_data_stack_dan']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

smap_1km_data_stack_dan = np.transpose(smap_1km_data_stack_dan, (2, 0, 1))
smap_9km_data_stack_dan = np.transpose(smap_9km_data_stack_dan, (2, 0, 1))

# Subplot maps
# Load in watershed shapefile boundaries
shapefile_dan = fiona.open(path_shp_dan + '/' + shp_dan_file, 'r')
crop_shape_dan = [feature["geometry"] for feature in shapefile_dan]
shp_dan_extent = list(shapefile_dan.bounds)

output_crs = 'EPSG:4326'

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_dan_1km_all = []
for n in range(smap_1km_data_stack_dan.shape[0]):
    sub_window_dan_1km = Window(col_dan_1km_ind[0], row_dan_1km_ind[0], len(col_dan_1km_ind), len(row_dan_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_dan_1km_output = sub_n_reproj(smap_1km_data_stack_dan[n, :, :], kwargs_1km_sub, sub_window_dan_1km, output_crs)

    masked_ds_dan_1km, mask_transform_ds_dan_1km = mask(dataset=smap_sm_dan_1km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_1km[np.where(masked_ds_dan_1km == 0)] = np.nan
    masked_ds_dan_1km = masked_ds_dan_1km.squeeze()

    masked_ds_dan_1km_all.append(masked_ds_dan_1km)

masked_ds_dan_1km_all = np.asarray(masked_ds_dan_1km_all)


# 9 km
masked_ds_dan_9km_all = []
for n in range(smap_9km_data_stack_dan.shape[0]):
    sub_window_dan_9km = Window(col_dan_9km_ind[0], row_dan_9km_ind[0], len(col_dan_9km_ind), len(row_dan_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    smap_sm_dan_9km_output = sub_n_reproj(smap_9km_data_stack_dan[n, :, :], kwargs_9km_sub, sub_window_dan_9km, output_crs)

    masked_ds_dan_9km, mask_transform_ds_dan_9km = mask(dataset=smap_sm_dan_9km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_9km[np.where(masked_ds_dan_9km == 0)] = np.nan
    masked_ds_dan_9km = masked_ds_dan_9km.squeeze()

    masked_ds_dan_9km_all.append(masked_ds_dan_9km)

masked_ds_dan_9km_all = np.asarray(masked_ds_dan_9km_all)
# masked_ds_dan_9km_all[masked_ds_dan_9km_all >= 0.5] = np.nan


# Make the subplot maps
title_content = ['1 km\n(Jan 2020)', '9 km\n(Jan 2020)', '1 km\n(Apr 2020)', '9 km\n(Apr 2020)', '1 km\n(Jul 2020)', '9 km\n(Jul 2020)']
feature_shp_dan = ShapelyFeature(Reader(path_shp_dan + '/' + shp_dan_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_dan = np.array(smap_sm_dan_1km_output.bounds)
extent_dan = extent_dan[[0, 2, 1, 3]]

fig = plt.figure(figsize=(7, 5), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.2)
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(3, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_dan)
    # ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_dan_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_dan)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(11, 44, title_content[ipt*2], fontsize=7, horizontalalignment='center',
            verticalalignment='top', weight='bold')
    # 9 km
    ax = fig.add_subplot(3, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_dan)
    # ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_dan_9km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_dan)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(11, 44, title_content[ipt*2+1], fontsize=7, horizontalalignment='center',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='horizontal', pad=0.1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_dan_1.png')
plt.close()


########################################################################################################################
# 4. Scatter plots
# 4.1 Select geograhical locations by using index tables, and plot delta T - SM relationship lines through each NDVI class

# Lat/lon of the locations in the world:

lat_slc = [32.34, -35.42, 40.33, 34.94, 40.88948, 55.8776, 50.5149, 51.38164]
lon_slc = [87.03, 146.2, -5.04, -97.65, 25.8522, 9.2683, 6.37559, -106.41583]
name_slc = ['CTP', 'OZNET', 'REMEDHUS', 'SOILSCAPE(Oklahoma)', 'GROW(47emqp81)',
            'HOBE(3.08)', 'TERENO(Schoeneseiffen)', 'RISMA(SK4)']
ndvi_class = np.linspace(0, 1, 11)
viridis_r = plt.cm.get_cmap('viridis_r', 10)

row_25km_ind_sub = []
col_25km_ind_sub = []
for ico in range(len(lat_slc)):
    row_dist = np.absolute(lat_slc[ico] - lat_world_ease_25km)
    row_match = np.argmin(row_dist)
    col_dist = np.absolute(lon_slc[ico] - lon_world_ease_25km)
    col_match = np.argmin(col_dist)
    # ind = np.intersect1d(row_match, col_match)[0]
    row_25km_ind_sub.append(row_match)
    col_25km_ind_sub.append(col_match)
    del(row_dist, row_match, col_dist, col_match)

row_25km_ind_sub = np.asarray(row_25km_ind_sub)
col_25km_ind_sub = np.asarray(col_25km_ind_sub)


hdf_file = path_model + '/gldas/ds_model_coef_nofill.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list = list(f_read.keys())

r_sq_all = []
rmse_all = []
for x in range(len(row_25km_ind_sub)):
    r_sq = f_read[varname_list[30]][row_25km_ind_sub[x], col_25km_ind_sub[x], ::2]
    rmse = f_read[varname_list[30]][row_25km_ind_sub[x], col_25km_ind_sub[x], 1::2]
    r_sq_all.append(r_sq)
    rmse_all.append(rmse)
r_sq_all = np.asarray(r_sq_all)
rmse_all = np.asarray(rmse_all)
metric_all = [r_sq_all, rmse_all]
metric_all = np.asarray(metric_all)
np.savetxt(path_model_evaluation + '/regression_metric.csv', r_sq_all, delimiter=",", fmt='%f')

# Extract the indexes for the arrays to make scatter plots
coord_25km_ind = [np.intersect1d(np.where(col_lmask_ease_25km_ind == col_25km_ind_sub[x]),
                                np.where(row_lmask_ease_25km_ind == row_25km_ind_sub[x]))[0]
                  for x in np.arange(len(row_25km_ind_sub))]

# Load in data
os.chdir(path_model)
hdf_file = path_model + '/gldas/ds_model_07.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list = list(f_read.keys())
hdf_file_2 = path_model + '/gldas/ds_model_01.hdf5'
f_read_2 = h5py.File(hdf_file_2, "r")
varname_list_2 = list(f_read_2.keys())

lst_am_delta = np.array([f_read[varname_list[0]][x, :] for x in coord_25km_ind])
lst_pm_delta = np.array([f_read[varname_list[1]][x, :] for x in coord_25km_ind])
ndvi = np.array([f_read[varname_list[2]][x, :] for x in coord_25km_ind])
sm_am = np.array([f_read[varname_list[3]][x, :] for x in coord_25km_ind])
sm_pm = np.array([f_read[varname_list[4]][x, :] for x in coord_25km_ind])

# lst_am_delta_2 = np.array([f_read_2[varname_list_2[0]][x, :] for x in coord_25km_ind])
# lst_pm_delta_2 = np.array([f_read_2[varname_list_2[1]][x, :] for x in coord_25km_ind])
# ndvi_2 = np.array([f_read_2[varname_list_2[2]][x, :] for x in coord_25km_ind])
# sm_am_2 = np.array([f_read_2[varname_list_2[3]][x, :] for x in coord_25km_ind])
# sm_pm_2 = np.array([f_read_2[varname_list_2[4]][x, :] for x in coord_25km_ind])
#
# # Replace the OZNET data from July by January
# lst_am_delta[1, :] = lst_am_delta_2[1, :]
# lst_pm_delta[1, :] = lst_pm_delta_2[1, :]
# ndvi[1, :] = ndvi_2[1, :]
# sm_am[1, :] = sm_am_2[1, :]
# sm_pm[1, :] = sm_pm_2[1, :]

# Subplots of GLDAS SM vs. LST difference
# 4.1.1.
fig = plt.figure(figsize=(11, 8), dpi=150)
plt.subplots_adjust(left=0.1, right=0.85, bottom=0.13, top=0.93, hspace=0.25, wspace=0.25)
for i in range(4):
    x = sm_am[i, :]
    y = lst_am_delta[i, :]
    c = ndvi[i, :]

    ax = fig.add_subplot(2, 2, i+1)
    sc = ax.scatter(x, y, c=c, marker='o', s=3, cmap='viridis_r')
    sc.set_clim(vmin=0, vmax=0.7)
    sc.set_label('NDVI')

    # Plot by NDVI classes
    for n in range(len(ndvi_class)-1):
        ndvi_ind = np.array([np.where((c>=ndvi_class[n]) & (c<ndvi_class[n+1]))[0]])
        if ndvi_ind.shape[1] > 10:
            coef, intr = np.polyfit(x[ndvi_ind].squeeze(), y[ndvi_ind].squeeze(), 1)
            ax.plot(x[ndvi_ind].squeeze(), intr + coef * x[ndvi_ind].squeeze(), '-', color=viridis_r.colors[n])
        else:
            pass

    plt.xlim(0, 0.4)
    ax.set_xticks(np.arange(0, 0.5, 0.1))
    plt.ylim(0, 40)
    ax.set_yticks(np.arange(0, 50, 10))
    ax.text(0.02, 35, name_slc[i],fontsize=12)
    plt.grid(linestyle='--')
    # cbar = plt.colorbar(sc, extend='both')
    # cbar.set_label('NDVI')
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(sc, cax=cbar_ax, extend='both', pad=0.1, orientation='vertical')
cbar_ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=8)
cbar.set_label('NDVI', fontsize=10, labelpad=10)
fig.text(0.5, 0.03, 'Soil Moisture ($\mathregular{m^3/m^3)}$', ha='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.5, 'Temperature Difference (K)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
plt.savefig(path_model_evaluation + '/gldas_comp_1.png')
plt.close()

