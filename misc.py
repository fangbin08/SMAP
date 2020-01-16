import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import fiona
import rasterio
import rasterio.mask
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import h5py
import calendar
import datetime
import glob
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# # Load in variables
# f = h5py.File("parameters.hdf5", "r")
# varname_list = list(f.keys())
#
# for x in range(len(varname_list)):
#     var_obj = f[varname_list[x]][()]
#     exec(varname_list[x] + '= var_obj')
# f.close()

####################################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_CONUS/codes_py'
# Path of source LTDR NDVI data
path_ltdr = '/Volumes/MyPassport/SMAP_Project/Datasets/LTDR/Ver5'
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

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '1981-01-01'
end_date = '2018-12-31'
year = 2018 - 1981 + 1

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
date_seq_doy = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(date_str.strftime('%Y%m%d'))
    date_seq_doy.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))


####################################################################################################################################
# 1. Check the completeness of downloaded GLDAS files
os.chdir(path_ltdr)
files = sorted(glob.glob('*.nc4'))
# date_seq_late = date_seq[6939:]
files_group = []
for idt in range(13879):
    # files_group_1day = [files.index(i) for i in files if 'A' + date_seq_late[idt] in i]
    files_group_1day = [files.index(i) for i in files if 'A' + date_seq[idt] in i]
    files_group.append(files_group_1day)
    print(idt)

ldasfile_miss = []
for idt in range(len(files_group)):
    if len(files_group[idt]) != 8:
        ldasfile_miss.append(date_seq[idt])
        print(date_seq[idt])
        # file_miss.append(date_seq_late[idt])
        # print(date_seq_late[idt])
        print(len(files_group[idt]))
    else:
        pass


####################################################################################################################################

# 2. Check the completeness of downloaded MODIS files
# os.chdir(path_modis + lst_folder + '2019/05/')
os.chdir('/Volumes/MyPassport/SMAP_Project/NewData/SMAP/2019')
# Downloaded files
files = sorted(glob.glob('*.h5'))

# List of files to download
os.chdir('/Users/binfang/Downloads/bash_codes/')
with open("3365467267-download.txt", "r") as ins:
    url_list = []
    for line in ins:
        url_list.append(line)
ins.close()

len(files) == len(url_list)

files_list = [url_list[x][:-1].split('/')[-1] for x in range(len(url_list))]
modfile_miss_ind = [files_list.index(i) for i in files_list if i not in files]
modfile_miss_url = [url_list[modfile_miss_ind[x]] for x in range(len(modfile_miss_ind))]

os.chdir('/Users/binfang/Downloads/bash_codes/missed')
file = open('smap_2019_miss.txt', 'w')
with open('smap_2019_miss.txt', 'w') as f:
    for item in modfile_miss_url:
        f.write("%s" % item)
f.close()




with fiona.open(path_shp_dan + '/Aqueduct_river_basins_DANUBE.shp', 'r') as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open("/Users/binfang/Downloads/Processing/Model_Input/MYD11A1/2019/modis_lst_1km_2019175.tif") as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = src.meta

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
    dest.write(out_image)

map_extent = [lon_world_min, lon_world_max, lat_world_min, lat_world_max]
x_wrd = np.linspace(0, lon_world_max, len(lon_world_ease_9km))
y_wrd = np.linspace(0, lat_world_max, len(lat_world_ease_9km))
y_wrd = y_wrd[::-1]
xx_wrd, yy_wrd = np.meshgrid(x_wrd, y_wrd)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.pcolormesh(xx_wrd, yy_wrd, smap_9km_mean_1, transform=ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray'}
gl.xlabel_style = {'color': 'red', 'weight': 'bold'}


map_wrd_mesh = plt.pcolormesh(xx_wrd, yy_wrd, smap_9km_mean_1, vmin=0, vmax=0.5, cmap='gist_earth_r')
# ax.set_extent(map_extent)
ax.imshow(smap_9km_mean_1, origin='upper', transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])




xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.pcolormesh(xx_wrd, yy_wrd, r2_mat_monthly[:,:,6], transform=ccrs.PlateCarree(), vmin=0, vmax=0.5, cmap='gist_earth_r')
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth = 0.5, alpha=0.5, color='black')
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER



xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km)

fig = plt.figure(figsize=(9, 4), frameon=True)
ax = fig.add_axes([0.08, 0.05, 0.8, 0.94], projection=ccrs.LambertConformal())
ax.set_extent([lon_world_min, lon_world_max, lat_world_min, lat_world_max], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
# gl = ax.gridlines(linestyle='--', linewidth = 0.5, alpha=0.5, color='black')
# gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
# gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# plt.pcolormesh(xx_wrd, yy_wrd, r2_mat_monthly[:,:,6], transform=ccrs.PlateCarree())

fig.canvas.draw()

# Define gridline locations and draw the lines using cartopy's built-in gridliner:
xticks = [-180, -90, 0, 90, 180]
yticks = [-90, -45, 0, 45, 90]
ax.gridlines(xlocs=xticks, ylocs=yticks)
# Label the end-points of the gridlines using the custom tick makers:
ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
lambert_xticks(ax, xticks)
lambert_yticks(ax, yticks)




# Create a Lambert Conformal projection:

# Draw a set of axes with coastlines:
fig = plt.figure(figsize=(9, 4), frameon=True)
ax = fig.add_axes([0.08, 0.05, 0.8, 0.94], projection=ccrs.LambertCylindrical())
ax.set_extent([-179, 180, 0, 90], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m')

# *must* call draw in order to get the axis boundary used to add ticks:
fig.canvas.draw()

# Define gridline locations and draw the lines using cartopy's built-in gridliner:
xticks = [-180, -90, 0, 90, 180]
yticks = [-90, -45, 0, 45, 90]
ax.gridlines(xlocs=xticks, ylocs=yticks)

# Label the end-points of the gridlines using the custom tick makers:
ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
lambert_xticks(ax, xticks)
lambert_yticks(ax, yticks)







from rasterio.mask import mask
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, Affine
from rasterio.windows import Window
from rasterio.crs import CRS
import gdal




# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
path_shp_dan = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_DANUBE'
shp_dan_file = "Aqueduct_river_basins_DANUBE.shp"
# shp_dan_ds = ogr.GetDriverByName("ESRI Shapefile").Open(path_shp_dan + '/' + shp_dan_file, 0)
# shp_dan_extent = list(shp_dan_ds.GetLayer().GetExtent())

shapefile = fiona.open(path_shp_dan + '/' + shp_dan_file, 'r')
crop_shape = [feature["geometry"] for feature in shapefile]
shp_dan_extent = list(shapefile.bounds)

#Subset the region of Danube RB (1 km)
[lat_1km_dan, row_dan_1km_ind, lon_1km_dan, col_dan_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_dan_extent[3], shp_dan_extent[1], shp_dan_extent[2], shp_dan_extent[0])

smap_sm_1km_file_path = '/Users/binfang/Downloads/Processing/Downscale/2019/smap_sm_1km_ds_2019270.tif'
smap_sm_1km_ds = rasterio.open(smap_sm_1km_file_path)
smap_sm_dan_1km_sub = smap_sm_1km_ds.read()[0, row_dan_1km_ind[0]:row_dan_1km_ind[-1]+1,
                         col_dan_1km_ind[0]:col_dan_1km_ind[-1]+1]


# Get the georeference and bounding parameters of subset image
window = Window(col_dan_1km_ind[0], row_dan_1km_ind[0], len(col_dan_1km_ind), len(row_dan_1km_ind))
kwargs = smap_sm_1km_ds.meta.copy()
kwargs.update({
    'height': window.height,
    'width': window.width,
    'count': 1,
    'transform': rasterio.windows.transform(window, smap_sm_1km_ds.transform)})
smap_sm_1km_subset = MemoryFile().open(**kwargs)

mat_sub = smap_sm_1km_ds.read(1, window=window)
mat_sub = np.expand_dims(mat_sub, axis=0)
smap_sm_1km_subset.write(mat_sub)



# Define the projection as WGS84

# transform = from_origin(shp_dan_extent[0], shp_dan_extent[3], cellsize_1km, cellsize_1km)
# reproj_data = MemoryFile().open('GTiff', width=smap_sm_1km.shape[1], height=smap_sm_1km.shape[0],
#                                 dtype=str(smap_sm_1km.dtype), count=1, crs='EPSG:4326', transform=transform)
# reproj_data.write(smap_sm_1km, 1)

# reproj_out = rasterio.open('/Users/binfang/Downloads/raster.tif', 'w', 'GTiff',
#                            width=(lon_1km_dan[-1] - lon_1km_dan[0])//cellsize_1km,
#                            height=(lat_1km_dan[0] - lat_1km_dan[-1])//cellsize_1km,
#                            dtype=str(smap_sm_1km.dtype), count=1, crs='EPSG:4326', transform=transform)
# reproj_out.write(smap_sm_1km, 1)
# reproj_out.close()




## Reproject a dataset
dst_crs = 'EPSG:4326'

# src = rasterio.open(smap_sm_1km_file_path)
transform, width, height = calculate_default_transform(smap_sm_1km_subset.crs, dst_crs, smap_sm_1km_subset.width,
                                                       smap_sm_1km_subset.height, *smap_sm_1km_subset.bounds)
kwargs = smap_sm_1km_subset.meta.copy()
kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

dst = MemoryFile().open(**kwargs)
reproject(source=rasterio.band(smap_sm_1km_subset, 1), destination=rasterio.band(dst, 1),
          src_transform=smap_sm_1km_subset.transform, src_crs=smap_sm_1km_subset.crs,
          dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)



# # Path of GIS data
# path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# path_shp_dan = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_DANUBE'
#
# with fiona.open(path_shp_dan + '/Aqueduct_river_basins_DANUBE.shp', 'r') as shapefile:
#     crop_shape = [feature["geometry"] for feature in shapefile]

# smap_sm_1km_file_path = '/Users/binfang/Downloads/Processing/Downscale/2019/smap_sm_1km_ds_2019270.tif'
# smap_sm_1km = rasterio.open(smap_sm_1km_file_path)

# #Subset the region of Danube RB (9 km)
# [lat_9km_dan, row_dan_9km_ind, lon_9km_dan, col_dan_9km_ind] = \
#     coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

masked, mask_transform = mask(dataset=dst, shapes=crop_shape, crop=True)


reproj_out = rasterio.open('/Users/binfang/Downloads/raster.tif', 'w', 'GTiff',
                           height=dst.height, width=dst.width, count=1, dtype='float32',
                           crs='EPSG:4326', transform=dst.transform)
reproj_out.write(dst.read().squeeze(), 1)
reproj_out.close()




########################################################################################################################
# Function 2. Subset and reproject the Geotiff data to WGS84 projection

def sub_n_reproj(input_mat, kwargs_sub, sub_window, output_crs):
    # Get the georeference and bounding parameters of subset image
    # kwargs_sub = input_ds.meta.copy()
    kwargs_sub.update({
        'height': sub_window.height,
        'width': sub_window.width,
        'transform': rasterio.windows.transform(sub_window, kwargs_sub['transform'])})

    # Generate subset rasterio dataset
    input_ds_subset = MemoryFile().open(**kwargs_sub)
    # mat_sub = input_ds.read(1, window=sub_window)
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
    reproject(source=rasterio.band(smap_sm_1km_subset, 1), destination=rasterio.band(output_ds, 1),
              src_transform=smap_sm_1km_subset.transform, src_crs=smap_sm_1km_subset.crs,
              dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)

    return output_ds


########################################################################################################################


output_crs = 'EPSG:4326'

sub_window_dan_1km = Window(col_dan_1km_ind[0], row_dan_1km_ind[0], len(col_dan_1km_ind), len(row_dan_1km_ind))
kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                  'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                  'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
smap_sm_dan_1km_output = sub_n_reproj(smap_sm_dan_1km_sub, kwargs_1km_sub, sub_window_dan_1km, output_crs)

masked_ds_dan_1km, mask_transform_ds_dan_1km = mask(dataset=smap_sm_dan_1km_output, shapes=crop_shape, crop=True)
masked_ds_dan_1km[np.where(masked_ds_dan_1km == 0)] = np.nan



sub_window_dan_9km = Window(col_dan_9km_ind[0], row_dan_9km_ind[0], len(col_dan_9km_ind), len(row_dan_9km_ind))
kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                  'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                  'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
smap_sm_dan_9km_output = sub_n_reproj(smap_9km_data_stack[0, :, :], kwargs_9km_sub, sub_window_dan_9km, output_crs)

masked_ds_dan_9km, mask_transform_ds_dan_9km = mask(dataset=smap_sm_dan_9km_output, shapes=crop_shape, crop=True)
masked_ds_dan_9km[np.where(masked_ds_dan_9km == 0)] = np.nan




feature_shp_dan = ShapelyFeature(Reader(path_shp_dan + '/' + shp_dan_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_dan = np.array(smap_sm_dan_1km_output.bounds)
extent_dan = extent_dan[[0, 2, 1, 3]]

fig = plt.figure(num=None, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(feature_shp_dan)
img = ax.imshow(masked_ds_dan_1km.squeeze(), origin='upper', cmap='gist_earth_r',
           extent=extent_dan)
plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
gl.xlocator = mticker.MultipleLocator(base=5)
gl.ylocator = mticker.MultipleLocator(base=3)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER




fig = plt.figure(num=None, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(feature_shp_dan)
img = ax.imshow(masked_ds_dan_9km.squeeze(), origin='upper', cmap='gist_earth_r',
           extent=extent_dan)
plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
gl.xlocator = mticker.MultipleLocator(base=5)
gl.ylocator = mticker.MultipleLocator(base=3)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER











hdf_files = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/HDF_Data/MYD11A1/2018/09/MYD11A1.A2018265.h13v09.006.2018333153325.hdf'

hdf_ds = gdal.Open(hdf_files, gdal.GA_ReadOnly)
# Loop read data of specified bands from subdataset_id
size_1dim = gdal.Open(hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly).ReadAsArray().astype(np.int16).shape





# Composite the first 16 days of one specific month
# Load in SMAP 9 km SM
year_plt = [yearname[4]]
month_plt = [4, 8]
days_begin = 1
days_end = 16
days_n = days_end - days_begin + 1

matsize_9km = [2, len(lat_world_ease_9km), len(lon_world_ease_9km)]
smap_9km_mean_1_all = np.empty(matsize_9km, dtype='float32')
smap_9km_mean_1_all[:] = np.nan
smap_9km_mean_2_all = np.copy(smap_9km_mean_1_all)

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_procdata + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
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
        for idt in range(days_n):  # 16 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap_sm_ds + '/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
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






















