import os
os.environ['PROJ_LIB'] = '/Users/binfang/anaconda3/pkgs/proj4-5.0.1-h1de35cc_0/share/proj'
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import numpy as np
import h5py
import datetime
import gdal
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
path_modis_ip = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Model_Input'
# Path of SM model output
path_model_op = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP_ds/Model_Output'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP_ds/Downscale'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_CONUS/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_CONUS/results/results_191202'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of downscaled SM
path_smap_sm_ds = '/Users/binfang/Downloads/Processing/Downscale'


lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_ease_25km', 'lon_world_ease_25km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Generate land/water mask provided by GLDAS/NASA
os.chdir(path_lmask)
lmask_file = open('EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin', 'r')
lmask_ease_25km = np.fromfile(lmask_file, dtype=np.dtype('uint8'))
lmask_ease_25km = np.reshape(lmask_ease_25km, [len(lat_world_ease_25km), len(lon_world_ease_25km)]).astype(float)
lmask_ease_25km[np.where(lmask_ease_25km != 0)] = np.nan
lmask_ease_25km[np.where(lmask_ease_25km == 0)] = 1
lmask_file.close()
# Obtain the indices of land pixels by the 25-km resolution land-ocean mask
[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind] = np.where(~np.isnan(lmask_ease_25km))


########################################################################################################################
# 1. Maps

matsize_1km_init = np.empty((len(lat_world_ease_1km), len(lon_world_ease_1km)), dtype='float32')
matsize_1km_init[:] = np.nan
matsize_9km_init = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km)), dtype='float32')
matsize_9km_init[:] = np.nan
matsize_25km_init = np.empty((len(lat_world_ease_25km), len(lon_world_ease_25km)), dtype='float32')
matsize_25km_init[:] = np.nan


# 1.1 SM model maps
hdf_file = path_procdata + '/ds_model_coef_nofill.hdf5'
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

# Build the map file
map_wrd = Basemap(projection='cea',llcrnrlat=lat_world_min,urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min,urcrnrlon=lon_world_max,resolution='c')
x_wrd = np.linspace(0, map_wrd.urcrnrx, r2_mat_monthly.shape[1])
y_wrd = np.linspace(0, map_wrd.urcrnry, r2_mat_monthly.shape[0])
y_wrd = y_wrd[::-1]
xx_wrd, yy_wrd = np.meshgrid(x_wrd, y_wrd)


# Single maps
os.chdir(path_results)
fig = plt.figure(num=None, figsize=(9.5, 3), dpi=100, facecolor='w', edgecolor='k')
map_wrd = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
map_wrd.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
map_wrd_mesh = map_wrd.pcolormesh(xx_wrd, yy_wrd, np.nanmean(r2_mat_monthly[:, :, :2], axis=2), vmin=0, vmax=0.6, cmap='inferno')
# draw parallels
map_wrd.drawparallels(np.arange(lat_world_min,lat_world_max,45), labels=[1,1,1,1])
# draw meridians
map_wrd.drawmeridians(np.arange(lon_world_min,lon_world_max,90), labels=[1,1,1,1])
map_wrd.colorbar(map_wrd_mesh, extend='both', location='bottom', pad='15%')
plt.xticks(fontsize=12, rotation=90)
plt.suptitle('SMAP SM 9 km', y=1, fontsize=15)
plt.show()

# Subplot maps
# R^2 of AM
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 6.5), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    map_wrd = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max, ax=axes.flat[ipt],
                  llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
    map_wrd.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
    map_wrd_mesh = map_wrd.pcolormesh(xx_wrd, yy_wrd, np.nanmean(r2_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                                      vmin=0, vmax=0.6, cmap='viridis')
    map_wrd.drawparallels(np.arange(lat_world_min, lat_world_max, 45), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.drawmeridians(np.arange(lon_world_min, lon_world_max, 90), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.colorbar(map_wrd_mesh, extend='both', location='bottom', pad='15%')
    axes.flat[ipt].set_title(title_content[ipt], pad=20, fontsize=15, weight='bold')
plt.suptitle('$\mathregular{R^2}$ of a.m.', fontsize=20, weight='bold')
plt.tight_layout()
plt.show()
plt.savefig(path_results + '/r2_world_am.png')

# RMSE of AM
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 6.5), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    map_wrd = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max, ax=axes.flat[ipt],
                  llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
    map_wrd.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
    map_wrd_mesh = map_wrd.pcolormesh(xx_wrd, yy_wrd, np.nanmean(rmse_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                                      vmin=0, vmax=0.08, cmap='plasma')
    map_wrd.drawparallels(np.arange(lat_world_min, lat_world_max, 45), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.drawmeridians(np.arange(lon_world_min, lon_world_max, 90), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.colorbar(map_wrd_mesh, extend='both', location='bottom', pad='15%')
    axes.flat[ipt].set_title(title_content[ipt], pad=20, fontsize=15, weight='bold')
plt.suptitle('RMSE of a.m.', fontsize=20, weight='bold')
plt.tight_layout()
plt.show()
plt.savefig(path_results + '/rmse_world_am.png')

# Slope of AM
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 6.5), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    map_wrd = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max, ax=axes.flat[ipt],
                  llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
    map_wrd.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
    map_wrd_mesh = map_wrd.pcolormesh(xx_wrd, yy_wrd, np.nanmean(slope_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                                      vmin=-0.015, vmax=0, cmap='viridis')
    map_wrd.drawparallels(np.arange(lat_world_min, lat_world_max, 45), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.drawmeridians(np.arange(lon_world_min, lon_world_max, 90), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.colorbar(map_wrd_mesh, extend='both', location='bottom', pad='15%')
    axes.flat[ipt].set_title(title_content[ipt], pad=20, fontsize=15, weight='bold')
plt.suptitle('Slope of a.m.', fontsize=20, weight='bold')
plt.tight_layout()
plt.show()
plt.savefig(path_results + '/slope_am.png')




########################################################################################################################

# 1.2 SMAP SM maps

# 1.2.1 Worldwide
year_plt = yearname[4]
month_plt = 8
days_begin = 1
days_end = 5
days_n = days_end - days_begin + 1

# Load in SMAP 9 km SM
for iyr in [year_plt]:  # range(yearname):
    for imo in [month_plt]:  # range(len(monthname)):
        hdf_file_smap_9km = path_procdata + '/smap_sm_9km_' + str(iyr) + monthname[imo-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

smap_9km_plot = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), days_n*2))
smap_9km_plot[:] = np.nan
for idt in range(days_begin-1, days_end-1):
    smap_9km_plot[:, :, idt] = f_read_smap_9km[varname_list_smap_9km[0]][:, :, idt] # AM
    smap_9km_plot[:, :, idt+days_n] = f_read_smap_9km[varname_list_smap_9km[1]][:, :, idt] # PM
f_read_smap_9km.close()


# smap_1km_plot = np.empty((len(lat_world_ease_1km), len(lon_world_ease_1km), days_n*2))
# smap_1km_plot[:] = np.nan
# Load in SMAP 1 km SM
for iyr in [year_plt]:  # range(yearname):
    for imo in [month_plt]:  # range(len(monthname)):
        for idt in range(days_begin-1, days_end-1):
            str_date = str(iyr) + '-' + monthname[imo-1] + '-' + str(days_begin).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap_sm_ds + '/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray().astype(np.float32)
            src_tf_arr = np.transpose(src_tf_arr, (1, 2, 0))
            src_tf_arr_mean = np.nanmean(src_tf_arr, axis=2)
            # smap_1km_plot[:, :, idt] = src_tf_arr[:, :, 0]
            # smap_1km_plot[:, :, idt+5] = src_tf_arr[:, :, 1]
            del(src_tf, src_tf_arr, tif_file_smap_1km)
            print(idt)





# Single maps
os.chdir(path_results)
# 9 km
# Build the map file
map_wrd_9km = Basemap(projection='cea',llcrnrlat=lat_world_min,urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min,urcrnrlon=lon_world_max,resolution='c')
x_wrd_9km = np.linspace(0, map_wrd_9km.urcrnrx, smap_9km_plot.shape[1])
y_wrd_9km = np.linspace(0, map_wrd_9km.urcrnry, smap_9km_plot.shape[0])
y_wrd_9km = y_wrd_9km[::-1]
xx_wrd_9km, yy_wrd_9km = np.meshgrid(x_wrd_9km, y_wrd_9km)

fig = plt.figure(num=None, figsize=(9.5, 3), dpi=100, facecolor='w', edgecolor='k')
map_wrd_9km = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
map_wrd_9km.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
map_wrd_mesh_9km = map_wrd_9km.pcolormesh(xx_wrd_9km, yy_wrd_9km, np.nanmean(smap_9km_plot, axis=2), vmin=0, vmax=0.5, cmap='viridis_r')
# draw parallels
map_wrd_9km.drawparallels(np.arange(lat_world_min,lat_world_max,45), labels=[1,1,1,1])
# draw meridians
map_wrd_9km.drawmeridians(np.arange(lon_world_min,lon_world_max,90), labels=[1,1,1,1])
map_wrd_9km.colorbar(map_wrd_mesh_9km, extend='both', location='bottom', pad='15%')
plt.suptitle('SMAP SM 9 km', y=0.99)
plt.show()

# 1 km
# Build the map file
map_wrd = Basemap(projection='cea',llcrnrlat=lat_world_min,urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min,urcrnrlon=lon_world_max,resolution='c')
x_wrd = np.linspace(0, map_wrd.urcrnrx, src_tf_arr_mean.shape[1])
y_wrd = np.linspace(0, map_wrd.urcrnry, src_tf_arr_mean.shape[0])
y_wrd = y_wrd[::-1]
xx_wrd, yy_wrd = np.meshgrid(x_wrd, y_wrd)

fig = plt.figure(num=None, figsize=(9.5, 3), dpi=50, facecolor='w', edgecolor='k')
map_wrd = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
map_wrd.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
map_wrd_mesh = map_wrd.pcolormesh(xx_wrd, yy_wrd, src_tf_arr_mean, vmin=0, vmax=0.5, cmap='viridis_r')
# draw parallels
map_wrd.drawparallels(np.arange(lat_world_min,lat_world_max,45), labels=[1,1,1,1])
# draw meridians
map_wrd.drawmeridians(np.arange(lon_world_min,lon_world_max,90), labels=[1,1,1,1])
map_wrd.colorbar(map_wrd_mesh, extend='both', location='bottom', pad='15%')
plt.suptitle('SMAP SM 1 km', y=0.99)
plt.show()
plt.savefig(path_results + '/smap_1km_world.png')

# Subplot maps
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 6.5), facecolor='w', edgecolor='k')
for ipt in range(3):
    map_wrd = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max, ax=axes.flat[ipt*2],
                  llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
    map_wrd.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
    map_wrd_mesh = map_wrd.pcolormesh(xx_wrd, yy_wrd, smap_9km_plot[:, :, ipt], vmin=0, vmax=0.5, cmap='viridis_r')
    map_wrd.drawparallels(np.arange(lat_world_min, lat_world_max, 45), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.drawmeridians(np.arange(lon_world_min, lon_world_max, 90), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.colorbar(map_wrd_mesh, extend='both', location='bottom', pad='15%')
    axes.flat[ipt*2].set_title('SMAP SM 1 km', pad=20, fontsize=15)

    map_wrd = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max, ax=axes.flat[ipt*2+1],
                  llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
    map_wrd.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
    map_wrd_mesh = map_wrd.pcolormesh(xx_wrd, yy_wrd, smap_9km_plot[:, :, ipt], vmin=0, vmax=0.5, cmap='viridis_r')
    map_wrd.drawparallels(np.arange(lat_world_min, lat_world_max, 45), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.drawmeridians(np.arange(lon_world_min, lon_world_max, 90), labels=[1, 1, 1, 1], linewidth=0.5)
    map_wrd.colorbar(map_wrd_mesh, extend='both', location='bottom', pad='15%')
    axes.flat[ipt*2+1].set_title('SMAP SM 9 km', pad=20, fontsize=15)
plt.tight_layout()
plt.show()
plt.savefig(path_results + '/sm_comp.png')



# 1.2.2 River Basins
#1.2.2.1 Danube RB
# Extent: 8.1541666669581900,42.0827660457165962; 29.7172841397513707,50.2457929826601912

# Single maps
os.chdir(path_results)
# 9 km
# Build the map file
map_dan_9km = Basemap(projection='cea',llcrnrlat=42.0828,urcrnrlat=50.2458,
              llcrnrlon=8.1542,urcrnrlon=29.7173,resolution='c')
x_dan_9km = np.linspace(0, map_dan_9km.urcrnrx, smap_9km_plot.shape[1])
y_dan_9km = np.linspace(0, map_dan_9km.urcrnry, smap_9km_plot.shape[0])
y_dan_9km = y_dan_9km[::-1]
xx_dan_9km, yy_dan_9km = np.meshgrid(x_dan_9km, y_dan_9km)

fig = plt.figure(num=None, figsize=(9.5, 3), dpi=100, facecolor='w', edgecolor='k')
map_wrd_9km = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
map_wrd_9km.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
map_wrd_mesh_9km = map_wrd_9km.pcolormesh(xx_wrd_9km, yy_wrd_9km, np.nanmean(smap_9km_plot, axis=2), vmin=0, vmax=0.5, cmap='viridis_r')
# draw parallels
map_wrd_9km.drawparallels(np.arange(lat_world_min,lat_world_max,45), labels=[1,1,1,1])
# draw meridians
map_wrd_9km.drawmeridians(np.arange(lon_world_min,lon_world_max,90), labels=[1,1,1,1])
map_wrd_9km.colorbar(map_wrd_mesh_9km, extend='both', location='bottom', pad='15%')
plt.suptitle('SMAP SM 9 km', y=0.99)
plt.show()








########################################################################################################################
# 2. Scatter plots
# 2.1 Select geograhical locations by using index tables, and plot delta T - SM relationship lines through each NDVI class

# Lat/lon of the locations in the world:
# Solimoes, Amazon RB: -4.636, -70.266
# Tonzi Ranch, California RB: 38.432,-120.966
# Walnut Gulch, Colorado RB: 31.733, -110.05
# Banloc, Danube RB: 45.383, 21.133
# Krong Kracheh, Mekong: 12.49, 106.029
# Yanco, Murrumbidgee, Murray-Darling: -34.604, 146.41
# Nanchong, Yangtze: 30.838, 106.111
# Kanpur, Ganga-Brahmaputra: 26.521, 80.231

lat_slc = [-4.636, 38.432, 31.733, 45.383, 12.49, -34.604, 30.838, 26.521]
lon_slc = [-70.266, -120.966, -110.05, 21.133, 106.029, 146.41, 106.111, 80.231]
name_slc = ['Amazon', 'Tonzi Ranch', 'Walnut Gulch', 'Danube', 'Mekong', 'Yanco', 'Nanchong', 'Kanpur']
ndvi_class = np.linspace(0, 1, 11)
viridis = plt.cm.get_cmap('viridis', 10)

coord_25km_ind = []
for ico in range(len(lat_slc)):
    row_dist = np.absolute(lat_slc[ico] - lat_world_ease_25km)
    row_match = np.where(row_lmask_ease_25km_ind == np.argmin(row_dist))
    col_dist = np.absolute(lon_slc[ico] - lon_world_ease_25km)
    col_match = np.where(col_lmask_ease_25km_ind == np.argmin(col_dist))
    ind = np.intersect1d(row_match, col_match)[0]
    coord_25km_ind.append(ind)
    del(row_dist, row_match, col_dist, col_match, ind)

coord_25km_ind = np.array(coord_25km_ind)

# Load in data
os.chdir(path_procdata)
hdf_file = path_procdata + '/ds_model_07.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list = list(f_read.keys())

lst_am_delta = np.array([f_read[varname_list[0]][x, :] for x in coord_25km_ind])
lst_pm_delta = np.array([f_read[varname_list[1]][x, :] for x in coord_25km_ind])
ndvi = np.array([f_read[varname_list[2]][x, :] for x in coord_25km_ind])
sm_am = np.array([f_read[varname_list[3]][x, :] for x in coord_25km_ind])
sm_pm = np.array([f_read[varname_list[4]][x, :] for x in coord_25km_ind])


# Subplots of GLDAS SM vs. LST difference
# 2.1.1.
fig = plt.figure(figsize=(11, 6.5))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(1, 5):
    x = sm_am[i-1, :]
    y = lst_am_delta[i-1, :]
    c = ndvi[i-1, :]

    ax = fig.add_subplot(2, 2, i)
    sc = ax.scatter(x, y, c=c, marker='o', s=3, cmap='viridis')
    sc.set_clim(vmin=0,vmax=0.7)

    # Plot by NDVI classes
    for n in range(len(ndvi_class)-1):
        ndvi_ind = np.array([np.where((c>=ndvi_class[n]) & (c<ndvi_class[n+1]))[0]])
        if ndvi_ind.shape[1] > 10:
            coef, intr = np.polyfit(x[ndvi_ind].squeeze(), y[ndvi_ind].squeeze(), 1)
            ax.plot(x[ndvi_ind].squeeze(), intr + coef * x[ndvi_ind].squeeze(), '-', color=viridis.colors[n])
        else:
            pass

    plt.xlim(0, 0.4)
    ax.set_xticks(np.arange(0, 0.5, 0.1))
    plt.ylim(0, 40)
    ax.set_yticks(np.arange(0, 50, 10))
    ax.text(0.02, 5, name_slc[i-1],fontsize=15)
    plt.grid(linestyle='--')
    plt.colorbar(sc, extend='both')

fig.text(0.5, 0.01, 'Soil Moisture ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.5, 'Temperature Difference (K)', va='center', rotation='vertical', fontsize=16, fontweight='bold')
plt.savefig(path_results + '/gldas_comp_1.png')


# 2.1.2.
fig = plt.figure(figsize=(11, 6.5))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(5, 9):
    x = sm_am[i-1, :]
    y = lst_am_delta[i-1, :]
    c = ndvi[i-1, :]

    ax = fig.add_subplot(2, 2, i-4)
    sc = ax.scatter(x, y, c=c, marker='o', s=3, cmap='viridis')
    sc.set_clim(vmin=0,vmax=0.7)

    # Plot by NDVI classes
    for n in range(len(ndvi_class)-1):
        ndvi_ind = np.array([np.where((c>=ndvi_class[n]) & (c<ndvi_class[n+1]))[0]])
        if ndvi_ind.shape[1] > 10:
            coef, intr = np.polyfit(x[ndvi_ind].squeeze(), y[ndvi_ind].squeeze(), 1)
            ax.plot(x[ndvi_ind].squeeze(), intr + coef * x[ndvi_ind].squeeze(), '-', color=viridis.colors[n])
        else:
            pass

    plt.xlim(0.1, 0.5)
    ax.set_xticks(np.arange(0.1, 0.6, 0.1))
    plt.ylim(0, 30)
    ax.set_yticks(np.arange(0, 40, 10))
    ax.text(0.37, 25, name_slc[i-1],fontsize=15)
    plt.grid(linestyle='--')
    plt.colorbar(sc, extend='both')

fig.text(0.5, 0.01, 'Soil Moisture ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.5, 'Temperature Difference (K)', va='center', rotation='vertical', fontsize=16, fontweight='bold')
plt.savefig(path_results + '/gldas_comp_2.png')



