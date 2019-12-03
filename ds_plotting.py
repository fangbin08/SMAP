import os
os.environ['PROJ_LIB'] = '/Users/binfang/anaconda3/pkgs/proj4-5.0.1-h1de35cc_0/share/proj'
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import numpy as np
import h5py
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_CONUS/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_CONUS/results/results_191202'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'

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


########################################################################################################################
# 1. Maps

hdf_file = path_procdata + '/smap_sm_9km_201908.hdf5'
f_smap_9km = h5py.File(hdf_file, "r")
varname_list_smap = list(f_smap_9km.keys())
days = 3

smap_9km_plot = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), 3))
smap_9km_plot[:] = np.nan
for idt in range(days):
    smap_9km_plot[:, :, idt] = f_smap_9km[varname_list_smap[0]][:, :, idt]
f_smap_9km.close()


# Plot the maps
map = Basemap(projection='cea',llcrnrlat=lat_world_min,urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min,urcrnrlon=lon_world_max,resolution='c')
x = np.linspace(0, map.urcrnrx, smap_9km_plot.shape[1])
y = np.linspace(0, map.urcrnry, smap_9km_plot.shape[0])
y = y[::-1]
xx, yy = np.meshgrid(x, y)

smap_1km_plot = np.copy(smap_9km_plot)



# Single maps
fig = plt.figure(num=None, figsize=(9.5, 3), dpi=100, facecolor='w', edgecolor='k')
map = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max,
              llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
map.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
map_f = map.pcolormesh(xx, yy, smap_9km_plot[:, :, 0], vmin=0, vmax=0.5, cmap='viridis_r')
# draw parallels
map.drawparallels(np.arange(lat_world_min,lat_world_max,45), labels=[1,1,1,1])
# draw meridians
map.drawmeridians(np.arange(lon_world_min,lon_world_max,90), labels=[1,1,1,1])
map.colorbar(map_f, extend='both', location='bottom', pad='15%')
plt.suptitle('SMAP SM 9 km')
plt.show()



# Subplot maps
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 6.5), facecolor='w', edgecolor='k')
for ipt in range(1):
    map = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max, ax=axes.flat[ipt*2],
                  llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
    map.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
    map_f = map.pcolormesh(xx, yy, smap_9km_plot[:, :, ipt], vmin=0, vmax=0.5, cmap='viridis_r')
    map.drawparallels(np.arange(lat_world_min, lat_world_max, 45), labels=[1, 1, 1, 1], linewidth=0.5)
    map.drawmeridians(np.arange(lon_world_min, lon_world_max, 90), labels=[1, 1, 1, 1], linewidth=0.5)
    map.colorbar(map_f, extend='both', location='bottom', pad='15%')
    axes.flat[ipt*2].set_title('SMAP SM 1 km', pad=20, fontsize=15)

    map = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max, ax=axes.flat[ipt*2+1],
                  llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
    map.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
    map_f = map.pcolormesh(xx, yy, smap_9km_plot[:, :, ipt], vmin=0, vmax=0.5, cmap='viridis_r')
    map.drawparallels(np.arange(lat_world_min, lat_world_max, 45), labels=[1, 1, 1, 1], linewidth=0.5)
    map.drawmeridians(np.arange(lon_world_min, lon_world_max, 90), labels=[1, 1, 1, 1], linewidth=0.5)
    map.colorbar(map_f, extend='both', location='bottom', pad='15%')
    axes.flat[ipt*2+1].set_title('SMAP SM 9 km', pad=20, fontsize=15)
plt.tight_layout()
plt.show()
plt.savefig(path_results + 'sm_comp.png')


########################################################################################################################
# 2. Scatter plots

fig = plt.figure(figsize=(11, 6.5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
           fontsize=18, ha='center')










# fig = plt.figure(num=None, figsize=(30, 8), dpi=100, facecolor='w', edgecolor='k')
# map.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
# map.pcolormesh(xx, yy, smap_9km_plot[:, :, 0], vmin=0, vmax=0.6)
# # cb = plt.colorbar( orientation='vertical', fraction=0.10, shrink=0.7)
#
# plt.title("SMAP SM 9 km")
# # plt.tight_layout()
# plt.show(map)
#
#
# for ipt in range(3):
#     ax = fig.add_subplot(2, 3, ipt*2+1)
#     ax.set_title("SMAP SM 9 km")
#     map = Basemap(projection='cea', llcrnrlat=lat_world_min, urcrnrlat=lat_world_max, ax=ax,
#                   llcrnrlon=lon_world_min, urcrnrlon=lon_world_max, resolution='c')
#     map.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1', 'GSHHS_f_L1', drawbounds=True)
#     # map.pcolormesh(xx, yy, smap_9km_plot[:, :, ipt], vmin=0, vmax=0.6)
#
# plt.title("SMAP SM")
# # plt.tight_layout()
# plt.show()
