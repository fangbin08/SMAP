import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams["font.family"] = "serif"
import h5py
import calendar
import datetime
import glob
import pandas as pd
import rasterio
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import skill_metrics as sm
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import itertools

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

####################################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of source LTDR NDVI data
path_ltdr = '/Volumes/MyPassport/SMAP_Project/Datasets/LTDR/Ver5'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data'
# Path of source MODIS data
path_modis = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/HDF'
# Path of source output MODIS data
path_modis_op = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Output'
# Path of MODIS data for SM downscaling model input
path_modis_model_ip = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Model_Input'
# Path of SM model output
path_model_op = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP_ds/Model_Output'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Downscale'
# Path of 9 km SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'
# Path of ISMN
path_ismn = '/Volumes/MyPassport/SMAP_Project/Datasets/ISMN/Ver_1/processed_data'
# Path of processed data
path_processed = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of GPM data
path_gpm = '/Volumes/MyPassport/SMAP_Project/Datasets/GPM'
# Path of Results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200810'

folder_400m = '/400m/'
folder_1km = '/1km/'
folder_9km = '/9km/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2019-12-31'
year = 2019 - 2015 + 1

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
date_seq_doy = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(date_str.strftime('%Y%m%d'))
    date_seq_doy.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))

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

# daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=1)
# ind_init = daysofmonth_seq_cumsum[2, :]
# ind_end = daysofmonth_seq_cumsum[8, :] - 1
# ind_gpm = np.stack((ind_init, ind_end), axis=1)
# ind_gpm[0, :] = ind_gpm[0, :] - 90

daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=0)
ind_init = daysofmonth_seq_cumsum[2, :]
ind_end = daysofmonth_seq_cumsum[8, :]
ind_gpm = np.stack((ind_init, ind_end), axis=1)

# Extract the indices of the months between April - September
date_seq_month = np.array([int(date_seq[x][4:6]) for x in range(len(date_seq))])
monthnum_conus = monthnum[3:9]
date_seq_doy_conus_ind = np.where((date_seq_month >= 4) & (date_seq_month <= 9))[0]
date_seq_doy_conus = [date_seq_doy[date_seq_doy_conus_ind[x]] for x in range(len(date_seq_doy_conus_ind))]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_conus_max', 'lat_conus_min', 'lon_conus_max', 'lon_conus_min', 'cellsize_400m', 'cellsize_9km',
                'lat_conus_ease_1km', 'lon_conus_ease_1km', 'lat_conus_ease_9km', 'lon_conus_ease_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_conus_ease_400m', 'lon_conus_ease_400m',
                'row_conus_ease_9km_ind', 'col_conus_ease_9km_ind', 'lat_world_geo_10km', 'lon_world_geo_10km']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


########################################################################################################################
# 1. Read SM data in CONUS
# 1.1 Load the site lat/lon from Excel files and Locate the SMAP 400m, 1/9 km SM positions by lat/lon of ISMN in-situ data

# Find the indices of the days between April - Sepetember
month_list = np.array([int(date_seq[x][4:6]) for x in range(len(date_seq))])
month_list_ind = np.where((month_list >= 4) & (month_list <= 9))[0]
month_list_ind = month_list_ind + 2 #First two columns are lat/lon

ismn_list = sorted(glob.glob(path_ismn + '/[A-Z]*.xlsx'))

coords_all = []
df_table_am_all = []
df_table_pm_all = []
for ife in range(14, len(ismn_list)):
    df_table_am = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='AM')
    df_table_pm = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='PM')

    netname = os.path.basename(ismn_list[ife]).split('_')[1]
    netname = [netname] * df_table_am.shape[0]
    coords = df_table_am[['lat', 'lon']]
    coords_all.append(coords)

    df_table_am_value = df_table_am.iloc[:, month_list_ind]
    df_table_am_value.insert(0, 'network', netname)
    df_table_pm_value = df_table_pm.iloc[:, month_list_ind]
    df_table_pm_value.insert(0, 'network', netname)
    df_table_am_all.append(df_table_am_value)
    df_table_pm_all.append(df_table_pm_value)
    del(df_table_am, df_table_pm, df_table_am_value, df_table_pm_value, coords, netname)
    print(ife)

df_coords = pd.concat(coords_all)
df_table_am_all = pd.concat(df_table_am_all)
df_table_pm_all = pd.concat(df_table_pm_all)

new_index = [df_coords.index[x].title() for x in range(len(df_coords.index))] # Capitalize each word
df_coords.index = new_index
df_table_am_all.index = new_index
df_table_pm_all.index = new_index

rec_list = ['Smap-Ok', 'Tony_Grove_Rs', 'Bedford_5_Wnw', 'Harrison_20_Sse', 'John_Day_35_Wnw']
rec_post_list = ['SMAP-OK', 'Tony_Grove_RS', 'Bedford_5_WNW', 'Harrison_20_SSE', 'John_Day_35_WNW']
# rec_list_ind = [np.where(df_table_am_all.index == rec_list[x])[0][0] for x in range(len(rec_list))]
for x in range(1, len(rec_list)):
    df_table_am_all.rename(index={rec_list[x]: rec_post_list[x]}, inplace=True)
    df_table_pm_all.rename(index={rec_list[x]: rec_post_list[x]}, inplace=True)
    df_coords.rename(index={rec_list[x]: rec_post_list[x]}, inplace=True)


########################################################################################################################
# 1.2 Extract 400 m, 1 km / 9 km SMAP by lat/lon

# Locate the SM pixel positions
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

stn_row_400m_ind_all = []
stn_col_400m_ind_all = []
stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
stn_row_9km_ind_all = []
stn_col_9km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_400m_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_conus_ease_400m)).item()
    stn_col_400m_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_conus_ease_400m)).item()
    stn_row_400m_ind_all.append(stn_row_400m_ind)
    stn_col_400m_ind_all.append(stn_col_400m_ind)
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_conus_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_conus_ease_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    stn_row_9km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_9km)).item()
    stn_col_9km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_9km)).item()
    stn_row_9km_ind_all.append(stn_row_9km_ind)
    stn_col_9km_ind_all.append(stn_col_9km_ind)
    del(stn_row_400m_ind, stn_col_400m_ind, stn_row_1km_ind, stn_col_1km_ind, stn_row_9km_ind, stn_col_9km_ind)


# 1.3 Extract 400 m SMAP SM (2019)
smap_400m_sta_all = []
tif_files_400m_name_ind_all = []
for iyr in [3, 4]:  # range(yearname):

    os.chdir(path_smap + folder_400m + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Extract the file name
    tif_files_name = [os.path.splitext(tif_files[x])[0].split('_')[-1] for x in range(len(tif_files))]
    tif_files_name_1year_ind = [date_seq_doy_conus.index(item) for item in tif_files_name if item in date_seq_doy_conus]
    date_seq_doy_conus_1year_ind = [tif_files_name.index(item) for item in tif_files_name if item in date_seq_doy_conus]

    tif_files_400m_name_ind_all.append(tif_files_name_1year_ind)
    del(tif_files_name, tif_files_name_1year_ind)

    smap_400m_sta_1year = []
    for idt in range(len(date_seq_doy_conus_1year_ind)):
        src_tf = rasterio.open(tif_files[date_seq_doy_conus_1year_ind[idt]]).read()
        smap_400m_sta_1day = src_tf[:, stn_row_400m_ind_all, stn_col_400m_ind_all]
        smap_400m_sta_1year.append(smap_400m_sta_1day)
        del(src_tf, smap_400m_sta_1day)
        print(tif_files[date_seq_doy_conus_1year_ind[idt]])

    smap_400m_sta_all.append(smap_400m_sta_1year)
    del(smap_400m_sta_1year, date_seq_doy_conus_1year_ind)

tif_files_400m_name_ind_all = np.concatenate(tif_files_400m_name_ind_all)
smap_400m_sta_all = np.concatenate(smap_400m_sta_all)

# Fill the extracted SMAP SM into the proper position of days
smap_400m_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-1), dtype='float32')
smap_400m_sta_am[:] = np.nan
for idt in range(len(tif_files_400m_name_ind_all)):
    smap_400m_sta_am[:, tif_files_400m_name_ind_all[idt]] = smap_400m_sta_all[idt, 0, :]



# 1.4 Extract 1km SMAP SM (2019)
smap_1km_sta_all = []
tif_files_1km_name_ind_all = []
for iyr in [3, 4]:  # range(yearname):

    os.chdir(path_smap + folder_1km + '/nldas/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Extract the file name
    tif_files_name = [os.path.splitext(tif_files[x])[0].split('_')[-1] for x in range(len(tif_files))]
    tif_files_name_1year_ind = [date_seq_doy_conus.index(item) for item in tif_files_name if item in date_seq_doy_conus]
    date_seq_doy_conus_1year_ind = [tif_files_name.index(item) for item in tif_files_name if item in date_seq_doy_conus]

    tif_files_1km_name_ind_all.append(tif_files_name_1year_ind)
    del(tif_files_name, tif_files_name_1year_ind)

    smap_1km_sta_1year = []
    for idt in range(len(date_seq_doy_conus_1year_ind)):
        src_tf = rasterio.open(tif_files[date_seq_doy_conus_1year_ind[idt]]).read()
        smap_1km_sta_1day = src_tf[:, stn_row_1km_ind_all, stn_col_1km_ind_all]
        smap_1km_sta_1year.append(smap_1km_sta_1day)
        del(src_tf, smap_1km_sta_1day)
        print(tif_files[date_seq_doy_conus_1year_ind[idt]])

    smap_1km_sta_all.append(smap_1km_sta_1year)
    del(smap_1km_sta_1year, date_seq_doy_conus_1year_ind)


tif_files_1km_name_ind_all = np.concatenate(tif_files_1km_name_ind_all)
smap_1km_sta_all = np.concatenate(smap_1km_sta_all)

# Fill the extracted SMAP SM into the proper position of days
smap_1km_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-1), dtype='float32')
smap_1km_sta_am[:] = np.nan
for idt in range(len(tif_files_1km_name_ind_all)):
    smap_1km_sta_am[:, tif_files_1km_name_ind_all[idt]] = smap_1km_sta_all[idt, 0, :]


# 1.5 Extract 9km SMAP SM (2019)
smap_9km_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-1), dtype='float32')
smap_9km_sta_am[:] = np.nan

for iyr in [3, 4]: #range(len(yearname)):

    smap_9km_sta_am_1year = []
    for imo in range(3, 9):#range(len(monthname)):

        smap_9km_sta_am_1month = []
        # Load in SMAP 9km SM data
        smap_file_path = path_smap + folder_9km + 'smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

        # Find the if the file exists in the directory
        if os.path.exists(smap_file_path) == True:

            f_smap_9km = h5py.File(smap_file_path, "r")
            varname_list_smap = list(f_smap_9km.keys())
            smap_9km_sta_am_1month = f_smap_9km[varname_list_smap[0]][()]
            smap_9km_sta_am_1month = smap_9km_sta_am_1month[stn_row_9km_ind_all, stn_col_9km_ind_all, :]

            print(smap_file_path)
            f_smap_9km.close()

        else:
            pass

        smap_9km_sta_am_1year.append(smap_9km_sta_am_1month)
        del(smap_9km_sta_am_1month)

    smap_9km_sta_am_1year = np.concatenate(smap_9km_sta_am_1year, axis=1)
    smap_9km_sta_am[:, iyr*183:(iyr+1)*183] = smap_9km_sta_am_1year
    del(smap_9km_sta_am_1year)


# Save variables
var_name_val = ['smap_400m_sta_am', 'smap_1km_sta_am', 'smap_9km_sta_am']
with h5py.File('/Users/binfang/Downloads/Processing/VIIRS/smap_validation_conus_viirs.hdf5', 'w') as f:
    for x in var_name_val:
        f.create_dataset(x, data=eval(x))
f.close()


########################################################################################################################
# 2. Scatterplots
# Site ID
# COSMOS: 0, 11, 25, 28, 34, 36, 42, 44
# SCAN: 250, 274, 279, 286, 296, 351, 362, 383
# SOILSCAPE: 860, 861, 870, 872, 896, 897, 904, 908
# USCRN: 918, 926, 961, 991, 1000，1002, 1012, 1016

# Number of sites for each SM network
# COSMOS 52
# iRON 9
# PBO_H2O 140
# RISMA 9
# SCAN 188
# SNOTEL 404
# SOILSCAPE 119
# USCRN 113

# Load in the saved parameters
f_mat = h5py.File('/Users/binfang/Downloads/Processing/VIIRS/smap_validation_conus_viirs.hdf5', 'r')
varname_list = list(f_mat.keys())
for x in range(len(varname_list)):
    var_obj = f_mat[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f_mat.close()

# os.chdir(path_results + '/single')
ismn_sm_am = np.array(df_table_am_all.iloc[:, 1:])
ismn_sm_pm = np.array(df_table_pm_all.iloc[:, 1:])


# 2.1 single plots
# stat_array_allnan = np.empty([3, 6], dtype='float32')
# stat_array_allnan[:] = np.nan
stat_array_400m = []
stat_array_1km = []
stat_array_9km = []
ind_slc_all = []
for ist in range(len(ismn_sm_am)):

    x = ismn_sm_am[ist, :].flatten()
    y1 = smap_400m_sta_am[ist, :].flatten()
    y2 = smap_1km_sta_am[ist, :].flatten()
    y3 = smap_9km_sta_am[ist, :].flatten()
    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2) & ~np.isnan(y3))[0]

    if len(ind_nonnan) > 5:
        x = x[ind_nonnan]
        y1 = y1[ind_nonnan]
        y2 = y2[ind_nonnan]
        y3 = y3[ind_nonnan]

        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(x, y1)
        y1_estimated = intercept_1 + slope_1 * x
        number_1 = len(y1)
        r_sq_1 = r_value_1 ** 2
        ubrmse_1 = np.sqrt(np.mean((x - y1_estimated) ** 2))
        bias_1 = np.mean(x - y1)
        conf_int_1 = std_err_1 * 1.96  # From the Z-value
        stdev_1 = np.std(y1)
        stat_array_1 = [number_1, r_sq_1, ubrmse_1, stdev_1, bias_1, p_value_1, conf_int_1]

        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(x, y2)
        y2_estimated = intercept_2 + slope_2 * x
        number_2 = len(y2)
        r_sq_2 = r_value_2 ** 2
        ubrmse_2 = np.sqrt(np.mean((x - y2_estimated) ** 2))
        bias_2 = np.mean(x - y2)
        conf_int_2 = std_err_2 * 1.96  # From the Z-value
        stdev_2 = np.std(y2)
        stat_array_2 = [number_2, r_sq_2, ubrmse_2, stdev_2, bias_2, p_value_2, conf_int_2]

        slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = stats.linregress(x, y3)
        y3_estimated = intercept_3 + slope_3 * x
        number_3 = len(y3)
        r_sq_3 = r_value_3 ** 2
        ubrmse_3 = np.sqrt(np.mean((x - y3_estimated) ** 2))
        bias_3 = np.mean(x - y3)
        conf_int_3 = std_err_3 * 1.96  # From the Z-value
        stdev_3 = np.std(y3)
        stat_array_3 = [number_3, r_sq_3, ubrmse_3, stdev_3, bias_3, p_value_3, conf_int_3]

        if ubrmse_1 - ubrmse_3 < 0:
            fig = plt.figure(figsize=(11, 6.5))
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            ax = fig.add_subplot(111)

            ax.scatter(x, y1, s=20, c='m', marker='s', label='400 m')
            ax.scatter(x, y2, s=20, c='b', marker='o', label='1 km')
            ax.scatter(x, y3, s=20, c='g', marker='^', label='9 km')
            ax.plot(x, intercept_1+slope_1*x, '-', color='m')
            ax.plot(x, intercept_2+slope_2*x, '-', color='b')
            ax.plot(x, intercept_3+slope_3*x, '-', color='g')

            plt.xlim(0, 0.4)
            ax.set_xticks(np.arange(0, 0.5, 0.1))
            plt.ylim(0, 0.4)
            ax.set_yticks(np.arange(0, 0.5, 0.1))
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            plt.grid(linestyle='--')
            plt.legend(loc='upper left', prop={'size': 13})
            # plt.title(network_name[ist], fontsize=18, fontweight='bold')
            # plt.show()
            # plt.savefig(path_results + '/validation/single_plots/' + df_table_am_all['network'][ist] + '_' + df_table_am_all.index[ist]
            #             + '_(' + str(ist) + ')' + '.png')
            plt.close(fig)
            stat_array_400m.append(stat_array_1)
            stat_array_1km.append(stat_array_2)
            stat_array_9km.append(stat_array_3)
            ind_slc_all.append(ist)
            print(ist)
            del(stat_array_1, stat_array_2, stat_array_3)

        else:
            pass

    else:
        pass


stat_array_400m = np.array(stat_array_400m)
stat_array_1km = np.array(stat_array_1km)
stat_array_9km = np.array(stat_array_9km)

columns_validation = ['number', 'r_sq', 'ubrmse', 'stdev', 'bias', 'p_value', 'conf_int']
index_validation = df_coords.index[ind_slc_all]
# index_validation = ['COSMOS', 'SCAN', 'USCRN']

# stat_array_400m = np.concatenate((id, stat_array_400m), axis=1)
# stat_array_1km = np.concatenate((id, stat_array_1km), axis=1)
# stat_array_9km = np.concatenate((id, stat_array_9km), axis=1)
df_stat_400m = pd.DataFrame(stat_array_400m, columns=columns_validation, index=index_validation)
# df_stat_400m = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_400m], axis=1)
df_stat_1km = pd.DataFrame(stat_array_1km, columns=columns_validation, index=index_validation)
# df_stat_1km = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_1km], axis=1)
df_stat_9km = pd.DataFrame(stat_array_9km, columns=columns_validation, index=index_validation)
# df_stat_9km = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_9km], axis=1)
writer_400m = pd.ExcelWriter(path_results + '/validation/stat_400m.xlsx')
writer_1km = pd.ExcelWriter(path_results + '/validation/stat_1km.xlsx')
writer_9km = pd.ExcelWriter(path_results + '/validation/stat_9km.xlsx')
df_stat_400m.to_excel(writer_400m)
df_stat_1km.to_excel(writer_1km)
df_stat_9km.to_excel(writer_9km)
writer_400m.save()
writer_1km.save()
writer_9km.save()

# ubrmse_diff = stat_array_400m[:, 2] - stat_array_9km[:, 2]
# ubrmse_diff_ind = np.where(ubrmse_diff<0)[0]
# ubrmse_good = df_table_am_all['network'][ubrmse_diff_ind]
stn_slc_all = df_table_am_all['network'][ind_slc_all]
stn_slc_all_unique = stn_slc_all.unique()
stn_slc_all_group = [np.where(stn_slc_all == stn_slc_all_unique[x]) for x in range(len(stn_slc_all_unique))]


# 2.2 subplots
# COSMOS: 3, 41
# SCAN: 211, 229, 254, 258, 272, 280, 298, 330, 352, 358
# SNOTEL: 427, 454, 492, 520, 522, 583, 714, 721, 750, 755
# USCRN: 914, 918, 920, 947, 952, 957, 961, 985, 1002, 1016
network_name = ['COSMOS', 'SCAN', 'SNOTEL', 'USCRN']
site_ind = [[3, 9, 23, 36, 41, 44], [211, 229, 254, 258, 272, 280, 298, 330, 352, 358], [427, 454, 492, 520, 522, 583, 714, 721, 750, 755],
            [914, 918, 920, 947, 952, 957, 961, 985, 1002, 1016]]
# network_name = list(stn_slc_all_unique)
# site_ind = stn_slc_all_group

for inw in range(1, len(site_ind)):
    fig = plt.figure(figsize=(11, 11))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.92, hspace=0.25, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = ismn_sm_am[site_ind[inw][ist], :].flatten()
        x[x == 0] = np.nan
        y1 = smap_400m_sta_am[site_ind[inw][ist], :].flatten()
        y2 = smap_1km_sta_am[site_ind[inw][ist], :].flatten()
        y3 = smap_9km_sta_am[site_ind[inw][ist], :].flatten()
        ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2) & ~np.isnan(y3))[0]

        x = x[ind_nonnan]
        y1 = y1[ind_nonnan]
        y2 = y2[ind_nonnan]
        y3 = y3[ind_nonnan]

        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(x, y1)
        y1_estimated = intercept_1 + slope_1 * x
        number_1 = len(y1)
        r_sq_1 = r_value_1 ** 2
        ubrmse_1 = np.sqrt(np.mean((x - y1_estimated) ** 2))
        bias_1 = np.mean(x - y1)
        conf_int_1 = std_err_1 * 1.96  # From the Z-value
        stat_array_1 = [number_1, r_sq_1, ubrmse_1, bias_1, p_value_1, conf_int_1]

        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(x, y2)
        y2_estimated = intercept_2 + slope_2 * x
        number_2 = len(y2)
        r_sq_2 = r_value_2 ** 2
        ubrmse_2 = np.sqrt(np.mean((x - y2_estimated) ** 2))
        bias_2 = np.mean(x - y2)
        conf_int_2 = std_err_2 * 1.96  # From the Z-value
        stat_array_2 = [number_2, r_sq_2, ubrmse_2, bias_2, p_value_2, conf_int_2]

        slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = stats.linregress(x, y3)
        y3_estimated = intercept_3 + slope_3 * x
        number_3 = len(y3)
        r_sq_3 = r_value_3 ** 2
        ubrmse_3 = np.sqrt(np.mean((x - y3_estimated) ** 2))
        bias_3 = np.mean(x - y3)
        conf_int_3 = std_err_3 * 1.96  # From the Z-value
        stat_array_3 = [number_3, r_sq_3, ubrmse_3, bias_3, p_value_3, conf_int_3]

        ax = fig.add_subplot(len(site_ind[inw])//2, 2, ist+1)
        sc1 = ax.scatter(x, y1, s=20, c='m', marker='s', label='400 m')
        sc2 = ax.scatter(x, y2, s=20, c='b', marker='o', label='1 km')
        sc3 = ax.scatter(x, y3, s=20, c='g', marker='^', label='9 km')
        ax.plot(x, intercept_1+slope_1*x, '-', color='m')
        ax.plot(x, intercept_2+slope_2*x, '-', color='b')
        ax.plot(x, intercept_3+slope_3*x, '-', color='g')

        plt.xlim(0, 0.4)
        ax.set_xticks(np.arange(0, 0.5, 0.1))
        plt.ylim(0, 0.4)
        ax.set_yticks(np.arange(0, 0.5, 0.1))
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        plt.grid(linestyle='--')
        ax.text(0.01, 0.35, df_table_am_all.index[site_ind[inw][ist]].replace('_', ' '), fontsize=13)

    # add all legends together
    handles = [sc1] + [sc2] + [sc3]
    labels = [l.get_label() for l in handles]
    # leg = plt.legend([sc1, sc2, sc3], labels, loc=(-0.6, 3.55), mode="expand", borderaxespad=0, ncol=3, prop={"size": 13})
    leg = plt.legend([sc1, sc2, sc3], labels, loc=(-0.6, 6.1), mode="expand", borderaxespad=0, ncol=3, prop={"size": 13})

    fig.text(0.52, 0.01, 'In Situ SM ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.4, 'SMAP SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=21, y=0.99, fontweight='bold')
    plt.show()

    plt.savefig(path_results + '/validation/subplots/' + network_name[inw] + '.png')
    plt.close(fig)


########################################################################################################################
# 3. Time-series plots
# 3.1 Locate the corresponding GPM 10 km data located by lat/lon of in-situ data

# df_slc_coords = pd.read_csv(path_results + '/slc_coords.csv', index_col=0)
# slc_coords = np.array(df_slc_coords.iloc[:, 1:])

stn_row_10km_ind_all = []
stn_col_10km_ind_all = []
for ist in range(df_coords.shape[0]):
    stn_row_10km_ind = np.argmin(np.absolute(df_coords.iloc[ist, 0] - lat_world_geo_10km)).item()
    stn_col_10km_ind = np.argmin(np.absolute(df_coords.iloc[ist, 1] - lon_world_geo_10km)).item()
    stn_row_10km_ind_all.append(stn_row_10km_ind)
    stn_col_10km_ind_all.append(stn_col_10km_ind)
    del(stn_row_10km_ind, stn_col_10km_ind)

# Extract the GPM data by indices
gpm_precip_ext_all = []
for iyr in [3, 4]:#range(len(yearname)-1):

    f_gpm = h5py.File(path_gpm + '/gpm_precip_' + str(yearname[iyr]) + '.hdf5', 'r')
    varname_list_gpm = list(f_gpm.keys())

    for x in range(len(varname_list_gpm)):
        var_obj = f_gpm[varname_list_gpm[x]][()]
        exec(varname_list_gpm[x] + '= var_obj')
        del (var_obj)
    f_gpm.close()

    exec('gpm_precip = gpm_precip_10km_' + str(yearname[iyr]))
    gpm_precip_ext = gpm_precip[stn_row_10km_ind_all, stn_col_10km_ind_all, :]
    gpm_precip_ext_all.append(gpm_precip_ext)
    print(iyr)
    del(gpm_precip, gpm_precip_ext)


ind_gpm = ind_gpm[-2:, :]
gpm_precip_ext_all = [gpm_precip_ext_all[x][:, ind_gpm[x, 0]:ind_gpm[x, 1]] for x in range(len(gpm_precip_ext_all))]
gpm_precip_ext_all = np.concatenate(gpm_precip_ext_all, axis=1)

gpm_precip_ext = np.empty((1034, 549), dtype='float32')
gpm_precip_ext[:] = np.nan

gpm_precip_ext = np.concatenate((gpm_precip_ext, gpm_precip_ext_all), axis=1)

# index = df_slc_sites.index
# columns = df_table_am_all.columns[1:]
# df_gpm_precip_ext = pd.DataFrame(gpm_precip_ext_all, index=index, columns=columns)
# df_gpm_precip_ext.to_csv(path_results + '/gpm_precip_ext.csv', index=True)


# 3.2 Make the time-series plots

# df_gpm_precip = pd.read_csv(path_results + '/gpm_precip_ext.csv', index_col=0)
# gpm_precip_ext = np.array(df_gpm_precip)

# site_ind = [[0, 11, 25, 28, 34, 36, 42, 44], [250, 274, 279, 286, 296, 351, 362, 383],
#             [860, 861, 870, 872, 896, 897, 904, 908], [918, 926, 961, 991, 1000, 1002, 1012, 1016]]
# network_name = ['COSMOS', 'SCAN', 'SOILSCAPE', 'USCRN']

network_name = ['COSMOS', 'SCAN', 'SNOTEL', 'USCRN']
site_ind = [[9, 23, 36, 41, 44], [229, 254, 280, 330, 352], [492, 520, 522, 714, 721],
            [947, 957, 985, 1002, 1016]]

# Find the indices from df_gpm_precip
# df_gpm_precip_ind = [df_gpm_precip.index.get_loc(df_table_am_all.index[site_ind[y][x]]) for y in range(len(site_ind)) for x in range(len(site_ind[y]))]
# df_gpm_precip_ind = [df_gpm_precip_ind[:8], df_gpm_precip_ind[8:16], df_gpm_precip_ind[16:24], df_gpm_precip_ind[24:]]

for inw in range(len(site_ind)):

    fig = plt.figure(figsize=(13, 8))
    plt.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.92, hspace=0.35, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = ismn_sm_am[site_ind[inw][ist], 549:]
        y1 = smap_400m_sta_am[site_ind[inw][ist], 549:]
        y2 = smap_1km_sta_am[site_ind[inw][ist], 549:]
        y3 = smap_9km_sta_am[site_ind[inw][ist], 549:]
        z = gpm_precip_ext[site_ind[inw][ist], 549:]
        # ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        # x = x[ind_nonnan]
        # y1 = y1[ind_nonnan]
        # y2 = y2[ind_nonnan]
        # z = z[ind_nonnan]

        ax = fig.add_subplot(5, 1, ist+1)
        lns1 = ax.plot(x, c='k', marker='+', label='In-situ', markersize=3, linestyle='None')
        lns2 = ax.plot(y1, c='m', marker='s', label='400 m', markersize=2, linestyle='None')
        lns3 = ax.plot(y2, c='b', marker='o', label='1 km', markersize=2, linestyle='None')
        lns4 = ax.plot(y3, c='g', marker='^', label='9 km', markersize=2, linestyle='None')
        ax.text(310, 0.4, df_table_am_all.index[site_ind[inw][ist]].replace('_', ' '), fontsize=11)

        plt.xlim(0, len(x)//2)
        ax.set_xticks(np.arange(0, len(x)//2*3, (len(x))//2))
        ax.set_xticklabels([])
        labels = ['2018', '2019']
        mticks = ax.get_xticks()
        ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(labels, minor=True)

        plt.ylim(0, 0.5)
        ax.set_yticks(np.arange(0, 0.6, 0.2))
        ax.tick_params(axis='y', labelsize=10)

        ax2 = ax.twinx()
        ax2.set_ylim(0, 64, 8)
        ax2.invert_yaxis()
        lns5 = ax2.bar(np.arange(len(x)), z, width=0.8, color='royalblue', label='Precip')
        ax2.tick_params(axis='y', labelsize=10)


    # add all legends together
    handles = lns1+lns2+lns3+lns4+[lns5]
    labels = [l.get_label() for l in handles]

    # handles, labels = ax.get_legend_handles_labels()
    plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.04, 6))
    fig.text(0.5, 0.01, 'Days', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.4, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.96, 0.4, 'GPM Precip (mm/day)', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=19, y=0.97, fontweight='bold')
    plt.savefig(path_results + '/validation/subplots/' + network_name[inw] + '_tseries' + '.png')
    plt.close(fig)


########################################################################################################################
# 4. Make CONUS maps for R^2
df_stats = pd.read_csv(path_results + '/validation/stat_all.csv', index_col=0)
stn_coords_ind = [np.where(df_coords.index == df_stats.index[x])[0][0] for x in range(len(df_stats))]
df_coords_slc = df_coords.iloc[stn_coords_ind]
# df_coords_slc = df_table_am_all.iloc[stn_coords_ind]
stn_lat = [df_coords_slc.iloc[x]['lat'] for x in range(len(df_stats))]
stn_lon = [df_coords_slc.iloc[x]['lon'] for x in range(len(df_stats))]

# site_ind = [[3, 9, 23, 36, 41, 44], [211, 229, 254, 258, 272, 280, 298, 330, 352, 358], [427, 454, 492, 520, 522, 583, 714, 721, 750, 755],
#             [914, 918, 920, 947, 952, 957, 961, 985, 1002, 1016]]
site_ind = [[3, 9, 23, 36, 41, 44], [211, 229, 254, 258, 272, 280, 298, 330, 352, 358], [427, 454, 492, 520, 522, 583, 714, 721, 750, 755],
            [914, 918, 920, 947, 952, 957, 961, 985, 1002, 1016]]
site_ind_flat = list(itertools.chain(*site_ind))
site_ind_name = df_table_am_all.iloc[site_ind_flat]
site_ind_name = site_ind_name['network']
df_stats_slc_ind = [np.where(df_stats.index == site_ind_name.index[x])[0][0] for x in range(len(site_ind_flat))]
df_stats_slc = df_stats.iloc[df_stats_slc_ind]
df_stats_slc_full = pd.concat([site_ind_name, df_stats_slc], axis=1)

# Write to file
writer_stn = pd.ExcelWriter(path_results + '/validation/stat_stn.xlsx')
df_stats_slc_full.to_excel(writer_stn)
writer_stn.save()

# Write coordinates and network to files
# df_coords_full = pd.concat([df_table_am_all['network'].to_frame().reset_index(drop=True, inplace=True),
#                             df_coords.reset_index(drop=True, inplace=True)], axis=1)

df_coords.iloc[ind_slc_all].to_csv(path_results + '/df_coords.csv', index=True)
df_table_am_all_slc = df_table_am_all.iloc[ind_slc_all]
df_network = df_table_am_all_slc['network'].to_frame()
df_network.to_csv(path_results + '/df_network.csv', index=True)


# 4.1 Make the maps
# Extract state name and center coordinates
shp_records = Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').records()
shp_records = list(shp_records)
state_name = [shp_records[x].attributes['STUSPS'] for x in range(len(shp_records))]
# name_lon = [(shp_records[x].bounds[0] + shp_records[x].bounds[2])/2 for x in range(len(shp_records))]
# name_lat = [(shp_records[x].bounds[1] + shp_records[x].bounds[3])/2 for x in range(len(shp_records))]

shape_conus = ShapelyFeature(Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
shape_conus_geometry = list(Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').geometries())
name_coords = [shape_conus_geometry[x].representative_point().coords[:] for x in range(len(shape_conus_geometry))]

c_rsq_400m = df_stats['r_sq_400m'].tolist()
c_rmse_400m = df_stats['ubrmse_400m'].tolist()
c_rsq_1km = df_stats['r_sq_1km'].tolist()
c_rmse_1km = df_stats['ubrmse_1km'].tolist()
c_rsq_9km = df_stats['r_sq_9km'].tolist()
c_rmse_9km = df_stats['ubrmse_9km'].tolist()


# 4.1.1 R^2
fig = plt.figure(figsize=(10, 12), dpi=100, facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)
# 400 m
ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-125, -67, 25, 50], ccrs.PlateCarree())
ax1.add_feature(shape_conus)
sc1 = ax1.scatter(stn_lon, stn_lat, c=c_rsq_400m, s=40, marker='^', edgecolors='k', cmap='jet')
sc1.set_clim(vmin=0, vmax=1)
ax1.text(-123, 27, '400 m', fontsize=16, fontweight='bold')
for x in range(len(shp_records)):
    ax1.annotate(s=state_name[x], xy=name_coords[x][0], horizontalalignment='center')
# 1 km
ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([-125, -67, 25, 50], ccrs.PlateCarree())
ax2.add_feature(shape_conus)
sc2 = ax2.scatter(stn_lon, stn_lat, c=c_rsq_1km, s=40, marker='^', edgecolors='k', cmap='jet')
sc2.set_clim(vmin=0, vmax=1)
ax2.text(-123, 27, '1 km', fontsize=16, fontweight='bold')
for x in range(len(shp_records)):
    ax2.annotate(s=state_name[x], xy=name_coords[x][0], horizontalalignment='center')
# 9 km
ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax3.set_extent([-125, -67, 25, 50], ccrs.PlateCarree())
ax3.add_feature(shape_conus)
sc3 = ax3.scatter(stn_lon, stn_lat, c=c_rsq_9km, s=40, marker='^', edgecolors='k', cmap='jet')
sc3.set_clim(vmin=0, vmax=1)
ax3.text(-123, 27, '9 km', fontsize=16, fontweight='bold')
for x in range(len(shp_records)):
    ax3.annotate(s=state_name[x], xy=name_coords[x][0], horizontalalignment='center')

cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
cbar = fig.colorbar(sc3, cax=cbar_ax, extend='both')
cbar.ax.locator_params(nbins=5)
cbar.ax.tick_params(labelsize=14)
plt.suptitle('$\mathregular{R^2}$', fontsize=20, y=0.98, fontweight='bold')
plt.savefig(path_results + '/validation/' + 'r2_map.png')
plt.close(fig)

# 4.1.2 RMSE
fig = plt.figure(figsize=(10, 12), dpi=100, facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)
# 400 m
ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-125, -67, 25, 50], ccrs.PlateCarree())
ax1.add_feature(shape_conus)
sc1 = ax1.scatter(stn_lon, stn_lat, c=c_rmse_400m, s=40, marker='^', edgecolors='k', cmap='jet')
sc1.set_clim(vmin=0, vmax=0.3)
ax1.text(-123, 27, '400 m', fontsize=16, fontweight='bold')
for x in range(len(shp_records)):
    ax1.annotate(s=state_name[x], xy=name_coords[x][0], horizontalalignment='center')
# 1 km
ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([-125, -67, 25, 50], ccrs.PlateCarree())
ax2.add_feature(shape_conus)
sc2 = ax2.scatter(stn_lon, stn_lat, c=c_rmse_1km, s=40, marker='^', edgecolors='k', cmap='jet')
sc2.set_clim(vmin=0, vmax=0.3)
ax2.text(-123, 27, '1 km', fontsize=16, fontweight='bold')
for x in range(len(shp_records)):
    ax2.annotate(s=state_name[x], xy=name_coords[x][0], horizontalalignment='center')
# 9 km
ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
ax3.set_extent([-125, -67, 25, 50], ccrs.PlateCarree())
ax3.add_feature(shape_conus)
sc3 = ax3.scatter(stn_lon, stn_lat, c=c_rmse_9km, s=40, marker='^', edgecolors='k', cmap='jet')
sc3.set_clim(vmin=0, vmax=0.3)
ax3.text(-123, 27, '9 km', fontsize=16, fontweight='bold')
for x in range(len(shp_records)):
    ax3.annotate(s=state_name[x], xy=name_coords[x][0], horizontalalignment='center')

cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
cbar = fig.colorbar(sc3, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=14)
cbar.ax.locator_params(nbins=6)
cbar.ax.tick_params(labelsize=14)

plt.suptitle('RMSE', fontsize=20, y=0.98, fontweight='bold')
plt.savefig(path_results + '/validation/' + 'rmse_map.png')
plt.close(fig)


# 4.1.3 R^2 and RMSE map
c_rsq_400m_3net = c_rsq_400m[0:88] + c_rsq_400m[255:]
c_rmse_400m_3net = c_rmse_400m[0:88] + c_rmse_400m[255:]
stn_lon_3net = stn_lon[0:88] + stn_lon[255:]
stn_lat_3net = stn_lat[0:88] + stn_lat[255:]

fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)
# R^2
ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-125, -67, 25, 50], ccrs.PlateCarree())
ax1.add_feature(shape_conus)
sc1 = ax1.scatter(stn_lon_3net, stn_lat_3net, c=c_rsq_400m_3net, s=40, marker='^', edgecolors='k', cmap='jet')
sc1.set_clim(vmin=0, vmax=1)
ax1.text(-123, 27, '$\mathregular{R^2}$', fontsize=16, fontweight='bold')
for x in range(len(shp_records)):
    ax1.annotate(s=state_name[x], xy=name_coords[x][0], horizontalalignment='center')
cbar_ax1 = fig.add_axes([0.9, 0.52, 0.015, 0.43])
cbar1 = fig.colorbar(sc1, cax=cbar_ax1, extend='both')
cbar1.ax.locator_params(nbins=5)
cbar1.ax.tick_params(labelsize=12)
# RMSE
ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
ax2.set_extent([-125, -67, 25, 50], ccrs.PlateCarree())
ax2.add_feature(shape_conus)
sc2 = ax2.scatter(stn_lon_3net, stn_lat_3net, c=c_rmse_400m_3net, s=40, marker='^', edgecolors='k', cmap='jet')
sc2.set_clim(vmin=0, vmax=0.3)
ax2.text(-123, 27, 'RMSE', fontsize=16, fontweight='bold')
for x in range(len(shp_records)):
    ax2.annotate(s=state_name[x], xy=name_coords[x][0], horizontalalignment='center')
cbar_ax2 = fig.add_axes([0.9, 0.05, 0.015, 0.43])
cbar2 = fig.colorbar(sc2, cax=cbar_ax2, extend='both')
cbar2.ax.locator_params(nbins=6)
cbar2.ax.tick_params(labelsize=12)
cbar2.set_label('$\mathregular{(m^3/m^3)}$', fontsize=14)
plt.savefig(path_results + '/validation/' + 'r2_rmse_map.png')
plt.close(fig)


########################################################################################################################
# 5. Taylor diagram
df_stats = pd.read_csv(path_results + '/validation/stat_all.csv', index_col=0)
stn_coords_ind = [np.where(df_coords.index == df_stats.index[x])[0][0] for x in range(len(df_stats))]

stdev_400m = np.array(df_stats['stdev_400m'])
rmse_400m = np.array(df_stats['ubrmse_400m'])
r_400m = np.array(np.sqrt(df_stats['r_sq_400m']))
stdev_1km = np.array(df_stats['stdev_1km'])
rmse_1km = np.array(df_stats['ubrmse_1km'])
r_1km = np.array(np.sqrt(df_stats['r_sq_1km']))
stdev_9km = np.array(df_stats['stdev_9km'])
rmse_9km = np.array(df_stats['ubrmse_9km'])
r_9km = np.array(np.sqrt(df_stats['r_sq_9km']))

# 5.1 Plot together
fig = plt.figure(figsize=(7, 14), dpi=100, facecolor='w', edgecolor='k')
# 400 m
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.9, hspace=0.2, wspace=0.2)
ax1 = fig.add_subplot(3, 1, 1)
sm.taylor_diagram(stdev_400m, rmse_400m, r_400m, markerColor='k', markerSize=10, alpha=0.0, markerLegend='off',
                  tickRMS=np.arange(0, 0.15, 0.03), colRMS='tab:green', styleRMS=':', widthRMS=1.0, titleRMS='on',
                  titleRMSDangle=40.0, showlabelsRMS='on', tickSTD=np.arange(0, 0.12, 0.03), axismax=0.12,
                  colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                  colCOR='tab:blue', styleCOR='--', widthCOR=1.0, titleCOR='on')
plt.xticks(np.arange(0, 0.15, 0.05))
ax1.text(0.1, 0.12, '400 m', fontsize=16, fontweight='bold')

# 1 km
ax2 = fig.add_subplot(3, 1, 2)
sm.taylor_diagram(stdev_1km, rmse_1km, r_1km, markerColor='k', markerSize=10, alpha=0.0, markerLegend='off',
                  tickRMS=np.arange(0, 0.15, 0.03), colRMS='tab:green', styleRMS=':', widthRMS=1.0, titleRMS='on',
                  titleRMSDangle=40.0, showlabelsRMS='on', tickSTD=np.arange(0, 0.12, 0.03), axismax=0.12,
                  colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                  colCOR='tab:blue', styleCOR='--', widthCOR=1.0, titleCOR='on')
plt.xticks(np.arange(0, 0.15, 0.05))
ax2.text(0.1, 0.12, '1 km', fontsize=16, fontweight='bold')

# 9 km
ax3 = fig.add_subplot(3, 1, 3)
sm.taylor_diagram(stdev_9km, rmse_9km, r_9km, markerColor='k', markerSize=10, alpha=0.0, markerLegend='off',
                  tickRMS=np.arange(0, 0.15, 0.03), colRMS='tab:green', styleRMS=':', widthRMS=1.0, titleRMS='on',
                  titleRMSDangle=40.0, showlabelsRMS='on', tickSTD=np.arange(0, 0.12, 0.03), axismax=0.12,
                  colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                  colCOR='tab:blue', styleCOR='--', widthCOR=1.0, titleCOR='on')
plt.xticks(np.arange(0, 0.15, 0.05))
ax3.text(0.1, 0.12, '9 km', fontsize=16, fontweight='bold')

plt.savefig(path_results + '/validation/' + 'td.png')


# 5.2 Plot 400 m
stdev_400m_3net = np.concatenate((stdev_400m[0:88], stdev_400m[255:]))
rmse_400m_3net = np.concatenate((rmse_400m[0:88], rmse_400m[255:]))
r_400m_3net = np.concatenate((r_400m[0:88], r_400m[255:]))

fig = plt.figure(figsize=(5, 5), dpi=200, facecolor='w', edgecolor='k')
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
# ax1 = fig.add_subplot(3, 1, 1)
sm.taylor_diagram(stdev_400m_3net, rmse_400m_3net, r_400m_3net, markerColor='k', markerSize=10, alpha=0.0, markerLegend='off',
                  tickRMS=np.arange(0, 0.15, 0.03), colRMS='tab:green', styleRMS=':', widthRMS=1.0, titleRMS='on',
                  titleRMSDangle=40.0, showlabelsRMS='on', tickSTD=np.arange(0, 0.12, 0.03), axismax=0.12,
                  colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                  colCOR='tab:blue', styleCOR='--', widthCOR=1.0, titleCOR='on')
plt.xticks(np.arange(0, 0.15, 0.05))
# plt.text(0.1, 0.12, '400 m', fontsize=16, fontweight='bold')

plt.savefig(path_results + '/validation/' + 'td_400m.png')

########################################################################################################################
# 6 Classify the stations
df_stat_1km = pd.read_excel(path_results + '/validation/stat_1km.xlsx', index_col=0)






