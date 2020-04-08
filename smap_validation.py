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
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

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
# Path of processed data
path_processed = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
#Path of SMAP Mat-files
path_matfile = '/Volumes/MyPassport/SMAP_Project/Datasets/CONUS'
# Path of Results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200317'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2018-12-31'
year = 2018 - 2015 + 1

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

daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=1)

ind_init = daysofmonth_seq_cumsum[2, :]
ind_end = daysofmonth_seq_cumsum[8, :] - 1
ind_gpm = np.stack((ind_init, ind_end), axis=1)
ind_gpm[0, :] = ind_gpm[0, :] - 90

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'lat_world_geo_10km', 'lon_world_geo_10km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


########################################################################################################################
# 1. Read SMAP 1 km SM data CONUS
# lat/lon of SMAP 1 km

# lat_ease_conus_9km = scipy.io.loadmat(path_matfile + '/parameters.mat')['lat_ease_conus_9km']
# lon_ease_conus_9km = scipy.io.loadmat(path_matfile + '/parameters.mat')['lon_ease_conus_9km']
# lat_ease_conus_1km = scipy.io.loadmat(path_matfile + '/parameters.mat')['lat_ease_conus_1km']
# lon_ease_conus_1km = scipy.io.loadmat(path_matfile + '/parameters.mat')['lon_ease_conus_1km']
#
# var_name = ['lat_ease_conus_9km', 'lon_ease_conus_9km', 'lat_ease_conus_1km', 'lon_ease_conus_1km']
# with h5py.File(path_matfile + '/coords.hdf5', 'w') as f:
#     for x in var_name:
#         f.create_dataset(x, data=eval(x))
# f.close()

# Load in the saved parameters
f_mat = h5py.File(path_matfile + '/coords.hdf5', 'r')
varname_list = list(f_mat.keys())

for x in range(len(varname_list)):
    var_obj = f_mat[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f_mat.close()



# 1.1 Load the site lat/lon from excel files and Locate the SMAP 1/9 km SM positions by lat/lon of in-situ data

# Find the indices of the days between April - Sepetember
month_list = np.array([int(date_seq[x][4:6]) for x in range(len(date_seq))])
month_list_ind = np.where((month_list >= 4) & (month_list <= 9))[0]
month_list_ind = month_list_ind + 2 #First two columns are lat/lon

ismn_list = sorted(glob.glob(path_processed + '/[A-Z]*.xlsx'))

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


########################################################################################################################
# 1.2 Extract 1 km / 9 km SMAP by lat/lon

# Locate the SM pixel positions
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
stn_row_9km_ind_all = []
stn_col_9km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_ease_conus_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_ease_conus_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    stn_row_9km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_ease_conus_9km)).item()
    stn_col_9km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_ease_conus_9km)).item()
    stn_row_9km_ind_all.append(stn_row_9km_ind)
    stn_col_9km_ind_all.append(stn_col_9km_ind)
    del(stn_row_1km_ind, stn_col_1km_ind, stn_row_9km_ind, stn_col_9km_ind)

# 1 km
smap_matfile_list = sorted(glob.glob((path_matfile + '/*sm_1km_ds*')))

smap_mat_1km_am_ext = []
smap_mat_1km_pm_ext = []
for ife in range(len(smap_matfile_list)):
    smap_matfile = h5py.File(smap_matfile_list[ife], 'r')
    var_list = list(smap_matfile.keys())
    smap_mat_1km_am = smap_matfile[var_list[0]]
    smap_mat_1km_pm = smap_matfile[var_list[1]]
    smap_mat_1km_am = np.transpose(smap_mat_1km_am, (2, 1, 0))
    smap_mat_1km_pm = np.transpose(smap_mat_1km_pm, (2, 1, 0))
    smap_mat_1km_am_ext_1month = smap_mat_1km_am[stn_row_1km_ind_all, stn_col_1km_ind_all, :]
    smap_mat_1km_pm_ext_1month = smap_mat_1km_pm[stn_row_1km_ind_all, stn_col_1km_ind_all, :]
    smap_mat_1km_am_ext.append(smap_mat_1km_am_ext_1month)
    smap_mat_1km_pm_ext.append(smap_mat_1km_pm_ext_1month)
    del(smap_matfile, var_list, smap_mat_1km_am, smap_mat_1km_pm, smap_mat_1km_am_ext, smap_mat_1km_pm_ext)
    print(ife)

smap_mat_1km_am_ext = np.concatenate(smap_mat_1km_am_ext, axis=1)
smap_mat_1km_pm_ext = np.concatenate(smap_mat_1km_pm_ext, axis=1)


# 9 km
smap_mat_9km = h5py.File(path_matfile + '/smap_read_9km.mat', 'r')
var_list = list(smap_mat_9km.keys())
smap_mat_9km_am = smap_mat_9km[var_list[8]]
smap_mat_9km_pm = smap_mat_9km[var_list[9]]
smap_mat_9km_am = np.transpose(smap_mat_9km_am, (2, 1, 0))
smap_mat_9km_pm = np.transpose(smap_mat_9km_pm, (2, 1, 0))
smap_mat_9km_am_ext = smap_mat_9km_am[stn_row_9km_ind_all, stn_col_9km_ind_all, :]
smap_mat_9km_pm_ext = smap_mat_9km_pm[stn_row_9km_ind_all, stn_col_9km_ind_all, :]

var_name_val = ['smap_mat_1km_am_ext', 'smap_mat_1km_pm_ext', 'smap_mat_9km_am_ext', 'smap_mat_9km_pm_ext']
with h5py.File(path_matfile + '/smap_validation_conus.hdf5', 'w') as f:
    for x in var_name_val:
        f.create_dataset(x, data=eval(x))
f.close()


########################################################################################################################
# 2.1 single plots
# Load in the saved parameters
f_mat = h5py.File(path_matfile + '/smap_validation_conus.hdf5', 'r')
varname_list = list(f_mat.keys())
for x in range(len(varname_list)):
    var_obj = f_mat[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f_mat.close()

ismn_sm_am = np.array(df_table_am_all.iloc[:, 1:])
ismn_sm_pm = np.array(df_table_pm_all.iloc[:, 1:])

# Make the plots
stat_array_1km = []
stat_array_9km = []
ind_slc_all = []
for ist in range(ismn_sm_am.shape[0]):

    x = ismn_sm_am[ist, :]
    y1 = smap_mat_1km_am_ext[ist, :]
    y2 = smap_mat_9km_am_ext[ist, :]
    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]

    if len(ind_nonnan) > 5:
        x = x[ind_nonnan]
        y1 = y1[ind_nonnan]
        y2 = y2[ind_nonnan]

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

        if r_sq_1 >= 0.1:

            fig = plt.figure(figsize=(11, 6.5))
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            ax = fig.add_subplot(111)

            ax.scatter(x, y1, s=10, c='m', marker='s', label='1 km')
            ax.scatter(x, y2, s=10, c='b', marker='o', label='9 km')
            ax.plot(x, intercept_1+slope_1*x, '-', color='m')
            ax.plot(x, intercept_2+slope_2*x, '-', color='b')

            plt.xlim(0, 0.4)
            ax.set_xticks(np.arange(0, 0.5, 0.1))
            plt.ylim(0, 0.4)
            ax.set_yticks(np.arange(0, 0.5, 0.1))
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            plt.grid(linestyle='--')
            plt.legend(loc='upper left')
            # plt.show()
            plt.savefig(path_results + '/single/' + df_table_am_all['network'][ist] + '_' + df_table_am_all.index[ist]  + '_(' + str(ist) + ')' + '.png')
            plt.close(fig)
            stat_array_1km.append(stat_array_1)
            stat_array_9km.append(stat_array_2)
            ind_slc_all.append(ist)
            print(ist)
            del(stat_array_1, stat_array_2)

        else:
            pass

    else:
        pass


stat_array_1km = np.array(stat_array_1km)
stat_array_9km = np.array(stat_array_9km)
id = np.array(ind_slc_all)
id = np.expand_dims(id, axis=1)

columns_validation = ['id', 'number', 'r_sq', 'ubrmse', 'bias', 'p_value', 'conf_int']
index_validation = df_coords.index[ind_slc_all]

stat_array_1km = np.concatenate((id, stat_array_1km), axis=1)
stat_array_9km = np.concatenate((id, stat_array_9km), axis=1)
df_stat_1km = pd.DataFrame(stat_array_1km, columns=columns_validation, index=index_validation)
df_stat_1km = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_1km], axis=1)
df_stat_9km = pd.DataFrame(stat_array_9km, columns=columns_validation, index=index_validation)
df_stat_9km = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_9km], axis=1)
writer_1km = pd.ExcelWriter(path_results + '/stat_1km.xlsx')
writer_9km = pd.ExcelWriter(path_results + '/stat_9km.xlsx')
df_stat_1km.to_excel(writer_1km)
df_stat_9km.to_excel(writer_9km)
writer_1km.save()
writer_9km.save()

# Read selected sites
df_slc_sites = pd.read_excel(path_results + '/validation_selected.xlsx', index_col=0)
slc_sites_ind = np.array(df_slc_sites['id'])
slc_coords = df_coords.iloc[slc_sites_ind][:]
slc_network = df_table_am_all.iloc[slc_sites_ind, 0]
slc_coords = pd.concat([slc_network, slc_coords], axis=1)
slc_coords.to_csv(path_results + '/slc_coords.csv', index=True)


########################################################################################################################
# 2.2 Subplots
# Site ID
# COSMOS: 0, 11, 25, 28, 34, 36, 42, 44
# SCAN: 250, 274, 279, 286, 296, 351, 362, 383
# SOILSCAPE: 860, 861, 870, 872, 896, 897, 904, 908
# USCRN: 918, 926, 961, 991, 1000，1002, 1012, 1016

# Load in the saved parameters
f_mat = h5py.File(path_matfile + '/smap_validation_conus.hdf5', 'r')
varname_list = list(f_mat.keys())
for x in range(len(varname_list)):
    var_obj = f_mat[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f_mat.close()

# os.chdir(path_results + '/single')
ismn_sm_am = np.array(df_table_am_all.iloc[:, 1:])
ismn_sm_pm = np.array(df_table_pm_all.iloc[:, 1:])

# Make the plot
site_ind = [[0, 11, 25, 28, 34, 36, 42, 44], [250, 274, 279, 286, 296, 351, 362, 383],
            [860, 861, 870, 872, 896, 897, 904, 908], [918, 926, 961, 991, 1000, 1002, 1012, 1016]]
network_name = ['COSMOS', 'SCAN', 'SOILSCAPE', 'USCRN']

for inw in range(len(site_ind)):

    fig = plt.figure(figsize=(11, 8))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.92, hspace=0.25, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = ismn_sm_am[site_ind[inw][ist], :]
        y1 = smap_mat_1km_am_ext[site_ind[inw][ist], :]
        y2 = smap_mat_9km_am_ext[site_ind[inw][ist], :]
        ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        x = x[ind_nonnan]
        y1 = y1[ind_nonnan]
        y2 = y2[ind_nonnan]

        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(x, y1)
        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(x, y2)

        ax = fig.add_subplot(4, 2, ist+1)
        ax.scatter(x, y1, s=10, c='m', marker='s', label='1 km')
        ax.scatter(x, y2, s=10, c='b', marker='o', label='9 km')
        ax.plot(x, intercept_1+slope_1*x, '-', color='m')
        ax.plot(x, intercept_2+slope_2*x, '-', color='b')

        plt.xlim(0, 0.4)
        ax.set_xticks(np.arange(0, 0.5, 0.1))
        plt.ylim(0, 0.4)
        ax.set_yticks(np.arange(0, 0.5, 0.1))
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        plt.grid(linestyle='--')
        plt.legend(loc='upper right')
        ax.text(0.02, 0.35, df_table_am_all.index[site_ind[inw][ist]].replace('_', ' '), fontsize=13)

    fig.text(0.52, 0.01, 'In Situ SM ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.03, 0.4, 'SMAP SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=21, y=0.98, fontweight='bold')
    plt.show()

    plt.savefig(path_results + '/subplots/' + network_name[inw] + '.png')
    plt.close(fig)


########################################################################################################################
# 3. Time-series plots
# 3.1 Locate the corresponding GPM 10 km data located by lat/lon of in-situ data

df_slc_coords = pd.read_csv(path_results + '/slc_coords.csv', index_col=0)
slc_coords = np.array(df_slc_coords.iloc[:, 1:])

stn_row_10km_ind_all = []
stn_col_10km_ind_all = []
for ist in range(slc_coords.shape[0]):
    stn_row_10km_ind = np.argmin(np.absolute(slc_coords[ist, 0] - lat_world_geo_10km)).item()
    stn_col_10km_ind = np.argmin(np.absolute(slc_coords[ist, 1] - lon_world_geo_10km)).item()
    stn_row_10km_ind_all.append(stn_row_10km_ind)
    stn_col_10km_ind_all.append(stn_col_10km_ind)
    del(stn_row_10km_ind, stn_col_10km_ind)

# Extract the GPM data by indices
gpm_precip_ext_all = []
for iyr in range(len(yearname)-1):

    f_gpm = h5py.File(path_model + '/gpm_precip_' + str(yearname[iyr]) + '.hdf5', 'r')
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

gpm_precip_ext_all = [gpm_precip_ext_all[x][:, ind_gpm[x, 0]:ind_gpm[x, 1]+1] for x in range(len(gpm_precip_ext_all))]
gpm_precip_ext_all = np.concatenate(gpm_precip_ext_all, axis=1)

index = df_slc_sites.index
columns = df_table_am_all.columns[1:]
df_gpm_precip_ext = pd.DataFrame(gpm_precip_ext_all, index=index, columns=columns)
df_gpm_precip_ext.to_csv(path_results + '/gpm_precip_ext.csv', index=True)


# 3.2 Make the time-series plots

df_gpm_precip = pd.read_csv(path_results + '/gpm_precip_ext.csv', index_col=0)
gpm_precip_ext = np.array(df_gpm_precip)

site_ind = [[0, 11, 25, 28, 34, 36, 42, 44], [250, 274, 279, 286, 296, 351, 362, 383],
            [860, 861, 870, 872, 896, 897, 904, 908], [918, 926, 961, 991, 1000, 1002, 1012, 1016]]
network_name = ['COSMOS', 'SCAN', 'SOILSCAPE', 'USCRN']


# Find the indices from df_gpm_precip
df_gpm_precip_ind = [df_gpm_precip.index.get_loc(df_table_am_all.index[site_ind[y][x]]) for y in range(len(site_ind)) for x in range(len(site_ind[y]))]
df_gpm_precip_ind = [df_gpm_precip_ind[:8], df_gpm_precip_ind[8:16], df_gpm_precip_ind[16:24], df_gpm_precip_ind[24:]]

for inw in range(len(site_ind)):

    fig = plt.figure(figsize=(13, 8))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.92, hspace=0.35, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = ismn_sm_am[site_ind[inw][ist], :]
        y1 = smap_mat_1km_am_ext[site_ind[inw][ist], :]
        y2 = smap_mat_9km_am_ext[site_ind[inw][ist], :]
        z = gpm_precip_ext[df_gpm_precip_ind[inw][ist], :]
        # ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        # x = x[ind_nonnan]
        # y1 = y1[ind_nonnan]
        # y2 = y2[ind_nonnan]
        # z = z[ind_nonnan]

        ax = fig.add_subplot(8, 1, ist+1)
        lns1 = ax.plot(x, c='k', marker='s', label='In-situ', markersize=2, linestyle='None')
        lns2 = ax.plot(y1, c='m', marker='s', label='1 km', markersize=2, linestyle='None')
        lns3 = ax.plot(y2, c='b', marker='o', label='9 km', markersize=2, linestyle='None')
        ax.text(15, 0.35, df_table_am_all.index[site_ind[inw][ist]].replace('_', ' '), fontsize=11)

        plt.xlim(0, len(x))
        ax.set_xticks(np.arange(0, len(x)+2, (len(x)+2)//4))
        ax.set_xticklabels([])
        labels = ['2015', '2016', '2017', '2018']
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
        lns4 = ax2.bar(np.arange(len(x)), z, width=0.8, color='royalblue', label='Precip')
        ax2.tick_params(axis='y', labelsize=10)

    # add all legends together
    handles = lns1+lns2+lns3+[lns4]
    labels = [l.get_label() for l in handles]

    # handles, labels = ax.get_legend_handles_labels()
    plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 10))
    fig.text(0.5, 0.01, 'Days', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.4, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.96, 0.4, 'GPM Precip (mm/day)', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=19, y=0.97, fontweight='bold')
    plt.savefig(path_results + '/subplots/' + network_name[inw] + '_tseries' + '.png')
    plt.close(fig)



# 3.3 Make auto-correlation plots

site_ind = [[34, 44], [250, 383], [861, 872], [926, 1016]]

for inw in [0, 1, 3]:#range(len(site_ind)):

    for ist in range(len(site_ind[inw])):

        x = ismn_sm_am[site_ind[inw][ist], :]
        y1 = smap_mat_1km_am_ext[site_ind[inw][ist], :]
        y2 = smap_mat_9km_am_ext[site_ind[inw][ist], :]
        array = [x, y1, y2]

        fig = plt.figure(figsize=(13, 8))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.92, hspace=0.25, wspace=0.25)
        for iar in range(len(array)):
            arr1 = array[iar][:183*1]
            ax1 = fig.add_subplot(4, 3, 1+iar)
            # fig = plot_acf(arr1[~np.isnan(arr1)], lags=len(arr1[~np.isnan(arr1)])-1, ax=ax1, title=None)
            fig = plot_acf(arr1[~np.isnan(arr1)], lags=10, ax=ax1, title=None)
            arr2 = array[iar][183*1:183*2]
            ax2 = fig.add_subplot(4, 3, 4+iar)
            fig = plot_acf(arr2[~np.isnan(arr2)], lags=10, ax=ax2, title=None)
            arr3 = array[iar][183*2:183*3]
            ax3 = fig.add_subplot(4, 3, 7+iar)
            fig = plot_acf(arr3[~np.isnan(arr3)], lags=10, ax=ax3, title=None)
            arr4 = array[iar][183*3:183*4]
            ax4 = fig.add_subplot(4, 3, 10+iar)
            fig = plot_acf(arr4[~np.isnan(arr4)], lags=10, ax=ax4, title=None)
        fig.text(0.03, 0.5, '$\mathregular{R^2}$', rotation='vertical', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.01, 'Lags', ha='center', fontsize=16, fontweight='bold')
        fig.text(0.21, 0.94, 'In-situ', ha='center', fontsize=12, fontweight='bold')
        fig.text(0.5, 0.94, '1 km', ha='center', fontsize=12, fontweight='bold')
        fig.text(0.8, 0.94, '9 km', ha='center', fontsize=12, fontweight='bold')
        fig.text(0.93, 0.8, '2015', rotation='vertical', fontsize=12)
        fig.text(0.93, 0.6, '2016', rotation='vertical', fontsize=12)
        fig.text(0.93, 0.4, '2017', rotation='vertical', fontsize=12)
        fig.text(0.93, 0.19, '2018', rotation='vertical', fontsize=12)
        plt.suptitle(df_table_am_all.index[site_ind[inw][ist]], fontsize=15, y=0.99, fontweight='bold')
        plt.savefig(path_results + '/subplots/' + df_table_am_all.index[site_ind[inw][ist]] + '_acf' + '.png')
        plt.close(fig)



# 3.4 Calculate R2 of autocorrelation at lag=1
# Find the indices of the 100 selected sites

df_slc_coords = pd.read_csv(path_results + '/slc_coords.csv', index_col=0)
slc_coords = np.array(df_slc_coords.iloc[:, 1:])

df_slc_ind = [df_table_am_all.index.get_loc(df_slc_coords.index[x]) for x in range(len(df_slc_coords.index))]

df_table_am_slc = df_table_am_all.iloc[df_slc_ind]
df_table_am_slc_arr = np.array(df_table_am_slc.iloc[:, 1:])
smap_mat_1km_am_ext_arr = smap_mat_1km_am_ext[df_slc_ind, :]
smap_mat_9km_am_ext_arr = smap_mat_9km_am_ext[df_slc_ind, :]

r_ac_all = []
for ist in range(df_table_am_slc_arr.shape[0]):

    ismm_arr = pd.Series(df_table_am_slc_arr[ist, :], index=None)
    smap_1km_arr = pd.Series(smap_mat_1km_am_ext_arr[ist, :], index=None)
    smap_9km_arr = pd.Series(smap_mat_9km_am_ext_arr[ist, :], index=None)
    stacked_arr = [ismm_arr, smap_1km_arr, smap_9km_arr]

    r_all = []
    for idt in range(len(stacked_arr)):
        r1 = stacked_arr[idt][:183*1].autocorr(lag=1)
        r2 = stacked_arr[idt][183*1:183*2].autocorr(lag=1)
        r3 = stacked_arr[idt][183*2:183*3].autocorr(lag=1)
        r4 = stacked_arr[idt][183*3:183*4].autocorr(lag=1)
        r = np.nanmean([r1, r2, r3, r4])
        r_all.append(r)
        del(r, r1, r2, r3, r4)
    r_all = np.array(r_all)

    r_ac_all.append(r_all)

r_ac_all = np.array(r_ac_all)

df_r_ac_all = pd.DataFrame(r_ac_all, columns=['ISMN', '1km', '9km'])
df_r_ac_all.to_csv(path_results + '/df_r_ac.csv', index=True)


# 3.5 Calculate spatial standard deviation
# Tonzi ranch: 44, 796, 801, 850, 851, 853, 854, 855, 856, 858, 860, 861, 862, 863, 864, 870, 872, 876, 881, 885,
# 892, 895, 896, 897, 900, 901, 904, 905, 906, 907, 908, 909, 911
# San Pedro: 947, 22, 24, 835, 842, 848, 37, 38, 383, 1016
# Okla: 0, 36, 202, 1012
# Colorado: 274, 279, 926, 980

spdev_ind = [[44, 796, 801, 850, 851, 853, 854, 855, 856, 858, 860, 861, 862, 863, 864, 870, 872, 876, 881, 885, 892,
              895, 896, 897, 900, 901, 904, 905, 906, 907, 908, 909, 911], [947, 22, 24, 835, 842, 848, 37, 38, 383, 1016],
             [0, 36, 202, 1012], [274, 279, 926, 980]]
index = ['ismn-avg', '1km-avg', '9km-avg', 'ismn-spdev', '1km-spdev', '9km-spdev'] * 4

stat_mat_all = []
for inw in range(len(spdev_ind)):
    ismn_array = np.array(df_table_am_all.iloc[spdev_ind[inw], 1:])
    smap_1km_array = smap_mat_1km_am_ext[spdev_ind[inw]]
    smap_9km_array = smap_mat_9km_am_ext[spdev_ind[inw]]

    mean_ismn_array = np.nanmean(ismn_array, axis=0)
    spdev_ismn = np.sqrt(np.nanmean((ismn_array - mean_ismn_array)**2, axis=0))
    spdev_ismn[spdev_ismn == 0] = np.nan

    mean_smap_1km_array = np.nanmean(smap_1km_array, axis=0)
    spdev_smap_1km = np.sqrt(np.nanmean((smap_1km_array - mean_smap_1km_array)**2, axis=0))
    spdev_smap_1km[spdev_smap_1km == 0] = np.nan

    mean_smap_9km_array = np.nanmean(smap_9km_array, axis=0)
    spdev_smap_9km = np.sqrt(np.nanmean((smap_9km_array - mean_smap_9km_array)**2, axis=0))
    spdev_smap_9km[spdev_smap_9km == 0] = np.nan

    mean_mat = np.stack([mean_ismn_array, mean_smap_1km_array, mean_smap_9km_array], axis=0)
    spdev_mat = np.stack([spdev_ismn, spdev_smap_1km, spdev_smap_9km], axis=0)

    stat_mat = np.concatenate([mean_mat, spdev_mat], axis=0)
    stat_mat_all.append(stat_mat)
    del(mean_mat, spdev_mat, stat_mat)

spdev_all = np.concatenate(stat_mat_all, axis=0)
df_spdev = pd.DataFrame(spdev_all, index=index)
df_spdev.to_csv(path_results + '/spdev.csv', index=True)


# 3.6 Calculate the percentage of available days between SMAP 1km/9km and ISMN data
# 3.6.1
network_name = ['COSMOS', 'SCAN', 'SOILSCAPE', 'USCRN']
network_ind = [13, 45, 80]

days_avai_9km = np.array([len(np.where(~np.isnan(df_table_am_slc_arr[x, :]) & ~np.isnan(smap_mat_9km_am_ext_arr[x, :]))[0])
             for x in range(df_table_am_slc_arr.shape[0])])
days_notavai_1km = np.array([len(np.where(~np.isnan(df_table_am_slc_arr[x, :])
                                 & ~np.isnan(smap_mat_9km_am_ext_arr[x, :])
                                 & np.isnan(smap_mat_1km_am_ext_arr[x, :]))[0]) for x in range(df_table_am_slc_arr.shape[0])])
days_avai = np.stack((days_avai_9km, days_notavai_1km), axis=0).transpose()


days_avai_group = [np.nanmean(days_avai[:13], axis=0), np.nanmean(days_avai[13:45], axis=0),
                 np.nanmean(days_avai[45:80], axis=0), np.nanmean(days_avai[80:], axis=0)]
df_avai_group = np.stack(days_avai_group, axis=0)

df_avai_group = pd.DataFrame(df_avai_group, index=network_name, columns=['ISMN/9 km SM availability', '1 km SM unavailability'])

ax = df_avai_group.plot.bar(figsize=(12,8), rot=0, color=('grey','black'))
fig = ax.get_figure()
plt.xticks(fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(0.4, 1))
fig.savefig(path_results + '/subplots/' + 'day_avai1.png')
plt.close(fig)

# 3.6.2
cover_notavai_1km = np.array([len(np.where(np.isnan(smap_mat_1km_am_ext_arr[x, :]))[0]) for x in range(smap_mat_1km_am_ext_arr.shape[0])])
cover_notavai_9km = np.array([len(np.where(np.isnan(smap_mat_9km_am_ext_arr[x, :]))[0]) for x in range(smap_mat_9km_am_ext_arr.shape[0])])
cover_avai = np.stack((cover_notavai_1km, cover_notavai_9km), axis=0).transpose()

cover_avai_group = [np.nanmean(cover_avai[:13], axis=0), np.nanmean(cover_avai[13:45], axis=0),
                 np.nanmean(cover_avai[45:80], axis=0), np.nanmean(cover_avai[80:], axis=0)]
df_cover_avai_group = np.stack(cover_avai_group, axis=0)

df_cover_avai_group = pd.DataFrame(df_cover_avai_group, index=network_name, columns=['MODIS', 'SMAP'])

ax = df_cover_avai_group.plot.bar(figsize=(12,8), rot=0, color=('grey','black'))
fig = ax.get_figure()
plt.xticks(fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(0.4, 1))
fig.savefig(path_results + '/subplots/' + 'cover_avai1.png')
plt.close(fig)





