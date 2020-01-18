import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import calendar
import datetime
import glob
import gdal
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

# import pandas as pd
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#########################################################################################################
# (Function 1) Define a function for reading and extracting useful information from each ISMN in-situ data file

def insitu_extraction(filepath):
    # Read each .stm file by line
    with open(filepath, "r") as ins:
        data_list = []
        for line in ins:
            data_list.append(line)
    ins.close()

    # Extract lat/lon, network and station name information from the first line of the current file
    net_name = data_list[0].split()[4]
    stn_name = data_list[0].split()[6]
    stn_lat = float(data_list[0].split()[7])
    stn_lon = float(data_list[0].split()[8])

    # Extract 6 AM/PM SM from current file
    sm_array = np.empty((2, len(date_seq)), dtype='float32')  # 2-dim for storing AM/PM overpass SM
    sm_array[:] = np.nan
    # sm_array = np.copy(sm_array_init)
    for itm in range(len(amsr2_overpass)):
        datatime_match_ind = [data_list.index(i) for i in data_list if amsr2_overpass[itm] in i]
        datatime_match = [data_list[datatime_match_ind[i]] for i in range(len(datatime_match_ind))]
        datatime_match_date = [datatime_match[i].split()[0].replace('/', '') for i in range(len(datatime_match))]

        datatime_match_date_ind = [datatime_match_date.index(item) for item in datatime_match_date if item in date_seq]
        datatime_match_date_seq_ind = [date_seq.index(item) for item in datatime_match_date if item in date_seq]

        if len(datatime_match_date_ind) != 0:
            sm_array_ext = [float(datatime_match[datatime_match_date_ind[i]].split()[12])
                            for i in range(len(datatime_match_date_ind))]  # Find the data values from the in situ data file
            sm_array[itm, datatime_match_date_seq_ind] = sm_array_ext  # Fill the data values to the corresponding place of date_seq
        else:
            sm_array_ext = []
            pass

        del(datatime_match_ind, datatime_match, datatime_match_date, datatime_match_date_ind, sm_array_ext)

    return net_name, stn_name, stn_lat, stn_lon, sm_array


####################################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of processed data
path_procdata = '/Users/binfang/Downloads/Processing/processed_data'
# Path of downscaled SM
path_amsr2_sm_ds = '/Users/binfang/Downloads/Processing/processed_data/AMSR2_ds'
# Path of in-situ data
path_insitu = '/Users/binfang/Documents/SMAP_Project/data/insitu_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200115'
# Path of processed data (External hard drive)
path_procdata_ext = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
amsr2_sm_9km_name = ['amsr2_sm_9km_am_slice', 'amsr2_sm_9km_pm_slice']
region_name = ['Danube', 'USA']
amsr2_overpass = ['01:00', '13:00']
yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2019-01-01'
end_date = '2019-12-31'
# year = 2018 - 1981 + 1

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

# 1. Extract SM in situ data from ISMN .stm files

net_name_all = []
stn_name_all = []
stn_lat_all = []
stn_lon_all = []
sm_array_all = []
for ire in range(len(region_name)):

    os.chdir(path_insitu + '/' + region_name[ire])
    insitu_files = sorted(glob.glob('*.stm'))

    # Extract data from each region
    net_name_1reg = []
    stn_name_1reg = []
    stn_lat_1reg = []
    stn_lon_1reg = []
    sm_array_1reg = []
    for idt in range(len(insitu_files)):
        net_name, stn_name, stn_lat, stn_lon, sm_array = insitu_extraction(insitu_files[idt])
        net_name_1reg.append(net_name)
        stn_name_1reg.append(stn_name)
        stn_lat_1reg.append(stn_lat)
        stn_lon_1reg.append(stn_lon)
        sm_array_1reg.append(sm_array)
        del(net_name, stn_name, stn_lat, stn_lon, sm_array)

    net_name_all.append(net_name_1reg)
    stn_name_all.append(stn_name_1reg)
    stn_lat_all.append(stn_lat_1reg)
    stn_lon_all.append(stn_lon_1reg)
    sm_array_all.append(sm_array_1reg)
    del (net_name_1reg, stn_name_1reg, stn_lat_1reg, stn_lon_1reg, sm_array_1reg)

    print(ire)



####################################################################################################################################
# 2.1 Locate the AMSR2 1/10 km SM positions by lat/lon of in-situ data

stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
stn_row_9km_ind_all = []
stn_col_9km_ind_all = []
for ire in range(len(stn_lat_all)):

    stn_row_1km_ind_1reg = []
    stn_col_1km_ind_1reg = []
    stn_row_9km_ind_1reg = []
    stn_col_9km_ind_1reg = []
    for idt in range(len(stn_lat_all[ire])):
        stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[ire][idt] - lat_world_ease_1km)).item()
        stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[ire][idt] - lon_world_ease_1km)).item()
        stn_row_1km_ind_1reg.append(stn_row_1km_ind)
        stn_col_1km_ind_1reg.append(stn_col_1km_ind)
        stn_row_9km_ind = np.argmin(np.absolute(stn_lat_all[ire][idt] - lat_world_ease_9km)).item()
        stn_col_9km_ind = np.argmin(np.absolute(stn_lon_all[ire][idt] - lon_world_ease_9km)).item()
        stn_row_9km_ind_1reg.append(stn_row_9km_ind)
        stn_col_9km_ind_1reg.append(stn_col_9km_ind)
        del(stn_row_1km_ind, stn_col_1km_ind, stn_row_9km_ind, stn_col_9km_ind)

    stn_row_1km_ind_all.append(stn_row_1km_ind_1reg)
    stn_col_1km_ind_all.append(stn_col_1km_ind_1reg)
    stn_row_9km_ind_all.append(stn_row_9km_ind_1reg)
    stn_col_9km_ind_all.append(stn_col_9km_ind_1reg)
    del(stn_row_1km_ind_1reg, stn_col_1km_ind_1reg, stn_row_9km_ind_1reg, stn_col_9km_ind_1reg)

########################################################################################################################

# 3. Extract the amsr2 1/9 km SM by the indexing files

year_plt = yearname[4]

# Load in amsr2 1 km SM

src_tf_ext_all = []
for iyr in [year_plt]:  # range(yearname):

    os.chdir(path_amsr2_sm_ds + '/' + str(iyr))
    tif_files = sorted(glob.glob('*.tif'))

    src_tf_ext_1yr = []
    for idt in range(len(tif_files)):

        src_tf_ext_1day = []
        src_tf = gdal.Open(tif_files[idt])
        for ire in range(len(stn_lat_all)):

            src_tf_ext_1reg = src_tf.ReadAsArray()[:, stn_row_1km_ind_all[ire], stn_col_1km_ind_all[ire]]
            src_tf_ext_1day.append(src_tf_ext_1reg)
            del(src_tf_ext_1reg)

        src_tf_ext_1yr.append(src_tf_ext_1day)
        print(tif_files[idt])
        del(src_tf_ext_1day)

    src_tf_ext_all.append(src_tf_ext_1yr)
    del(src_tf_ext_1yr)



doy_file = [os.path.splitext(tif_files[x])[0].split('_')[-1] for x in range(len(tif_files))]
match_date_seq_ind = [date_seq_doy.index(item) for item in doy_file if item in date_seq_doy]

for iyr in [0]:

    sm_array_1km_all = []
    for ire in range(len(sm_array_all)):  # each region

        sm_array_1km_1re = np.empty((len(sm_array_all[ire]), 2, len(date_seq))) # initial a matrix for matched SM to 1 km data
        sm_array_1km_1re[:] = np.nan

        for ist in range(len(sm_array_all[ire])):  # each station
            for idt in range(len(match_date_seq_ind)):  # each day
                sm_array_1km_1re[ist, :, match_date_seq_ind[idt]] = src_tf_ext_all[iyr][idt][ire][:, ist]

        sm_array_1km_all.append(sm_array_1km_1re)
        del(sm_array_1km_1re)


sm_array_1km_reg1 = sm_array_1km_all[0]
sm_array_1km_reg2 = sm_array_1km_all[1]

os.chdir(path_procdata)
var_name = ['sm_array_1km_reg1', 'sm_array_1km_reg2']
with h5py.File('amsr2_sm_match.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()




# Load in amsr2 9 km SM

sm_array_9km_reg1 = np.empty((2, len(stn_row_9km_ind_all[0]), 0))
sm_array_9km_reg1[:] = np.nan
sm_array_9km_reg2 = np.empty((2, len(stn_row_9km_ind_all[1]), 0))
sm_array_9km_reg2[:] = np.nan
append_array_reg1 = np.empty((len(stn_row_9km_ind_all[0]), 2, 365-304))
append_array_reg2 = np.empty((len(stn_row_9km_ind_all[1]), 2, 365-304))

for iyr in [4]: #range(len(yearname)):

    for imo in range(len(monthname)):

        # Load in amsr2 9km SM data
        amsr2_file_path = path_procdata + '/amsr2_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

        # Find the if the file exists in the directory
        if os.path.exists(amsr2_file_path) == True:

            f_amsr2_9km = h5py.File(amsr2_file_path, "r")
            varname_list_amsr2 = list(f_amsr2_9km.keys())

            sm_array_9km_reg = []
            for ire in range(len(stn_row_9km_ind_all)):

                for x in range(len(varname_list_amsr2)):
                    var_obj = f_amsr2_9km[varname_list_amsr2[x]][()]
                    linear_ind = np.ravel_multi_index([stn_row_9km_ind_all[ire], stn_col_9km_ind_all[ire]], (var_obj.shape[0], var_obj.shape[1]))
                    var_obj = np.reshape(var_obj, (var_obj.shape[0]*var_obj.shape[1], var_obj.shape[2])) # Convert from 3D to 2D
                    var_obj = var_obj[linear_ind, :]

                    exec(amsr2_sm_9km_name[x] + '= var_obj')
                    del(var_obj)

                amsr2_sm_9km_slice = np.stack((amsr2_sm_9km_am_slice, amsr2_sm_9km_pm_slice))
                sm_array_9km_reg.append(amsr2_sm_9km_slice)
                del (amsr2_sm_9km_am_slice, amsr2_sm_9km_pm_slice, amsr2_sm_9km_slice)

            sm_array_9km_reg1 = np.append(sm_array_9km_reg1, sm_array_9km_reg[0], axis=2)
            sm_array_9km_reg2 = np.append(sm_array_9km_reg2, sm_array_9km_reg[1], axis=2)

            print(imo)
            f_amsr2_9km.close()


        else:
            pass


sm_array_9km_reg1 = np.transpose(sm_array_9km_reg1, (1, 0, 2))
sm_array_9km_reg2 = np.transpose(sm_array_9km_reg2, (1, 0, 2))

sm_array_9km_reg1 = np.concatenate((sm_array_9km_reg1, append_array_reg1), axis=2)
sm_array_9km_reg2 = np.concatenate((sm_array_9km_reg2, append_array_reg2), axis=2)

os.chdir(path_procdata)
var_name = ['sm_array_9km_reg1', 'sm_array_9km_reg2']
with h5py.File('amsr2_sm_match.hdf5', 'a') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()



########################################################################################################################
# 4. Plot validation results between 1 km, 9 km and in-situ data

# Load in 1 km / 9 km matched data for validation
os.chdir(path_procdata)
f = h5py.File("amsr2_sm_match.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


# 4.1 single plots
os.chdir(path_results + '/scatterplots')

for ire in range(2):

    if ire == 0:

        for ist in range(len(sm_array_all[ire])):

            x = np.stack((sm_array_all[ire][ist][0, :], sm_array_all[ire][ist][1, :])).flatten()
            y1 = np.stack((sm_array_1km_reg1[ist, 0, :], sm_array_1km_reg1[ist, 1, :])).flatten()
            y2 = np.stack((sm_array_9km_reg1[ist, 0, :], sm_array_9km_reg1[ist, 1, :])).flatten()

            ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
            if len(ind_nonnan) > 5:
                x = x[ind_nonnan]
                y1 = y1[ind_nonnan]
                y2 = y2[ind_nonnan]

                fig = plt.figure(figsize=(11, 6.5))
                fig.subplots_adjust(hspace=0.2, wspace=0.2)
                ax = fig.add_subplot(111)

                ax.scatter(x, y1, s=10, c='m', marker='s', label='1 km')
                ax.scatter(x, y2, s=10, c='b', marker='o', label='9 km')

                coef1, intr1 = np.polyfit(x.squeeze(), y1.squeeze(), 1)
                ax.plot(x.squeeze(), intr1 + coef1 * x.squeeze(), '-', color='m')

                coef2, intr2 = np.polyfit(x.squeeze(), y2.squeeze(), 1)
                ax.plot(x.squeeze(), intr2 + coef2 * x.squeeze(), '-', color='b')

                plt.xlim(0, 0.4)
                ax.set_xticks(np.arange(0, 0.5, 0.1))
                plt.ylim(0, 0.4)
                ax.set_yticks(np.arange(0, 0.5, 0.1))
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
                plt.grid(linestyle='--')
                plt.legend(loc='upper left')
                # plt.show()
                plt.savefig(path_results + '/scatterplots/' + str(ire) + '_' + str(ist) + '.png')
                plt.close(fig)
            else:
                pass

    elif ire == 1:

        for ist in range(len(sm_array_all[ire])):
            x = np.stack((sm_array_all[ire][ist][0, :], sm_array_all[ire][ist][1, :])).flatten()
            y1 = np.stack((sm_array_1km_reg2[ist, 0, :], sm_array_1km_reg2[ist, 1, :])).flatten()
            y2 = np.stack((sm_array_9km_reg2[ist, 0, :], sm_array_9km_reg2[ist, 1, :])).flatten()

            ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]

            if len(ind_nonnan) > 5:
                x = x[ind_nonnan]
                y1 = y1[ind_nonnan]
                y2 = y2[ind_nonnan]

                fig = plt.figure(figsize=(11, 6.5))
                fig.subplots_adjust(hspace=0.2, wspace=0.2)
                ax = fig.add_subplot(111)

                ax.scatter(x, y1, s=10, c='m', marker='s', label='1 km')
                ax.scatter(x, y2, s=10, c='b', marker='o', label='9 km')

                coef1, intr1 = np.polyfit(x.squeeze(), y1.squeeze(), 1)
                ax.plot(x.squeeze(), intr1 + coef1 * x.squeeze(), '-', color='m')
                coef2, intr2 = np.polyfit(x.squeeze(), y2.squeeze(), 1)
                ax.plot(x.squeeze(), intr2 + coef2 * x.squeeze(), '-', color='b')

                plt.xlim(0, 0.4)
                ax.set_xticks(np.arange(0, 0.5, 0.1))
                plt.ylim(0, 0.4)
                ax.set_yticks(np.arange(0, 0.5, 0.1))
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
                plt.grid(linestyle='--')
                plt.legend(loc='upper left')
                # plt.show()
                plt.savefig(path_results + '/scatterplots/' + str(ire) + '_' + str(ist) + '.png')
                plt.close(fig)

            else:
                pass






# 4.2 Subplots

# 4.2.1 Region 1 (Danube)
# reg1_site = np.array([2, 3, 8, 15, 22, 23])
reg1_site = np.array([1, 3, 9, 11, 13, 15])

fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

stat_array_1_all = []
stat_array_2_all = []

for ist in range(6):

    x = np.stack((sm_array_all[0][reg1_site[ist]][0, :], sm_array_all[0][reg1_site[ist]][1, :])).flatten()
    y1 = np.stack((sm_array_1km_reg1[reg1_site[ist], 0, :], sm_array_1km_reg1[reg1_site[ist], 1, :])).flatten()
    y2 = np.stack((sm_array_9km_reg1[reg1_site[ist], 0, :], sm_array_9km_reg1[reg1_site[ist], 1, :])).flatten()

    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
    x = x[ind_nonnan]
    y1 = y1[ind_nonnan]
    y2 = y2[ind_nonnan]

    ax = fig.add_subplot(2, 3, ist+1)

    ax.scatter(x, y1, s=10, c='m', marker='s', label='1 km')
    ax.scatter(x, y2, s=10, c='b', marker='o', label='10 km')

    coef1, intr1 = np.polyfit(x.squeeze(), y1.squeeze(), 1)
    corrcoef_mat_1 = np.corrcoef(x, y1)
    estimated_1 = intr1 + coef1 * x
    r_sq_1 = (corrcoef_mat_1[1, 0]) ** 2
    ubrmse_1 = np.sqrt(np.mean((x - estimated_1) ** 2))
    bias_1 = np.mean(x - y1)
    number_1 = len(y1)
    spdev_0 = np.sqrt(np.mean((x - np.mean(x)) ** 2))
    spdev_1 = np.sqrt(np.mean((y1 - np.mean(y1)) ** 2))
    stat_array_1 = [r_sq_1, ubrmse_1, bias_1, number_1, spdev_1, spdev_0]
    stat_array_1_all.append(stat_array_1)

    ax.plot(x.squeeze(), intr1 + coef1 * x.squeeze(), '-', color='m')

    coef2, intr2 = np.polyfit(x.squeeze(), y2.squeeze(), 1)
    corrcoef_mat_2 = np.corrcoef(x, y2)
    estimated_2 = intr2 + coef2 * x
    r_sq_2 = (corrcoef_mat_2[1, 0]) ** 2
    ubrmse_2 = np.sqrt(np.mean((x - estimated_2) ** 2))
    bias_2 = np.mean(x - y2)
    number_2 = len(y2)
    spdev_0 = np.sqrt(np.mean((x - np.mean(x)) ** 2))
    spdev_2 = np.sqrt(np.mean((y2 - np.mean(y2)) ** 2))
    stat_array_2 = [r_sq_2, ubrmse_2, bias_2, number_2, spdev_2, spdev_0]
    stat_array_2_all.append(stat_array_2)

    ax.plot(x.squeeze(), intr2 + coef2 * x.squeeze(), '-', color='b')

    plt.xlim(0, 0.4)
    ax.set_xticks(np.arange(0, 0.5, 0.1))
    plt.ylim(0, 0.4)
    ax.set_yticks(np.arange(0, 0.5, 0.1))
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    plt.grid(linestyle='--')
    plt.legend(loc='upper left')
    ax.text(0.15, 0.05, net_name_all[0][reg1_site[ist]] + '-' + stn_name_all[0][reg1_site[ist]], fontsize=13)

fig.text(0.52, 0.01, 'In Situ ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.4, 'amsr2 ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
plt.suptitle('Danube River Basin', fontsize=23, y=0.97, fontweight='bold')

plt.savefig(path_results + '/scatterplots/' + 'danube_1' + '.png')
plt.close(fig)

stat_array_1_all = np.asarray(stat_array_1_all)
stat_array_2_all = np.asarray(stat_array_2_all)
stat_array_all = np.concatenate([stat_array_1_all, stat_array_2_all])
np.savetxt(path_results + '/scatterplots/' + 'stat_danube.csv', stat_array_all, delimiter=",", fmt='%f')




# 4.2.2 Region 2 (Mississippi)

reg2_site = np.array([9, 11, 13, 16, 18, 27])

fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

stat_array_1_all = []
stat_array_2_all = []
for ist in range(6):

    x = np.stack((sm_array_all[1][reg2_site[ist]][0, :], sm_array_all[1][reg2_site[ist]][1, :])).flatten()
    y1 = np.stack((sm_array_1km_reg2[reg2_site[ist], 0, :], sm_array_1km_reg2[reg2_site[ist], 1, :])).flatten()
    y2 = np.stack((sm_array_9km_reg2[reg2_site[ist], 0, :], sm_array_9km_reg2[reg2_site[ist], 1, :])).flatten()

    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
    x = x[ind_nonnan]
    y1 = y1[ind_nonnan]
    y2 = y2[ind_nonnan]

    ax = fig.add_subplot(2, 3, ist+1)

    ax.scatter(x, y1, s=10, c='m', marker='s', label='1 km')
    ax.scatter(x, y2, s=10, c='b', marker='o', label='10 km')

    coef1, intr1 = np.polyfit(x.squeeze(), y1.squeeze(), 1)
    corrcoef_mat_1 = np.corrcoef(x, y1)
    estimated_1 = intr1 + coef1 * x
    r_sq_1 = (corrcoef_mat_1[1, 0]) ** 2
    ubrmse_1 = np.sqrt(np.mean((x - estimated_1) ** 2))
    bias_1 = np.mean(x - y1)
    number_1 = len(y1)
    spdev_0 = np.sqrt(np.mean((x - np.mean(x)) ** 2))
    spdev_1 = np.sqrt(np.mean((y1 - np.mean(y1)) ** 2))
    stat_array_1 = [r_sq_1, ubrmse_1, bias_1, number_1, spdev_1, spdev_0]
    stat_array_1_all.append(stat_array_1)

    ax.plot(x.squeeze(), intr1 + coef1 * x.squeeze(), '-', color='m')

    coef2, intr2 = np.polyfit(x.squeeze(), y2.squeeze(), 1)
    corrcoef_mat_2 = np.corrcoef(x, y2)
    estimated_2 = intr2 + coef2 * x
    r_sq_2 = (corrcoef_mat_2[1, 0]) ** 2
    ubrmse_2 = np.sqrt(np.mean((x - estimated_2) ** 2))
    bias_2 = np.mean(x - y2)
    number_2 = len(y2)
    spdev_2 = np.sqrt(np.mean((y2 - np.mean(y2)) ** 2))
    stat_array_2 = [r_sq_2, ubrmse_2, bias_2, number_2, spdev_2, spdev_0]
    stat_array_2_all.append(stat_array_2)

    ax.plot(x.squeeze(), intr2 + coef2 * x.squeeze(), '-', color='b')

    plt.xlim(0, 0.4)
    ax.set_xticks(np.arange(0, 0.5, 0.1))
    plt.ylim(0, 0.4)
    ax.set_yticks(np.arange(0, 0.5, 0.1))
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    plt.grid(linestyle='--')
    plt.legend(loc='upper left')
    ax.text(0.2, 0.05, net_name_all[1][reg2_site[ist]] + '-' + stn_name_all[1][reg2_site[ist]], fontsize=15)

fig.text(0.52, 0.01, 'In Situ ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.4, 'AMSR2 ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
plt.suptitle('USA', fontsize=22, y=0.97, fontweight='bold')

plt.savefig(path_results + '/scatterplots/' + 'miss_1' + '.png')
plt.close(fig)

stat_array_1_all = np.asarray(stat_array_1_all)
stat_array_2_all = np.asarray(stat_array_2_all)
stat_array_all = np.concatenate([stat_array_1_all, stat_array_2_all])
np.savetxt(path_results + '/scatterplots/' + 'stat_miss.csv', stat_array_all, delimiter=",", fmt='%f')


########################################################################################################################
# 5. Time-series plots

# Load in 1 km / 9 km matched data for validation
os.chdir(path_procdata)
f = h5py.File("amsr2_sm_match.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


os.chdir(path_procdata_ext)
f_gpm = h5py.File("gpm_precip_2019.hdf5", "r")
varname_list_gpm = list(f_gpm.keys())

for x in range(len(varname_list_gpm)):
    var_obj = f_gpm[varname_list_gpm[x]][()]
    exec(varname_list_gpm[x] + '= var_obj')
    del(var_obj)
f_gpm.close()


# Locate the corresponding GPM 10 km data located by lat/lon of in-situ data

stn_row_10km_ind_all = []
stn_col_10km_ind_all = []
for ire in range(len(stn_lat_all)):

    stn_row_10km_ind_1reg = []
    stn_col_10km_ind_1reg = []
    for idt in range(len(stn_lat_all[ire])):
        stn_row_10km_ind = np.argmin(np.absolute(stn_lat_all[ire][idt] - lat_world_geo_10km)).item()
        stn_col_10km_ind = np.argmin(np.absolute(stn_lon_all[ire][idt] - lon_world_geo_10km)).item()
        stn_row_10km_ind_1reg.append(stn_row_10km_ind)
        stn_col_10km_ind_1reg.append(stn_col_10km_ind)
        del(stn_row_10km_ind, stn_col_10km_ind)

    stn_row_10km_ind_all.append(stn_row_10km_ind_1reg)
    stn_col_10km_ind_all.append(stn_col_10km_ind_1reg)
    del(stn_row_10km_ind_1reg, stn_col_10km_ind_1reg)


gpm_precip_ext = []
for ire in range(len(stn_row_10km_ind_all)):
    linear_ind = np.ravel_multi_index([stn_row_10km_ind_all[ire], stn_col_10km_ind_all[ire]],
                                   (gpm_precip_10km_2019.shape[0], gpm_precip_10km_2019.shape[1]))
    gpm_precip_res = np.reshape(gpm_precip_10km_2019, (gpm_precip_10km_2019.shape[0] * gpm_precip_10km_2019.shape[1],
                                                gpm_precip_10km_2019.shape[2]))  # Convert from 3D to 2D
    gpm_precip_ext_1reg = gpm_precip_res[linear_ind, :]
    gpm_precip_ext.append(gpm_precip_ext_1reg)



# 5.1 Region 1 (Danube)

reg1_site = np.array([1, 3, 9, 11, 13, 15])

fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

stat_array_1_all = []
stat_array_2_all = []
for ist in range(6):

    x = np.nanmean((sm_array_all[0][reg1_site[ist]][0, :], sm_array_all[0][reg1_site[ist]][1, :]), axis=0)
    y1 = np.nanmean((sm_array_1km_reg1[reg1_site[ist], 0, :], sm_array_1km_reg1[reg1_site[ist], 1, :]), axis=0)
    y2 = np.nanmean((sm_array_9km_reg1[reg1_site[ist], 0, :], sm_array_9km_reg1[reg1_site[ist], 1, :]), axis=0)
    z = gpm_precip_ext[0][reg1_site[ist], :]

    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
    x = x[ind_nonnan]
    y1 = y1[ind_nonnan]
    y2 = y2[ind_nonnan]
    z = z[ind_nonnan]

    ax = fig.add_subplot(2, 3, ist+1)


    lns1 = ax.plot(x, c='k', marker='s', label='In-situ', markersize=5)
    lns2 = ax.plot(y1, c='m', marker='s', label='1 km', markersize=5)
    lns3 = ax.plot(y2, c='b', marker='o', label='10 km', markersize=5)

    plt.xlim(0, len(x))
    ax.set_xticks(np.arange(0, len(x)+2, (len(x)+2)//6))
    plt.ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    ax.tick_params(axis='y', labelsize=10)
    ax.text(6, 0.02, net_name_all[0][reg1_site[ist]] + '-' + stn_name_all[0][reg1_site[ist]], fontsize=10)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 32, 8)
    ax2.invert_yaxis()
    lns4 = ax2.bar(np.arange(len(x)), z, width = 0.8, color='royalblue', label='Precip')
    ax2.tick_params(axis='y', labelsize=10)

# add all legends together
handles = lns1+lns2+lns3+[lns4]
labels = [l.get_label() for l in handles]

# handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 2.1))
fig.text(0.52, 0.01, 'Days', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.4, 'amsr2 SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.95, 0.4, 'GPM Precip (mm/day)', rotation='vertical', fontsize=16, fontweight='bold')
plt.suptitle('Danube River Basin', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/scatterplots/' + 'dan_1_tseries' + '.png')
plt.close(fig)


# 5.2 Region 2 (Mississippi)

reg2_site = np.array([9, 11, 13, 16, 18, 27])

fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

stat_array_1_all = []
stat_array_2_all = []
for ist in range(6):

    x = np.nanmean((sm_array_all[1][reg2_site[ist]][0, :], sm_array_all[1][reg2_site[ist]][1, :]), axis=0)
    y1 = np.nanmean((sm_array_1km_reg2[reg2_site[ist], 0, :], sm_array_1km_reg2[reg2_site[ist], 1, :]), axis=0)
    y2 = np.nanmean((sm_array_9km_reg2[reg2_site[ist], 0, :], sm_array_9km_reg2[reg2_site[ist], 1, :]), axis=0)
    z = gpm_precip_ext[1][reg2_site[ist], :]

    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
    x = x[ind_nonnan]
    y1 = y1[ind_nonnan]
    y2 = y2[ind_nonnan]
    z = z[ind_nonnan]

    ax = fig.add_subplot(2, 3, ist+1)


    lns1 = ax.plot(x, c='k', marker='s', label='In-situ', markersize=5)
    lns2 = ax.plot(y1, c='m', marker='s', label='1 km', markersize=5)
    lns3 = ax.plot(y2, c='b', marker='o', label='10 km', markersize=5)

    plt.xlim(0, len(x))
    ax.set_xticks(np.arange(0, len(x)+2, (len(x)+2)//6))
    plt.ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    ax.tick_params(axis='y', labelsize=10)
    ax.text(6, 0.02, net_name_all[1][reg2_site[ist]] + '-' + stn_name_all[1][reg2_site[ist]], fontsize=10)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 32, 8)
    ax2.invert_yaxis()
    lns4 = ax2.bar(np.arange(len(x)), z, width = 0.8, color='royalblue', label='Precip')
    ax2.tick_params(axis='y', labelsize=10)

# add all legends together
handles = lns1+lns2+lns3+[lns4]
labels = [l.get_label() for l in handles]

# handles, labels = ax.get_legend_handles_labels()
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 2.3))
fig.text(0.52, 0.01, 'Days', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.4, 'AMSR2 SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.95, 0.35, 'GPM Precipitation (mm/day)', rotation='vertical', fontsize=16, fontweight='bold')
plt.suptitle('USA', fontsize=19, y=0.97, fontweight='bold')
plt.savefig(path_results + '/scatterplots/' + 'miss_1_tseries' + '.png')
plt.close(fig)
