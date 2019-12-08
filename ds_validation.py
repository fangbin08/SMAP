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
# # Ignore runtime warning
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    for itm in range(len(smap_overpass)):
        datatime_match_ind = [data_list.index(i) for i in data_list if smap_overpass[itm] in i]
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
path_smap_sm_ds = '/Users/binfang/Downloads/Processing/Downscale/2019/Downscale'
# Path of in-situ data
path_insitu = '/Users/binfang/Documents/SMAP_Project/data/insitu_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_191202'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am_slice', 'smap_sm_9km_pm_slice']
region_name = ['Danube', 'USA']
smap_overpass = ['06:00', '18:00']
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



# 2. Locate the SMAP 1/9 km SM positions by lat/lon of in-situ data

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

# 3. Extract the SMAP 1/9 km SM by the indexing files

year_plt = yearname[4]

# Load in SMAP 1 km SM

src_tf_ext_all = []
for iyr in [year_plt]:  # range(yearname):

    os.chdir(path_smap_sm_ds + '/' + str(iyr))
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

var_name = ['sm_array_1km_reg1', 'sm_array_1km_reg2']
with h5py.File('smap_sm_match.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()




# Load in SMAP 9 km SM

sm_array_9km_reg1 = np.empty((2, len(stn_row_9km_ind_all[0]), 0))
sm_array_9km_reg1[:] = np.nan
sm_array_9km_reg2 = np.empty((2, len(stn_row_9km_ind_all[1]), 0))
sm_array_9km_reg2[:] = np.nan
append_array_reg1 = np.empty((len(stn_row_9km_ind_all[0]), 2, 365-304))
append_array_reg2 = np.empty((len(stn_row_9km_ind_all[1]), 2, 365-304))

for iyr in [4]: #range(len(yearname)):

    for imo in range(len(monthname)):

        # Load in SMAP 9km SM data
        smap_file_path = path_procdata + '/smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

        # Find the if the file exists in the directory
        if os.path.exists(smap_file_path) == True:

            f_smap_9km = h5py.File(smap_file_path, "r")
            varname_list_smap = list(f_smap_9km.keys())

            sm_array_9km_reg = []
            for ire in range(len(stn_row_9km_ind_all)):

                for x in range(len(varname_list_smap)):
                    var_obj = f_smap_9km[varname_list_smap[x]][()]
                    lin_ind = np.ravel_multi_index([stn_row_9km_ind_all[ire], stn_col_9km_ind_all[ire]], (var_obj.shape[0], var_obj.shape[1]))
                    var_obj = np.reshape(var_obj, (var_obj.shape[0]*var_obj.shape[1], var_obj.shape[2])) # Convert from 3D to 2D
                    var_obj = var_obj[lin_ind, :]

                    exec(smap_sm_9km_name[x] + '= var_obj')
                    del(var_obj)

                smap_sm_9km_slice = np.stack((smap_sm_9km_am_slice, smap_sm_9km_pm_slice))
                sm_array_9km_reg.append(smap_sm_9km_slice)
                del (smap_sm_9km_am_slice, smap_sm_9km_pm_slice, smap_sm_9km_slice)

            sm_array_9km_reg1 = np.append(sm_array_9km_reg1, sm_array_9km_reg[0], axis=2)
            sm_array_9km_reg2 = np.append(sm_array_9km_reg2, sm_array_9km_reg[1], axis=2)

            print(imo)
            f_smap_9km.close()


        else:
            pass


sm_array_9km_reg1 = np.transpose(sm_array_9km_reg1, (1, 0, 2))
sm_array_9km_reg2 = np.transpose(sm_array_9km_reg2, (1, 0, 2))

sm_array_9km_reg1 = np.concatenate((sm_array_9km_reg1, append_array_reg1), axis=2)
sm_array_9km_reg2 = np.concatenate((sm_array_9km_reg2, append_array_reg2), axis=2)

os.chdir(path_workspace)
var_name = ['sm_array_9km_reg1', 'sm_array_9km_reg2']
with h5py.File('smap_sm_match.hdf5', 'a') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()




########################################################################################################################

# 4. Plot validation results between 1 km, 9 km and in-situ data

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
reg1_site = np.array([2, 3, 8, 15, 22, 23])

fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
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
    ax.text(0.2, 0.05, net_name_all[0][reg1_site[ist]] + '-' + stn_name_all[0][reg1_site[ist]], fontsize=15)

fig.text(0.52, 0.01, 'In Situ ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.4, 'SMAP ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
plt.suptitle('Danube River Basin', fontsize=23, y=0.97, fontweight='bold')

plt.savefig(path_results + '/scatterplots/' + 'danube_1' + '.png')
plt.close(fig)


# 4.2.2 Region 2 (USA)
reg2_site = np.array([11, 12, 18, 22, 30, 31])

fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
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
    ax.text(0.2, 0.05, net_name_all[1][reg2_site[ist]] + '-' + stn_name_all[1][reg2_site[ist]], fontsize=15)

fig.text(0.52, 0.01, 'In Situ ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.4, 'SMAP ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
plt.suptitle('Mississippi River Basin', fontsize=23, y=0.97, fontweight='bold')

plt.savefig(path_results + '/scatterplots/' + 'miss_1' + '.png')
plt.close(fig)





