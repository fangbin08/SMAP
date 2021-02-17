import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import calendar
import datetime
import glob
import rasterio
import gdal
import pandas as pd
from scipy import stats
import itertools
plt.rcParams["font.family"] = "serif"
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data'
# Path of downscaled SM
path_smap_sm_ds = '/Users/binfang/Downloads/Processing/Downscale'
# Path of in-situ data
path_insitu = '/Users/binfang/Documents/SMAP_Project/data/insitu_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_201204'
# Path of 9 km SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'
# Path of validation data
path_validation = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of ISMN data
path_ismn = '/Volumes/MyPassport/SMAP_Project/Datasets/ISMN'
# Path of GPM
path_gpm = '/Volumes/MyPassport/SMAP_Project/Datasets/GPM'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am_slice', 'smap_sm_9km_pm_slice']
region_name = ['Oceania', 'Europe', 'SouthAmerica', 'Africa', 'Asia', 'NorthAmerica']
smap_overpass = ['06:00', '18:00']
yearname = np.linspace(2015, 2020, 6, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2020-12-31'
year = 2020 - 2015 + 1

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
yearname = np.linspace(2015, 2020, 6, dtype='int')
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

daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=0)
ind_init = daysofmonth_seq_cumsum[2, :]
ind_end = daysofmonth_seq_cumsum[8, :]
ind_gpm = np.stack((ind_init, ind_end), axis=1)

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_geo_10km', 'lon_world_geo_10km',]

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


####################################################################################################################################
# 1.1 Extract SM in situ data from ISMN .stm files

columns = ['lat', 'lon'] + date_seq_doy
folder_region = sorted(glob.glob(path_ismn + '/Ver_2/*/'))

for ire in range(len(folder_region)): # Region (Continent) folders
    folder_network = sorted([name for name in os.listdir(folder_region[ire]) if os.path.isdir(os.path.join(folder_region[ire], name))])

    for inw in range(len(folder_network)): # Network folders
        folder_site = sorted([name for name in os.listdir(folder_region[ire]+folder_network[inw])
                       if os.path.isdir(os.path.join(folder_region[ire]+folder_network[inw], name))])

        stn_name_all = []
        stn_lat_all = []
        stn_lon_all = []
        sm_array_am_all = []
        sm_array_pm_all = []
        for ist in range(len(folder_site)): # Site folders
            sm_file_path = folder_region[ire] + folder_network[inw] + '/' + folder_site[ist]
            sm_file_list = sorted(glob.glob(sm_file_path + '/*_sm_*'))
            if len(sm_file_list) != 0:
                sm_file = sm_file_list[0]
                net_name, stn_name, stn_lat, stn_lon, sm_array = insitu_extraction(sm_file)
                sm_array_am = sm_array[0, :]
                sm_array_pm = sm_array[1, :]

                stn_name_all.append(stn_name)
                stn_lat_all.append(stn_lat)
                stn_lon_all.append(stn_lon)
                sm_array_am_all.append(sm_array_am)
                sm_array_pm_all.append(sm_array_pm)
                print(sm_file)
            else:
                pass

        sm_mat_am = np.concatenate(
            (np.expand_dims(np.array(stn_lat_all), axis=1),
             np.expand_dims(np.array(stn_lon_all), axis=1),
             np.stack(sm_array_am_all)),
            axis=1)
        sm_mat_pm = np.concatenate(
            (np.expand_dims(np.array(stn_lat_all), axis=1),
             np.expand_dims(np.array(stn_lon_all), axis=1),
             np.stack(sm_array_pm_all)),
            axis=1)

        df_sm_am = pd.DataFrame(sm_mat_am, columns=columns, index=stn_name_all)
        df_sm_pm = pd.DataFrame(sm_mat_pm, columns=columns, index=stn_name_all)
        writer = pd.ExcelWriter(path_ismn + '/processed_data/' + region_name[ire] + '_' + folder_network[inw] + '_' + 'ismn_sm.xlsx')
        df_sm_am.to_excel(writer, sheet_name='AM')
        df_sm_pm.to_excel(writer, sheet_name='PM')
        writer.save()

        del(sm_mat_am, sm_mat_pm, df_sm_am, df_sm_pm, writer)



# 1.2 Extract Land cover types and main soil types
# folder_region = sorted(glob.glob('/Volumes/MyPassport/SMAP_Project/Datasets/ISMN/*/'))

for ire in range(len(folder_region)): # Region folders
    folder_network = sorted([name for name in os.listdir(folder_region[ire]) if os.path.isdir(os.path.join(folder_region[ire], name))])

    for inw in range(len(folder_network)): # Network folders
        folder_site = sorted([name for name in os.listdir(folder_region[ire]+folder_network[inw])
                       if os.path.isdir(os.path.join(folder_region[ire]+folder_network[inw], name))])

        stn_name_all = []
        landcover_all = []
        soiltype_all = []
        for ist in range(len(folder_site)): # Site folders
            csv_file_path = folder_region[ire] + folder_network[inw] + '/' + folder_site[ist]
            csv_file_list = glob.glob(csv_file_path + '/*.csv')
            # stn_name = folder_site[ist]
            sm_file_list = glob.glob(csv_file_path + '/*_sm_*')

            if len(csv_file_list) != 0 and len(sm_file_list) != 0:
                csv_file = csv_file_list[0]
                df_file = pd.read_csv(csv_file, index_col=0, sep=';')
                if len(df_file.index) >= 14:
                    sm_file = sm_file_list[0]
                    net_name, stn_name, stn_lat, stn_lon, sm_array = insitu_extraction(sm_file)
                    landcover = df_file.loc['land cover classification']['description'][0]
                    soiltype_array = df_file.loc[['clay fraction', 'sand fraction', 'silt fraction']]['value']
                    soiltype_array = [float(soiltype_array[x]) for x in range(len(soiltype_array))]
                    soiltype_ratio = np.array([soiltype_array[x*2]+soiltype_array[x*2+1] for x in range(2)])
                    soiltype = ['clay fraction', 'sand fraction', 'silt fraction'][np.where(soiltype_ratio == np.max(soiltype_ratio))[0][0].item()]

                    landcover_all.append(landcover)
                    soiltype_all.append(soiltype)
                    # stn_name_all.append(stn_name)
                    print(csv_file_list[0])

                else:
                    landcover_all.append('')
                    soiltype_all.append('')

                stn_name_all.append(stn_name)

            else:
                pass

        df_landcover = pd.DataFrame({'land cover': landcover_all, 'soiltype': soiltype_all}, index=stn_name_all)
        writer = pd.ExcelWriter(path_ismn + '/landcover/' + region_name[ire] + '_' + folder_network[inw] + '_' + 'landcover.xlsx')
        df_landcover.to_excel(writer)
        writer.save()

        del(df_landcover, writer)



####################################################################################################################################
# 2.1 Load the site lat/lon from Excel files and Locate the 1/9 km SM positions by lat/lon of ISMN in-situ data

ismn_list = sorted(glob.glob(path_ismn + '/processed_data/[A-Z]*.xlsx'))

coords_all = []
df_table_am_all = []
df_table_pm_all = []
for ife in range(len(ismn_list)):
    df_table_am = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='AM')
    df_table_pm = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='PM')

    netname = os.path.basename(ismn_list[ife]).split('_')[1]
    netname = [netname] * df_table_am.shape[0]
    coords = df_table_am[['lat', 'lon']]
    coords_all.append(coords)

    df_table_am_value = df_table_am.iloc[:, 2:]
    df_table_am_value.insert(0, 'network', netname)
    df_table_pm_value = df_table_pm.iloc[:, 2:]
    df_table_pm_value.insert(0, 'network', netname)
    df_table_am_all.append(df_table_am_value)
    df_table_pm_all.append(df_table_pm_value)
    del(df_table_am, df_table_pm, df_table_am_value, df_table_pm_value, coords, netname)
    print(ife)

df_coords = pd.concat(coords_all)
df_table_am_all = pd.concat(df_table_am_all)
df_table_pm_all = pd.concat(df_table_pm_all)

new_index = [df_coords.index[x] for x in range(len(df_coords.index))] # Capitalize each word
# new_index = [df_coords.index[x].title() for x in range(len(df_coords.index))]
df_coords.index = new_index
df_table_am_all.index = new_index
df_table_pm_all.index = new_index


# 2.2 Locate the SMAP 1/9 km SM positions by lat/lon of in-situ data
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
stn_row_9km_ind_all = []
stn_col_9km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    stn_row_9km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_9km)).item()
    stn_col_9km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_9km)).item()
    stn_row_9km_ind_all.append(stn_row_9km_ind)
    stn_col_9km_ind_all.append(stn_col_9km_ind)
    del(stn_row_1km_ind, stn_col_1km_ind, stn_row_9km_ind, stn_col_9km_ind)


########################################################################################################################
# 3. Extract the SMAP 1/9 km SM by the indexing files

# 3.1 Extract 1km SMAP SM
smap_1km_sta_all = []
tif_files_1km_name_ind_all = []
for iyr in range(len(yearname)):

    os.chdir(path_smap +'/1km' + '/gldas/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Extract the file name
    tif_files_name = [os.path.splitext(tif_files[x])[0].split('_')[-1] for x in range(len(tif_files))]
    tif_files_name_1year_ind = [date_seq_doy.index(item) for item in tif_files_name if item in date_seq_doy]
    date_seq_doy_1year_ind = [tif_files_name.index(item) for item in tif_files_name if item in date_seq_doy]

    tif_files_1km_name_ind_all.append(tif_files_name_1year_ind)
    del(tif_files_name, tif_files_name_1year_ind)

    smap_1km_sta_1year = []
    for idt in range(len(date_seq_doy_1year_ind)):
        src_tf = rasterio.open(tif_files[date_seq_doy_1year_ind[idt]]).read()
        smap_1km_sta_1day = src_tf[:, stn_row_1km_ind_all, stn_col_1km_ind_all]
        smap_1km_sta_1year.append(smap_1km_sta_1day)
        del(src_tf, smap_1km_sta_1day)
        print(tif_files[date_seq_doy_1year_ind[idt]])

    smap_1km_sta_all.append(smap_1km_sta_1year)
    del(smap_1km_sta_1year, date_seq_doy_1year_ind)


tif_files_1km_name_ind_all = np.concatenate(tif_files_1km_name_ind_all)
smap_1km_sta_all = np.concatenate(smap_1km_sta_all)


# Fill the extracted SMAP SM into the proper position of days
smap_1km_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-1), dtype='float32')
smap_1km_sta_am[:] = np.nan
for idt in range(len(tif_files_1km_name_ind_all)):
    smap_1km_sta_am[:, tif_files_1km_name_ind_all[idt]] = smap_1km_sta_all[idt, 0, :]

smap_1km_sta_pm = np.empty((df_table_pm_all.shape[0], df_table_pm_all.shape[1]-1), dtype='float32')
smap_1km_sta_pm[:] = np.nan
for idt in range(len(tif_files_1km_name_ind_all)):
    smap_1km_sta_pm[:, tif_files_1km_name_ind_all[idt]] = smap_1km_sta_all[idt, 1, :]

index_validation = df_table_am_all.index
columns_validation = df_table_am_all.columns
network_validation = df_table_am_all[['network']]

smap_1km_sta_am = pd.DataFrame(smap_1km_sta_am, columns=date_seq_doy, index=index_validation)
df_smap_1km_sta_am = pd.concat([network_validation, smap_1km_sta_am], axis=1, sort=False)
smap_1km_sta_pm = pd.DataFrame(smap_1km_sta_pm, columns=date_seq_doy, index=index_validation)
df_smap_1km_sta_pm = pd.concat([network_validation, smap_1km_sta_pm], axis=1, sort=False)


# 3.2 Extract 9km SMAP SM
smap_9km_sta_am = []
smap_9km_sta_pm = []
for iyr in range(len(yearname)):

    smap_9km_sta_am_1year = []
    smap_9km_sta_pm_1year = []
    for imo in range(len(monthname)):

        smap_9km_sta_am_1month = []
        smap_9km_sta_pm_1month = []
        # Load in SMAP 9km SM data
        smap_file_path = path_smap + '/9km/' + 'smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

        # Find the if the file exists in the directory
        if os.path.exists(smap_file_path) == True:

            f_smap_9km = h5py.File(smap_file_path, "r")
            varname_list_smap = list(f_smap_9km.keys())
            smap_9km_sta_am_1month = f_smap_9km[varname_list_smap[0]][()]
            smap_9km_sta_am_1month = smap_9km_sta_am_1month[stn_row_9km_ind_all, stn_col_9km_ind_all, :]
            smap_9km_sta_pm_1month = f_smap_9km[varname_list_smap[1]][()]
            smap_9km_sta_pm_1month = smap_9km_sta_pm_1month[stn_row_9km_ind_all, stn_col_9km_ind_all, :]

            print(smap_file_path)
            f_smap_9km.close()

        else:
            pass

        smap_9km_sta_am_1year.append(smap_9km_sta_am_1month)
        smap_9km_sta_pm_1year.append(smap_9km_sta_pm_1month)
        del(smap_9km_sta_am_1month, smap_9km_sta_pm_1month)

    smap_9km_sta_am.append(smap_9km_sta_am_1year)
    smap_9km_sta_pm.append(smap_9km_sta_pm_1year)
    del(smap_9km_sta_am_1year, smap_9km_sta_pm_1year)

# Remove the empty lists
smap_9km_sta_am[0] = smap_9km_sta_am[0][3:]
smap_9km_sta_pm[0] = smap_9km_sta_pm[0][3:]

smap_9km_sta_am = list(itertools.chain(*smap_9km_sta_am))
smap_9km_sta_am = np.concatenate(smap_9km_sta_am, axis=1)
smap_9km_sta_pm = list(itertools.chain(*smap_9km_sta_pm))
smap_9km_sta_pm = np.concatenate(smap_9km_sta_pm, axis=1)

smap_9km_sta_am = pd.DataFrame(smap_9km_sta_am, columns=date_seq_doy, index=index_validation)
df_smap_9km_sta_am = pd.concat([network_validation, smap_9km_sta_am], axis=1, sort=False)
smap_9km_sta_pm = pd.DataFrame(smap_9km_sta_pm, columns=date_seq_doy, index=index_validation)
df_smap_9km_sta_pm = pd.concat([network_validation, smap_9km_sta_pm], axis=1, sort=False)


# Save variables
writer_insitu = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_insitu.xlsx')
writer_1km = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_1km.xlsx')
writer_9km = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_9km.xlsx')
writer_coords = pd.ExcelWriter(path_ismn + '/extraction/smap_coords.xlsx')
df_table_am_all.to_excel(writer_insitu, sheet_name='AM')
df_table_pm_all.to_excel(writer_insitu, sheet_name='PM')
df_smap_1km_sta_am.to_excel(writer_1km, sheet_name='AM')
df_smap_1km_sta_pm.to_excel(writer_1km, sheet_name='PM')
df_smap_9km_sta_am.to_excel(writer_9km, sheet_name='AM')
df_smap_9km_sta_pm.to_excel(writer_9km, sheet_name='PM')
df_coords.to_excel(writer_coords)
writer_insitu.save()
writer_1km.save()
writer_9km.save()
writer_coords.save()


# 3.3 Extract the GPM data by indices
df_coords = pd.read_excel(path_ismn + '/extraction/smap_coords.xlsx', index_col=0)
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

# Locate the corresponding GPM 10 km data located by lat/lon of in-situ data
stn_row_10km_ind_all = []
stn_col_10km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_10km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_geo_10km)).item()
    stn_col_10km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_geo_10km)).item()
    stn_row_10km_ind_all.append(stn_row_10km_ind)
    stn_col_10km_ind_all.append(stn_col_10km_ind)
    del(stn_row_10km_ind, stn_col_10km_ind)

gpm_precip_ext_all = []
for iyr in range(len(yearname)):

    f_gpm = h5py.File(path_gpm + '/gpm_precip_' + str(yearname[iyr]) + '.hdf5', 'r')
    varname_list_gpm = list(f_gpm.keys())

    for x in range(len(varname_list_gpm)):
        var_obj = f_gpm[varname_list_gpm[x]][()]
        exec(varname_list_gpm[x] + '= var_obj')
        del(var_obj)
    f_gpm.close()

    exec('gpm_precip = gpm_precip_10km_' + str(yearname[iyr]))
    gpm_precip_ext = gpm_precip[stn_row_10km_ind_all, stn_col_10km_ind_all, :]
    gpm_precip_ext_all.append(gpm_precip_ext)
    print(iyr)
    del(gpm_precip, gpm_precip_ext)

gpm_precip_ext_array = np.concatenate(gpm_precip_ext_all, axis=1)

# Save variables
df_table_gpm = pd.DataFrame(gpm_precip_ext_array, columns=columns_validation[1:], index=index_validation)
df_table_gpm = pd.concat([network_validation, df_table_gpm], axis=1)
writer_gpm = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_gpm.xlsx')
df_table_gpm.to_excel(writer_gpm)
writer_gpm.save()


########################################################################################################################
# 4. Plot validation results between 1 km, 9 km and in-situ data

# Load in 1 km / 9 km matched data for validation
# os.chdir(path_model)
# f = h5py.File("smap_sm_match.hdf5", "r")
# varname_list = list(f.keys())
#
# for x in range(len(varname_list)):
#     var_obj = f[varname_list[x]][()]
#     exec(varname_list[x] + '= var_obj')
#     del(var_obj)
# f.close()

df_smap_insitu_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_insitu.xlsx', index_col=0, sheet_name='AM')
df_smap_1km_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_1km.xlsx', index_col=0, sheet_name='AM')
df_smap_9km_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_9km.xlsx', index_col=0, sheet_name='AM')
df_smap_gpm = pd.read_excel(path_ismn + '/extraction/smap_validation_gpm.xlsx', index_col=0)
size_validation = np.shape(df_smap_insitu_sta_am)
stn_name = df_smap_insitu_sta_am.index
network_name = df_smap_insitu_sta_am['network'].tolist()
network_unique = df_smap_insitu_sta_am['network'].unique()

# Create folders for each network
for ife in range(len(network_unique)):
    os.mkdir(path_results + '/validation/single_plots/' + network_unique[ife])

# 4.1 single plots
stat_array_1km = []
stat_array_9km = []
ind_slc_all = []
for ist in range(size_validation[0]):
    x = np.array(df_smap_insitu_sta_am.iloc[ist, 1:], dtype=np.float)
    y1 = np.array(df_smap_1km_sta_am.iloc[ist, 1:], dtype=np.float)
    y2 = np.array(df_smap_9km_sta_am.iloc[ist, 1:], dtype=np.float)
    x[x == 0] = np.nan
    y1[y1 == 0] = np.nan
    y2[y2 == 0] = np.nan

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

        fig = plt.figure(figsize=(8, 5), dpi=200)
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        ax = fig.add_subplot(111)

        ax.scatter(x, y1, s=5, c='r', marker='o', label='1 km')
        ax.scatter(x, y2, s=5, c='k', marker='^', label='9 km')

        ax.plot(x, intercept_1 + slope_1 * x, '-', color='r')
        ax.plot(x, intercept_2 + slope_2 * x, '-', color='k')

        # coef1, intr1 = np.polyfit(x.squeeze(), y1.squeeze(), 1)
        # ax.plot(x.squeeze(), intr1 + coef1 * x.squeeze(), '-', color='r')
        #
        # coef2, intr2 = np.polyfit(x.squeeze(), y2.squeeze(), 1)
        # ax.plot(x.squeeze(), intr2 + coef2 * x.squeeze(), '-', color='k')

        plt.xlim(0, 0.4)
        ax.set_xticks(np.arange(0, 0.5, 0.1))
        plt.ylim(0, 0.4)
        ax.set_yticks(np.arange(0, 0.5, 0.1))
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        plt.grid(linestyle='--')
        plt.legend(loc='upper left')
        plt.suptitle(network_name[ist] +', ' + str(stn_name[ist]), fontsize=18, y=0.99, fontweight='bold')
        # plt.show()
        # plt.savefig(path_results +'/validation/single_plots/' + network_name[ist] + '/' + str(stn_name[ist]) + '.png')
        plt.close(fig)
        stat_array_1km.append(stat_array_1)
        stat_array_9km.append(stat_array_2)
        ind_slc_all.append(ist)
        print(ist)
        del(stat_array_1, stat_array_2)
    else:
        pass

stat_array_1km = np.array(stat_array_1km)
stat_array_9km = np.array(stat_array_9km)

columns_validation = ['number', 'r_sq', 'ubrmse', 'stdev', 'bias', 'p_value', 'conf_int']
index_validation = df_smap_1km_sta_am.index[ind_slc_all]
network_validation = df_smap_1km_sta_am['network'].iloc[ind_slc_all]

df_stat_1km = pd.DataFrame(stat_array_1km, columns=columns_validation, index=index_validation)
df_stat_9km = pd.DataFrame(stat_array_9km, columns=columns_validation, index=index_validation)
df_stat_1km = pd.concat([network_validation, df_stat_1km], axis=1, sort=False)
df_stat_9km = pd.concat([network_validation, df_stat_9km], axis=1, sort=False)
writer_1km = pd.ExcelWriter(path_results + '/validation/stat_1km.xlsx')
writer_9km = pd.ExcelWriter(path_results + '/validation/stat_9km.xlsx')
df_stat_1km.to_excel(writer_1km)
df_stat_9km.to_excel(writer_9km)
writer_1km.save()
writer_9km.save()


# df_stat_1km = pd.read_excel('/Users/binfang/Documents/SMAP_Project/results/results_201204/validation/stat_1km.xlsx',
#                                    index_col=0)
# df_stat_9km = pd.read_excel('/Users/binfang/Documents/SMAP_Project/results/results_201204/validation/stat_9km.xlsx',
#                                    index_col=0)


# 4.2 subplots
# REMEDHUS: Carretoro, Casa_Periles, El_Tomillar, Granja_g, La_Atalaya, Las_Bodegas, Las_Vacas, Zamarron
# REMEDHUS ID:382, 383, 386, 387, 389, 392, 396, 400
# SOILSCAPE: node403, 404, 405, 406, 408, 415, 416, 417
# SOILSCAPE ID: 1425, 1426, 1427, 1428, 1429, 1436, 1437, 1438
# CTP: L18, L19, L21, L27, L33, L34, L36, L37
# CTP ID: 38, 39, 41, 46, 51, 52, 54, 55
# OZNET: Alabama, Banandra, Bundure, Samarra, Uri Park, Wollumbi, Yamma Road, Yammacoona
# OZNET ID: 1619, 1620, 1621, 1629, 1632, 1634, 1636, 1637

network_name = ['REMEDHUS', 'SOILSCAPE', 'CTP', 'OZNET']
site_ind = [[382, 383, 386, 387, 389, 392, 396, 400], [1425, 1426, 1427, 1428, 1429, 1436, 1437, 1438],
            [38, 39, 41, 46, 51, 52, 54, 55], [1619, 1620, 1621, 1629, 1632, 1634, 1636, 1637]]
# network_name = list(stn_slc_all_unique)
# site_ind = stn_slc_all_group

for inw in range(len(site_ind)):
    fig = plt.figure(figsize=(11, 11))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.92, hspace=0.25, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = np.array(df_smap_insitu_sta_am.iloc[site_ind[inw][ist], 1:], dtype=np.float64)
        x[x == 0] = np.nan
        y1 = np.array(df_smap_1km_sta_am.iloc[site_ind[inw][ist], 1:], dtype=np.float64)
        y2 = np.array(df_smap_9km_sta_am.iloc[site_ind[inw][ist], 1:], dtype=np.float64)
        ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]

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

        ax = fig.add_subplot(len(site_ind[inw])//2, 2, ist+1)
        sc1 = ax.scatter(x, y1, s=10, c='r', marker='o', label='1 km')
        sc2 = ax.scatter(x, y2, s=10, c='k', marker='^', label='9 km')
        ax.plot(x, intercept_1+slope_1*x, '-', color='r')
        ax.plot(x, intercept_2+slope_2*x, '-', color='k')

        plt.xlim(0, 0.4)
        ax.set_xticks(np.arange(0, 0.5, 0.1))
        plt.ylim(0, 0.4)
        ax.set_yticks(np.arange(0, 0.5, 0.1))
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        plt.grid(linestyle='--')
        ax.text(0.01, 0.35, df_smap_1km_sta_am.index[site_ind[inw][ist]].replace('_', ' '), fontsize=13)

    # add all legends together
    handles = [sc1] + [sc2]
    labels = [l.get_label() for l in handles]
    # leg = plt.legend([sc1, sc2, sc3], labels, loc=(-0.6, 3.55), mode="expand", borderaxespad=0, ncol=3, prop={"size": 13})
    leg = plt.legend([sc1, sc2], labels, loc=(-0.7, 4.8), mode="expand", borderaxespad=0, ncol=3, prop={"size": 13})

    fig.text(0.52, 0.01, 'In Situ SM ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.4, 'SMAP SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=21, y=0.99, fontweight='bold')
    plt.show()

    plt.savefig(path_results + '/validation/network_plots/' + network_name[inw] + '.png')
    plt.close(fig)



# 4.3 Make the time-series plots
# Generate index tables for calculating monthly averages
monthly_seq = np.reshape(daysofmonth_seq, (1, -1), order='F')
monthly_seq = monthly_seq[:, 3:] # Remove the first 3 months in 2015
monthly_seq_cumsum = np.cumsum(monthly_seq)
array_allnan = np.empty([size_validation[0], 3], dtype='float32')
array_allnan[:] = np.nan

smap_insitu_am_split = \
    np.hsplit(np.array(df_smap_insitu_sta_am.iloc[:, 1:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_insitu_am_monthly = [np.nanmean(smap_insitu_am_split[x], axis=1) for x in range(len(smap_insitu_am_split))]
smap_insitu_am_monthly = np.stack(smap_insitu_am_monthly, axis=0)
smap_insitu_am_monthly = np.transpose(smap_insitu_am_monthly, (1, 0))
smap_insitu_am_monthly = smap_insitu_am_monthly[:, :-1]
smap_insitu_am_monthly = np.concatenate([array_allnan, smap_insitu_am_monthly], axis=1)

smap_1km_am_split = \
    np.hsplit(np.array(df_smap_1km_sta_am.iloc[:, 1:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_1km_am_monthly = [np.nanmean(smap_1km_am_split[x], axis=1) for x in range(len(smap_1km_am_split))]
smap_1km_am_monthly = np.stack(smap_1km_am_monthly, axis=0)
smap_1km_am_monthly = np.transpose(smap_1km_am_monthly, (1, 0))
smap_1km_am_monthly = smap_1km_am_monthly[:, :-1]
smap_1km_am_monthly = np.concatenate([array_allnan, smap_1km_am_monthly], axis=1)

smap_9km_am_split = \
    np.hsplit(np.array(df_smap_9km_sta_am.iloc[:, 1:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_9km_am_monthly = [np.nanmean(smap_9km_am_split[x], axis=1) for x in range(len(smap_9km_am_split))]
smap_9km_am_monthly = np.stack(smap_9km_am_monthly, axis=0)
smap_9km_am_monthly = np.transpose(smap_9km_am_monthly, (1, 0))
smap_9km_am_monthly = smap_9km_am_monthly[:, :-1]
smap_9km_am_monthly = np.concatenate([array_allnan, smap_9km_am_monthly], axis=1)

smap_gpm_split = \
    np.hsplit(np.array(df_smap_gpm.iloc[:, 1:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_gpm_monthly = [np.nansum(smap_gpm_split[x], axis=1) for x in range(len(smap_gpm_split))]
smap_gpm_monthly = np.stack(smap_gpm_monthly, axis=0)
smap_gpm_monthly = np.transpose(smap_gpm_monthly, (1, 0))
smap_gpm_monthly = smap_gpm_monthly[:, :-1]
smap_gpm_monthly = np.concatenate([array_allnan, smap_gpm_monthly], axis=1)

# Make the time-series plots
network_name = ['REMEDHUS', 'SOILSCAPE', 'CTP', 'OZNET']
# site_ind = [[382, 383, 386, 387, 389, 392, 396, 400], [1425, 1426, 1427, 1428, 1429, 1436, 1437, 1438],
#             [38, 39, 41, 46, 51, 52, 54, 55], [1619, 1620, 1621, 1629, 1632, 1634, 1636, 1637]]
site_ind = [[382, 383, 387, 396], [1425, 1427, 1429, 1438],
            [38, 39, 52, 55], [1632, 1634, 1636, 1637]]

# Network 1
for inw in [0]:#range(len(site_ind)):

    fig = plt.figure(figsize=(10, 8), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.08, top=0.9, hspace=0.3, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = smap_insitu_am_monthly[site_ind[inw][ist], :]
        y1 = smap_1km_am_monthly[site_ind[inw][ist], :]
        y2 = smap_9km_am_monthly[site_ind[inw][ist], :]
        z = smap_gpm_monthly[site_ind[inw][ist], :]
        # ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        # x = x[ind_nonnan]
        # y1 = y1[ind_nonnan]
        # y2 = y2[ind_nonnan]
        # z = z[ind_nonnan]

        ax = fig.add_subplot(4, 1, ist+1)
        lns1 = ax.plot(x, c='b', marker='s', label='In-situ', markersize=3, linestyle='--', linewidth=1)
        lns2 = ax.plot(y1, c='r', marker='o', label='1 km', markersize=3, linestyle='--', linewidth=1)
        lns3 = ax.plot(y2, c='k', marker='^', label='9 km', markersize=3, linestyle='--', linewidth=1)
        ax.text(0, 0.35, df_smap_1km_sta_am.index[site_ind[inw][ist]].replace('_', ' '), fontsize=11, fontweight='bold')

        plt.xlim(0, len(x))
        ax.set_xticks(np.arange(0, len(x)+12, len(x)//6))
        ax.set_xticklabels([])
        labels = ['2015', '2016', '2017', '2018', '2019', '2020']
        mticks = ax.get_xticks()
        ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(labels, minor=True)

        plt.ylim(0, 0.4)
        ax.set_yticks(np.arange(0, 0.5, 0.1))
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(linestyle='--')

        ax2 = ax.twinx()
        ax2.set_ylim(0, 120)
        ax2.invert_yaxis()
        ax2.set_yticks(np.arange(0, 150, 30))
        lns4 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
        ax2.tick_params(axis='y', labelsize=10)

    # add all legends together
    handles = lns1+lns2+lns3+[lns4]
    labels = [l.get_label() for l in handles]

    plt.legend(handles, labels, loc=(0, 4.95), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
    fig.text(0.5, 0.02, 'Years', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.4, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.96, 0.4, 'GPM Precipitation (mm)', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=19, y=0.98, fontweight='bold')
    plt.savefig(path_results + '/validation/network_plots/' + network_name[inw] + '_tseries' + '.png')
    plt.close(fig)


# Network 2
for inw in [1]:#range(len(site_ind)):

    fig = plt.figure(figsize=(10, 8), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.08, top=0.9, hspace=0.3, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = smap_insitu_am_monthly[site_ind[inw][ist], :]
        y1 = smap_1km_am_monthly[site_ind[inw][ist], :]
        y2 = smap_9km_am_monthly[site_ind[inw][ist], :]
        z = smap_gpm_monthly[site_ind[inw][ist], :]
        # ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        # x = x[ind_nonnan]
        # y1 = y1[ind_nonnan]
        # y2 = y2[ind_nonnan]
        # z = z[ind_nonnan]

        ax = fig.add_subplot(4, 1, ist+1)
        lns1 = ax.plot(x, c='b', marker='s', label='In-situ', markersize=3, linestyle='--', linewidth=1)
        lns2 = ax.plot(y1, c='r', marker='o', label='1 km', markersize=3, linestyle='--', linewidth=1)
        lns3 = ax.plot(y2, c='k', marker='^', label='9 km', markersize=3, linestyle='--', linewidth=1)
        ax.text(0, 0.35, df_smap_1km_sta_am.index[site_ind[inw][ist]].replace('_', ' '), fontsize=11, fontweight='bold')

        plt.xlim(0, len(x))
        ax.set_xticks(np.arange(0, len(x)+12, len(x)//6))
        ax.set_xticklabels([])
        labels = ['2015', '2016', '2017', '2018', '2019', '2020']
        mticks = ax.get_xticks()
        ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(labels, minor=True)

        plt.ylim(0, 0.4)
        ax.set_yticks(np.arange(0, 0.5, 0.1))
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(linestyle='--')

        ax2 = ax.twinx()
        ax2.set_ylim(0, 160)
        ax2.invert_yaxis()
        ax2.set_yticks(np.arange(0, 200, 40))
        lns4 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
        ax2.tick_params(axis='y', labelsize=10)

    # add all legends together
    handles = lns1+lns2+lns3+[lns4]
    labels = [l.get_label() for l in handles]

    plt.legend(handles, labels, loc=(0, 4.95), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
    fig.text(0.5, 0.02, 'Years', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.4, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.96, 0.4, 'GPM Precipitation (mm)', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=19, y=0.98, fontweight='bold')
    plt.savefig(path_results + '/validation/network_plots/' + network_name[inw] + '_tseries' + '.png')
    plt.close(fig)


# Network 3
for inw in [2]:#range(len(site_ind)):

    fig = plt.figure(figsize=(10, 8), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.08, top=0.9, hspace=0.3, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = smap_insitu_am_monthly[site_ind[inw][ist], :]
        y1 = smap_1km_am_monthly[site_ind[inw][ist], :]
        y2 = smap_9km_am_monthly[site_ind[inw][ist], :]
        z = smap_gpm_monthly[site_ind[inw][ist], :]
        # ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        # x = x[ind_nonnan]
        # y1 = y1[ind_nonnan]
        # y2 = y2[ind_nonnan]
        # z = z[ind_nonnan]

        ax = fig.add_subplot(4, 1, ist+1)
        lns1 = ax.plot(x, c='b', marker='s', label='In-situ', markersize=3, linestyle='--', linewidth=1)
        lns2 = ax.plot(y1, c='r', marker='o', label='1 km', markersize=3, linestyle='--', linewidth=1)
        lns3 = ax.plot(y2, c='k', marker='^', label='9 km', markersize=3, linestyle='--', linewidth=1)
        ax.text(0, 0.45, df_smap_1km_sta_am.index[site_ind[inw][ist]].replace('_', ' '), fontsize=11, fontweight='bold')

        plt.xlim(0, len(x))
        ax.set_xticks(np.arange(0, len(x)+12, len(x)//6))
        ax.set_xticklabels([])
        labels = ['2015', '2016', '2017', '2018', '2019', '2020']
        mticks = ax.get_xticks()
        ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(labels, minor=True)

        plt.ylim(0, 0.5)
        ax.set_yticks(np.arange(0, 0.6, 0.1))
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(linestyle='--')

        ax2 = ax.twinx()
        ax2.set_ylim(0, 50)
        ax2.invert_yaxis()
        ax2.set_yticks(np.arange(0, 60, 10))
        lns4 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
        ax2.tick_params(axis='y', labelsize=10)

    # add all legends together
    handles = lns1+lns2+lns3+[lns4]
    labels = [l.get_label() for l in handles]

    plt.legend(handles, labels, loc=(0, 4.95), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
    fig.text(0.5, 0.02, 'Years', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.4, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.96, 0.4, 'GPM Precipitation (mm)', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=19, y=0.98, fontweight='bold')
    plt.savefig(path_results + '/validation/network_plots/' + network_name[inw] + '_tseries' + '.png')
    plt.close(fig)


# Network 4
for inw in [3]:#range(len(site_ind)):

    fig = plt.figure(figsize=(10, 8), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.08, top=0.9, hspace=0.3, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = smap_insitu_am_monthly[site_ind[inw][ist], :]
        y1 = smap_1km_am_monthly[site_ind[inw][ist], :]
        y2 = smap_9km_am_monthly[site_ind[inw][ist], :]
        z = smap_gpm_monthly[site_ind[inw][ist], :]
        # ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        # x = x[ind_nonnan]
        # y1 = y1[ind_nonnan]
        # y2 = y2[ind_nonnan]
        # z = z[ind_nonnan]

        ax = fig.add_subplot(4, 1, ist+1)
        lns1 = ax.plot(x, c='b', marker='s', label='In-situ', markersize=3, linestyle='--', linewidth=1)
        lns2 = ax.plot(y1, c='r', marker='o', label='1 km', markersize=3, linestyle='--', linewidth=1)
        lns3 = ax.plot(y2, c='k', marker='^', label='9 km', markersize=3, linestyle='--', linewidth=1)
        ax.text(0, 0.45, df_smap_1km_sta_am.index[site_ind[inw][ist]].replace('_', ' '), fontsize=11, fontweight='bold')

        plt.xlim(0, len(x))
        ax.set_xticks(np.arange(0, len(x)+12, len(x)//6))
        ax.set_xticklabels([])
        labels = ['2015', '2016', '2017', '2018', '2019', '2020']
        mticks = ax.get_xticks()
        ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(labels, minor=True)

        plt.ylim(0, 0.5)
        ax.set_yticks(np.arange(0, 0.6, 0.1))
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(linestyle='--')

        ax2 = ax.twinx()
        ax2.set_ylim(0, 120)
        ax2.invert_yaxis()
        ax2.set_yticks(np.arange(0, 144, 24))
        lns4 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
        ax2.tick_params(axis='y', labelsize=10)

    # add all legends together
    handles = lns1+lns2+lns3+[lns4]
    labels = [l.get_label() for l in handles]

    plt.legend(handles, labels, loc=(0, 4.95), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
    fig.text(0.5, 0.02, 'Years', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.4, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.96, 0.4, 'GPM Precipitation (mm)', rotation='vertical', fontsize=16, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=19, y=0.98, fontweight='bold')
    plt.savefig(path_results + '/validation/network_plots/' + network_name[inw] + '_tseries' + '.png')
    plt.close(fig)




