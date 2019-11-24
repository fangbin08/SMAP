# Load in variables
f = h5py.File("parameters.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()

####################################################################################################################################
# * Check the completeness of downloaded files
os.chdir(path_ltdr)
files = sorted(glob.glob('*.nc4'))
# date_seq_late = date_seq[6939:]
files_group = []
for idt in range(13879):
    # files_group_1day = [files.index(i) for i in files if 'A' + date_seq_late[idt] in i]
    files_group_1day = [files.index(i) for i in files if 'A' + date_seq[idt] in i]
    files_group.append(files_group_1day)
    print(idt)

file_miss = []
for idt in range(len(files_group)):
    if len(files_group[idt]) != 8:
        file_miss.append(date_seq[idt])
        print(date_seq[idt])
        # file_miss.append(date_seq_late[idt])
        # print(date_seq_late[idt])
        print(len(files_group[idt]))
    else:
        pass

    
# Find the indices of each month in the list of days between 1981 - 2018    
month_num = np.arange(1, 13)
month_num = [str(i).zfill(2) for i in month_num]
date_seq_mo = [date_seq[i][4:6] for i in range(len(date_seq))] # Create a sequence containing only month names

ind_month_gldas = []
for m in range(len(month_num)):
    ind_month_gldas_1mo = [i for i, s in enumerate(date_seq_mo) if month_num[m] in s]
    ind_month_gldas.append(ind_month_gldas_1mo)

# Initialize empty matrices
lst_gldas_am_delta_all = np.empty((len(row_lmask_ease_25km_ind), 0), float).astype('float32')
lst_gldas_pm_delta_all = np.empty((len(row_lmask_ease_25km_ind), 0), float).astype('float32')
sm_gldas_am_all = np.empty((len(row_lmask_ease_25km_ind), 0), float).astype('float32')
sm_gldas_pm_all = np.empty((len(row_lmask_ease_25km_ind), 0), float).astype('float32')
ltdr_ndvi_all = np.empty((len(row_lmask_ease_25km_ind), 0), float).astype('float32')
    
            os.chdir(path_modis_model + modis_folders[ifo] + subfolders[iyr])
            if modis_mat_ease.shape[2] == 2: #MODIS LST
                var_name = ['modis_lst_1km_day_' + mos_file_name[-7:], 'modis_lst_1km_night_' + mos_file_name[-7:]]
                with h5py.File('modis_lst_1km_' + mos_file_name[-7:] + '.hdf5', 'w') as f:
                    for idv in range(modis_mat_ease.shape[2]):
                        f.create_dataset(var_name[idv], data=modis_mat_ease[:, :, idv])
                f.close()
            else: # MODIS NDVI
                var_name = ['modis_ndvi_1km_' + mos_file_name[-7:]]
                with h5py.File('modis_ndvi_1km_' + mos_file_name[-7:] + '.hdf5', 'w') as f:
                    for idv in range(modis_mat_ease.shape[2]):
                        f.create_dataset(var_name[idv], data=modis_mat_ease[:, :, idv])
                f.close()
