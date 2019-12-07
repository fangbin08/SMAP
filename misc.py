import os
import numpy as np
import matplotlib.pyplot as plt
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





import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.axis([1, 10000, 1, 100000])
ax.loglog()

plt.show()

x = np.arange(10)

for a in [1.0, 2.0, 3.0]:
    plt.plot(x, a*x)

plt.show()

# Sample data
xx = np.arange(10)
yy = -5 * xx + 10

# Fit with polyfit
b, m = np.polyfit(xx, yy, 1)

plt.scatter(xx, yy)


password:
Nova=0411
Nova_0411
Nova-0411
Nuevo-0411
Nuevo_0411
Nuevo=0411
Colu_0411
Bin-0411
BinF0411

joyousviolin875
192.168.0.17

档案号: 201712130229084

Dept: 13040 Fund: FC01 Object: 52022
account: 161341, GP10216, 31350 EN-Eng Sys and Environment

DL Number: F04180930010822
Passport: EB0598660

BOA: 223006133635
Routing No.: 053904483

Visa Card: 4147 2023 2709 6523, 4147202327096523
CVV: 999

Phone account number: 534017067098
pin number: 8340

H1B receipt number (UVA): WAC1908350447
EB1A receipt number: SRC1890397889
EB2-NIW receipt number: SRC1890397888

VIN: JM3KE2CY4G0619273

Convolution Neural Network
Maxwell-Boltzmann Distribution
Support Vector Regression


8809 5289 98
 

UVA computing ID: BF3FH
Email: bf3fh@virginia.edu
password: Sc_nova_0411
UVA ID: 596996604

passcode: tswbiv5

4349965047





