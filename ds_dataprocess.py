import os, gdal, osr, glob
import numpy as np
import matplotlib.pyplot as plt
import datetime

# (Function 1) Extract data layers from MODIS HDF5 files and write to GeoTiff
def hdf_subdataset_extraction(hdf_files, dst_dir, subdataset_id, band_n):

    # Open the dataset
    global band_ds
    hdf_ds = gdal.Open(hdf_files, gdal.GA_ReadOnly)

    # Loop read data of specified bands from subdataset_id
    size_1dim = gdal.Open(hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly).ReadAsArray().astype(np.int16).shape
    band_array = np.empty([size_1dim[0], size_1dim[1], len(subdataset_id)], dtype=int)
    for idn in range(len(subdataset_id)):
        band_ds = gdal.Open(hdf_ds.GetSubDatasets()[subdataset_id[idn]][0], gdal.GA_ReadOnly)
        # Read into numpy array
        band_array[:, :, idn] = band_ds.ReadAsArray().astype(np.int16)

    # Build output path
    band_path = os.path.join(dst_dir, os.path.basename(os.path.splitext(hdf_files)[0]) + "-ctd" + ".tif")
    # Write raster
    out_ds = gdal.GetDriverByName('GTiff').Create(band_path, band_ds.RasterXSize, band_ds.RasterYSize, band_n, #Number of bands
                                  gdal.GDT_Int16, ['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(band_ds.GetGeoTransform())
    out_ds.SetProjection(band_ds.GetProjection())

    # Loop write each band to Geotiff file
    for idb in range(len(subdataset_id)):
        out_ds.GetRasterBand(idb+1).WriteArray(band_array[:, :, idb])
        out_ds.GetRasterBand(idb+1).SetNoDataValue(0)
    out_ds = None  #close dataset to write to disc

########################################################################################################################

# 0. Generate sequence of string between start and end dates (Year + DOY)

start_date = '2019-01-01'
end_date = '2019-12-31'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))


# 1. Extract data layers from MODIS HDF5 files and write to GeoTiff

src_dir = '/Users/binfang/Downloads/MODIS_new/test/input'
dst_dir = '/Users/binfang/Downloads/MODIS_new/test/output'
subdataset_id = [0, 1, 4, 5] # ith of Layers to be extracted. For MODIS LST data: LST_Day_1km, QC_Day, LST_Night_1km, QC_Night
                             # The number of layers can be acquired by GetSubDatasets()
band_n = 4
# Boundary coordinates (CONUS)
lat_roi_max = 53
lat_roi_min = 25
lon_roi_max = -67
lon_roi_min = -125

os.chdir(src_dir)
hdf_files = sorted(glob.glob('*.hdf'))

for idt in range(len(hdf_files)):
    hdf_subdataset_extraction(hdf_files[idt], dst_dir, subdataset_id, band_n)
    print(hdf_files[idt]) # Print the file being processed


# 2. Group the Geotiff files by dates from their names and
#    build virtual dataset VRT files for mosaicking MODIS geotiff files in the list

os.chdir(dst_dir)
tif_files = sorted(glob.glob('*.tif'))
tif_files_group = []
for idt in range(len(date_seq)):
    tif_files_group_1day = [tif_files.index(i) for i in tif_files if 'A' + date_seq[idt] in i]
    tif_files_group.append(tif_files_group_1day)

vrt_options = gdal.BuildVRTOptions(resampleAlg='near', addAlpha=None, bandList=None)
for idt in range(len(tif_files_group)):
    if len(tif_files_group[idt]) != 0:
        tif_files_toBuild = [tif_files[i] for i in tif_files_group[idt]]
        vrt_files_name = '_'.join(tif_files[tif_files_group[idt][0]].split('.')[0:2])
        gdal.BuildVRT('mosaic_sinu_' + vrt_files_name + '.vrt', tif_files_toBuild, options=vrt_options)
        exec('mosaic_sinu_' + vrt_files_name + '= None')


# 3. Mosaic the list of MODIS geotiff files and reproject to lat/lon projection

# Define target SRS
dst_srs = osr.SpatialReference()
dst_srs.ImportFromEPSG(4326) # WGS 84 projection
dst_wkt = dst_srs.ExportToWkt()
error_threshold = 0.1  # error threshold
resampling = gdal.GRA_NearestNeighbour

vrt_files = sorted(glob.glob('*.vrt'))
for idt in range(len(vrt_files)):
    # Open file
    src_ds = gdal.Open(vrt_files[idt])
    mos_file_name = '_'.join(os.path.splitext(vrt_files[idt])[0].split('_')[2:4])
    # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
    tmp_ds = gdal.AutoCreateWarpedVRT(src_ds, None, dst_wkt, resampling, error_threshold)
    # Crop to CONUS extent and create the final warped raster
    dst_ds = gdal.Translate(mos_file_name + '.tif', tmp_ds,
                            projWin=[lon_roi_min, lat_roi_max, lon_roi_max, lat_roi_min])
    dst_ds = None

    print(mos_file_name)


