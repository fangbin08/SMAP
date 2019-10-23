# Load in variables
f = h5py.File("parameters.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()
