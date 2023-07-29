import numpy as np
from os.path import join

# Directory of CCFs/EGFs file
base_path = "/home/shijie/data/hawaii/DISP4/disp_pycode_mpi/"
input_path = join(base_path, "CCFs")
output_path = join(base_path, "disps")
is_save_fig = False
fig_path = join(base_path, "fig")
# CF for Cross-correlation function, EGF for Empirical Green's Function
input_type = "CF" 
# Define time window by min and max surface wave velocity (km/s)
minv, maxv = 1.2, 4.2
# Period band
periods = np.arange(2.5, 10, 0.5)
# filter design, for ripple and width, refer to the doc of scipy.signal.kaiserord
ripple = 65
width = 0.01
# width of band pass filter (second)
band_pass_width = 0.5
# minimum ratio between interstation distance and surface wave wavelength
min_lambda_ratio = 2.0
# minimum distance (km)
min_dist = 4

search_strategy = 'ref_curve' # 'point' or 'ref_curve'
######## point #################
# Initial point of image search
init_per, init_phav = 8, 3.0
####### ref curve ##############
# Reference dispersion curve
# 1st column period (s), 2nd column velocity (km/s)
ref_disp_path = join(base_path, 'ref_RayPhaseVdisp_hawaii.txt')
