import numpy as np

# Directory of CCFs/EGFs file
input_path = "/home/shijie/data/huidong/DISP_2_1/CCFs/"
output_path = "/home/shijie/data/huidong/DISP_2_1/disp/"
is_save_fig = True
fig_path = "/home/shijie/data/huidong/DISP_2_1/disp/fig/"
# CF for Cross-correlation function, EGF for Empirical Green's Function
input_type = "CF" 
# Define time window by min and max surface wave velocity (km/s)
minv, maxv = 1.6, 5.0
# Period band
periods = np.arange(0.5, 3.0, 0.25)
# width of band pass filter (second)
band_pass_width = 0.1
# minimum ratio between interstation distance and surface wave wavelength
min_lambda_ratio = 2.0
# Initial point of image search
init_per, init_phav = 2, 4.0
# Parameters for filter design
ripple, width = 63, 0.05
