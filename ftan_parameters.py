import numpy as np

# Directory of CCFs/EGFs file
input_path = "/home/shijie/data/hawaii/DISP4/CCFs/"
output_path = "/home/shijie/data/hawaii/DISP4/disp_pycode/"
is_save_fig = True
fig_path = "/home/shijie/data/hawaii/DISP4/disp_pycode/fig/"
# CF for Cross-correlation function, EGF for Empirical Green's Function
input_type = "CF" 
# Define time window by min and max surface wave velocity (km/s)
minv, maxv = 1.2, 4.2
# Period band
periods = np.arange(2.5, 10, 0.5)
# width of band pass filter (second)
band_pass_width = 0.5
# minimum ratio between interstation distance and surface wave wavelength
min_lambda_ratio = 2.0
# Initial point of image search
init_per, init_phav = 8, 3.0