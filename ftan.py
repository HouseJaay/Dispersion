import numpy as np
import sys, os
import importlib
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy import fftpack
from obspy.geodetics.base import gps2dist_azimuth
from glob import glob
from os.path import join
import matplotlib.pyplot as plt

try:
    proj_path = sys.argv[1]
    proj_name = sys.argv[2]
except IndexError:
    print("Error, Need to provide project path")
    print("For example, ftan_parameters.py is in /home/shijie/project/sembawang/tomo/checker1/")
    print("1st parameter: /home/shijie/project")
    print("2cn parameter: sembawang.tomo.checker1")
    exit()

sys.path.append(proj_path)
par = importlib.import_module(proj_name + '.ftan_parameters')

def read_data(fpath):
    with open(fpath, 'r') as f:
        tmp = (f.readline()).split()
        lon1, lat1 = float(tmp[0]), float(tmp[1])
        tmp = (f.readline()).split()
        lon2, lat2 = float(tmp[0]), float(tmp[1])
    data = np.loadtxt(fpath, skiprows=2)
    maxamp = np.max(data[:,[1,2]])
    if maxamp > 0:
        data[:,[1,2]] /= maxamp
    if par.input_type == "CF":
        data[:,1] = np.imag(hilbert(data[:,1]))
        data[:,2] = np.imag(hilbert(data[:,2]))
    d, _, __ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    dist = d / 1000.0
    return data, dist


def cos_window(npts, n1, n2):
    window = np.zeros(npts)
    window[n1:n2] = 1
    taper_n = round(n2-n1)*0.05
    window[n1:(n1+taper_n)] = np.sin(0.5*np.pi*np.arange(taper_n)/taper_n)
    window[(n2-taper_n):n2] = np.sin(0.5*np.pi*np.arange(taper_n,0,-1)/taper_n)
    return window


falpha = interp1d([0, 100, 250, 500, 1000, 2000, 4000, 20000],
                      [5, 8, 12, 20, 25, 35, 50, 75])

def gauss_window(alpha, freqs, cf):
    return np.exp(-alpha * ((freqs-cf)/cf)**2)

def envelope_image(signal, samplef, periods, dist):
    alpha = falpha(dist)
    image = np.zeros(len(periods), len(signal))
    nfft = fftpack.next_fast_len(len(signal))
    fft = fftpack.rfft(signal, nfft)
    for ip in range(len(periods)):
        cf = 1.0 / periods[ip]


all_inputs = glob(join(par.input_path, "*"))
os.makedirs(par.output_path, exist_ok=True)
if par.is_save_fig:
    os.makedirs(par.fig_path, exist_ok=True)


for fpath in all_inputs:
    data, dist = read_data(fpath)
    time = data[:,0]
    delta = time[1] - time[0]
    samplef = 1.0 / delta
    n1 = int(dist/par.maxv/delta)
    n2 = int(dist/par.minv/delta)
    if n2 > len(time)-1:
        n2 = len(time)-1
    window = cos_window(len(time), n1, n2)
    stack_egf = data[:,1] + data[:,2]
    noise = stack_egf[n2:]
    stack_egf *= window
    envelope_image(stack_egf, samplef, par.periods, dist)
    