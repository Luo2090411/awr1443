import numpy as np
import mmwave.dsp as dsp

numFrames = 12000
numADCSamples = 128
numTxAntennas = 2
numRxAntennas = 4
numLoopsPerFrame = 12000
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
dataSizeOneChirp = numADCSamples * numRxAntennas
dataSizeOneFrame = dataSizeOneChirp * numChirpsPerFrame * 2  # Complex data
numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64

range_resolution, bandwidth = dsp.range_resolution(numADCSamples, dig_out_sample_rate=6000, freq_slope_const=29.982)

Rmin_val = 0.0
Rmax_val = range_resolution * numRangeBins


data = {}
data['N_pulse'] = numDopplerBins
data['Nch'] = numRxAntennas * numTxAntennas
data['c'] = 299792458.0
data['fc'] = 77e9
data['B'] = bandwidth
data['PRI'] = 0.005  # Frame-Periodicity
data['VPC_pos0'] = [0.0, 0.0]
data['FoV_min'] = [0.0, -5]
data['FoV_max'] = [25.0, 25]

data['vego'] = [0.0, 0.381]
data['dy'] = data['c'] / (4 * data['fc'])
data['Rmin'] = np.full((data['N_pulse'], data['Nch']), Rmin_val)
data['Rmax'] = np.full((data['N_pulse'], data['Nch']), Rmax_val)

Antx = np.zeros((data['Nch'], data['N_pulse']))
Anty = np.zeros((data['Nch'], data['N_pulse']))
for ch in range(data['Nch']):
    for tau in range(data['N_pulse']):
        Antx[ch, tau] = data['VPC_pos0'][0] + data['vego'][0] * tau * data['PRI']
        Anty[ch, tau] = data['VPC_pos0'][1] + ch * data['dy'] + data['vego'][1] * tau * data['PRI']

data['Antx'] = Antx
data['Anty'] = Anty


pixel_spacing = 0.1
x_vec = np.arange(data['FoV_min'][0], data['FoV_max'][0] + pixel_spacing, pixel_spacing)
y_vec = np.arange(data['FoV_min'][1], data['FoV_max'][1] + pixel_spacing, pixel_spacing)

data['x_mat'], data['y_mat'] = np.meshgrid(x_vec, y_vec)

def bp(radar_cube):

    image_total = np.zeros_like(data['x_mat'], dtype=complex)
    Nch = data['Nch']
    N_pulse = data['N_pulse']
    data['sRC'] = radar_cube

    for N in range(Nch):
        sRC = data['sRC'][:, :, N]  # (N_pulse, Nfast)
        image = np.zeros_like(data['x_mat'], dtype=complex)
        Nfast = sRC.shape[1]

        for tau in range(N_pulse):

            rmin_val = data['Rmin'][tau, N]
            rmax_val = data['Rmax'][tau, N]
            r_vec = np.linspace(rmin_val, rmax_val, Nfast)
            rc = sRC[tau, :]

            dR = np.sqrt((data['Antx'][N, tau] - data['x_mat']) ** 2 +
                         (data['Anty'][N, tau] - data['y_mat']) ** 2)

            mask = (dR > r_vec.min()) & (dR < r_vec.max())
            if np.any(mask):

                interp_vals = np.interp(dR[mask], r_vec, rc)
                phase = np.exp(1j * 4 * np.pi * data['fc'] * dR[mask] / data['c'])
                image[mask] += interp_vals * phase
        image_total += image

    data['image'] = image_total
    return data
