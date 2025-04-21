import numpy as np
import sys
sys.path.append('.')
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt
import time
import BP

numFrames = 12000
numADCSamples = 128
numTxAntennas = 2
numRxAntennas = 4
numLoopsPerFrame = 1
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
dataSizeOneChirp = numADCSamples * numRxAntennas
dataSizeOneFrame = dataSizeOneChirp * numChirpsPerFrame * 2
dataSize = dataSizeOneFrame * numFrames

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64

if __name__ == '__main__':
    ims = []
    max_size = 0
    dca = DCA1000()

    last_time = None
    total_time = 0
    time_diff = None
    record_frame = 0
    start_timestamp = time.time()

    plt.ion()
    fig, ax = plt.subplots()
    while True:
        for i in range(numFrames):
            one_frame = np.fromfile(f'./SAVE/adc_data_{i+1}.bin', dtype=np.uint16)
            timeDomainData = one_frame - (one_frame >= 2 ** 15) * 2 ** 16
            rawData8 = timeDomainData.reshape(8, (int(len(one_frame) / 8)), order='F')
            rawDataI = np.reshape(rawData8[0:4, :], [dataSizeOneFrame//2, 1], order='F')
            rawDataQ = np.reshape(rawData8[4:8, :], [dataSizeOneFrame//2, 1], order='F')
            frameData = np.hstack((rawDataI, rawDataQ))
            frameCplx = frameData[:, 0] + 1j * frameData[:, 1]
            temp = np.reshape(frameCplx, [dataSizeOneChirp, numChirpsPerFrame], order='F')
            frameComplex = np.zeros((numChirpsPerFrame, numRxAntennas, numADCSamples), dtype=complex)
            multi_frame = np.zeros((numDopplerBins, numRxAntennas*numTxAntennas, numRangeBins), dtype=complex)

            for chirp in range(numLoopsPerFrame * numTxAntennas):
                frameComplex[chirp, :, :] = np.reshape(temp[:, chirp], [4, 128], order='F')

            TDM_radar = dsp.separate_tx(frameComplex, numTxAntennas)
            TDM_radar = dsp.range_processing(TDM_radar, window_type_1d=None)
            radar_cube = np.transpose(TDM_radar, (0, 2, 1))

            data = BP.bp(radar_cube)
            image_bp = np.abs(data['image'])
            image_bp = np.transpose(image_bp)
            y_vec = BP.y_vec
            x_vec = BP.x_vec

            ax.pcolormesh(y_vec, x_vec, image_bp, shading='auto', cmap='terrain', vmin=0.0)
            plt.xlabel('y [m]')
            plt.ylabel('x [m]')

            plt.pause(0.1)
            ax.cla()