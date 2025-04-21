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
        adc_data = np.fromfile('./adc_data_goodcondition_1634_Raw_0.bin', dtype=np.uint16)
        N_frame = dataSizeOneFrame * 0
        for i in range(0, len(adc_data), dataSize):
            multi_frame = adc_data[i+N_frame: i + dataSize + N_frame]
            timeDomainData = multi_frame - (multi_frame >= 2 ** 15) * 2 ** 16
            rawData8 = timeDomainData.reshape(8, (int(len(multi_frame) / 8)), order='F')
            rawDataI = np.reshape(rawData8[0:4, :], [dataSizeOneFrame//2 * numFrames, 1], order='F')
            rawDataQ = np.reshape(rawData8[4:8, :], [dataSizeOneFrame//2 * numFrames, 1], order='F')
            frameData = np.hstack((rawDataI, rawDataQ))
            frameCplx = frameData[:, 0] + 1j * frameData[:, 1]
            temp = np.reshape(frameCplx, [dataSizeOneChirp, numChirpsPerFrame * numFrames], order='F')
            frameComplex = np.zeros((numChirpsPerFrame * numFrames, numRxAntennas, numADCSamples), dtype=complex)
            multi_frame = np.zeros((numDopplerBins * numFrames, numRxAntennas*numTxAntennas, numRangeBins), dtype=complex)

            for chirp in range(numTxAntennas*numFrames):
                frameComplex[chirp, :, :] = np.reshape(temp[:, chirp], [4, 128], order='F')

            TDM_radar = dsp.separate_tx(frameComplex, numTxAntennas)
            TDM_radar = dsp.range_processing(TDM_radar, window_type_1d=None)
            radar_cube = np.transpose(TDM_radar, (0, 2, 1))

            data = BP.bp(radar_cube)
            image_bp = np.abs(data['image'])
            image_bp = np.transpose(image_bp)
            image_bp = (image_bp - np.min(image_bp)) / np.max(image_bp)
            y_vec = BP.y_vec
            x_vec = BP.x_vec

            ax.pcolormesh(y_vec, x_vec, image_bp, shading='auto', cmap='terrain', vmin=0.1)

            plt.xlabel('y [m]')
            plt.ylabel('x [m]')


            plt.pause(0.1)
            ax.cla()
            break