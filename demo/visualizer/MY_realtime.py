#  Copyright (c) 2003-2024 Movella Technologies B.V. or subsidiaries worldwide.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification,
#  are permitted provided that the following conditions are met:
#
#  1.	Redistributions of source code must retain the above copyright notice,
#  	this list of conditions, and the following disclaimer.
#
#  2.	Redistributions in binary form must reproduce the above copyright notice,
#  	this list of conditions, and the following disclaimer in the documentation
#  	and/or other materials provided with the distribution.
#
#  3.	Neither the names of the copyright holders nor the names of their contributors
#  	may be used to endorse or promote products derived from this software without
#  	specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
#  THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
#  OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
#  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.THE LAWS OF THE NETHERLANDS
#  SHALL BE EXCLUSIVELY APPLICABLE AND ANY DISPUTES SHALL BE FINALLY SETTLED UNDER THE RULES
#  OF ARBITRATION OF THE INTERNATIONAL CHAMBER OF COMMERCE IN THE HAGUE BY ONE OR MORE
#  ARBITRATORS APPOINTED IN ACCORDANCE WITH SAID RULES.
#

import sys
# import xsensdeviceapi.xsensdeviceapi_py313_64 as xda
import xsensdeviceapi as xda
from threading import Lock
import os
import numpy as np
import sys
sys.path.append('.')
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
# from demo.Lab0008.lab0008_visualize import ellipse_visualize
import matplotlib.pyplot as plt
import time
import socket
from scapy.all import sniff



class XdaCallback(xda.XsCallback):
    def __init__(self, max_buffer_size = 5):
        xda.XsCallback.__init__(self)
        self.m_maxNumberOfPacketsInBuffer = max_buffer_size
        self.m_packetBuffer = list()
        self.m_lock = Lock()

    def packetAvailable(self):
        self.m_lock.acquire()
        res = len(self.m_packetBuffer) > 0
        self.m_lock.release()
        return res

    def getNextPacket(self):
        self.m_lock.acquire()
        assert(len(self.m_packetBuffer) > 0)
        oldest_packet = xda.XsDataPacket(self.m_packetBuffer.pop(0))
        self.m_lock.release()
        return oldest_packet

    def onLiveDataAvailable(self, dev, packet):
        self.m_lock.acquire()
        assert(packet != 0)
        while len(self.m_packetBuffer) >= self.m_maxNumberOfPacketsInBuffer:
            self.m_packetBuffer.pop()
        self.m_packetBuffer.append(xda.XsDataPacket(packet))
        self.m_lock.release()

plt.close('all')

# QOL settings
loadData = True

numADCSamples = 128
numTxAntennas = 2
numRxAntennas = 4
numLoopsPerFrame = 1
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64

save_dir = "D:\\bbbbbbbshe\\OpenRadar-master\\SAVE"
max_records = 100

record_count = 0

if os.path.exists(save_dir):
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    print("Creating XsControl object...")
    control = xda.XsControl_construct()
    assert(control != 0)

    xdaVersion = xda.XsVersion()
    xda.xdaVersion(xdaVersion)
    print("Using XDA version %s" % xdaVersion.toXsString())

    try:
        print("Scanning for devices...")
        portInfoArray = xda.XsScanner_scanPorts()


        # Find an MTi device
        # mtPort = xda.XsPortInfo()
        mtPort = xda.XsPortInfo("COM11", 115200)

        for i in range(portInfoArray.size()):
            if portInfoArray[i].deviceId().isMti() or portInfoArray[i].deviceId().isMtig():
                mtPort = portInfoArray[i]
                break

        if mtPort.empty():
            raise RuntimeError("No MTi device found. Aborting.")

        did = mtPort.deviceId()
        print("Found a device with:")
        print(" Device ID: %s" % did.toXsString())
        print(" Port name: %s" % mtPort.portName())

        print("Opening port...")
        if not control.openPort(mtPort.portName(), mtPort.baudrate()):
            raise RuntimeError("Could not open port. Aborting.")

        # Get the device object
        device = control.device(did)
        assert(device != 0)

        print("Device: %s, with ID: %s opened." % (device.productCode(), device.deviceId().toXsString()))

        # Create and attach callback handler to device
        callback = XdaCallback()
        device.addCallbackHandler(callback)

        # Put the device into configuration mode before configuring the device
        print("Putting device into configuration mode...")
        if not device.gotoConfig():
            raise RuntimeError("Could not put device into configuration mode. Aborting.")

        print("Configuring the device...")
        configArray = xda.XsOutputConfigurationArray()
        configArray.push_back(xda.XsOutputConfiguration(xda.XDI_PacketCounter, 0))
        configArray.push_back(xda.XsOutputConfiguration(xda.XDI_SampleTimeFine, 0))

        if device.deviceId().isGnss():
            sample_rate = 100
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_Acceleration, sample_rate))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_RateOfTurn, sample_rate))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_MagneticField, sample_rate))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_Quaternion, sample_rate))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_LatLon, sample_rate))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_AltitudeEllipsoid, sample_rate))
            configArray.push_back(xda.XsOutputConfiguration(xda.XDI_VelocityXYZ, sample_rate))

        else:
            raise RuntimeError("Unknown device while configuring. Aborting.")

        if not device.setOutputConfiguration(configArray):
            raise RuntimeError("Could not configure the device. Aborting.")

        print("Creating a log file...")
        logFileName = "logfile.mtb"
        if device.createLogFile(logFileName) != xda.XRV_OK:
            raise RuntimeError("Failed to create a log file. Aborting.")
        else:
            print("Created a log file: %s" % logFileName)

        print("Putting device into measurement mode...")
        if not device.gotoMeasurement():
            raise RuntimeError("Could not put device into measurement mode. Aborting.")

        print("Starting recording...")
        if not device.startRecording():
            raise RuntimeError("Failed to start recording. Aborting.")

        print("start setting rader")

        ims = []
        max_size = 0
        dca = DCA1000()
        record_chrip = 0


        print("Main loop")

        startTime = xda.XsTimeStamp_nowMs()
        while xda.XsTimeStamp_nowMs() - startTime <= 10000:
            if callback.packetAvailable():
                # Retrieve a packet
                packet = callback.getNextPacket()

                s = ""

                if packet.containsCalibratedData():
                    acc = packet.calibratedAcceleration()
                    s = "Acc X: %.2f" % acc[0] + ", Acc Y: %.2f" % acc[1] + ", Acc Z: %.2f" % acc[2]

                    gyr = packet.calibratedGyroscopeData()
                    s += " |Gyr X: %.2f" % gyr[0] + ", Gyr Y: %.2f" % gyr[1] + ", Gyr Z: %.2f" % gyr[2]

                    mag = packet.calibratedMagneticField()
                    s += " |Mag X: %.2f" % mag[0] + ", Mag Y: %.2f" % mag[1] + ", Mag Z: %.2f" % mag[2]




                adc_data = dca.read()

                bin_path = os.path.join(save_dir, f"frame_{record_chrip}_{xda.XsTimeStamp_nowMs()}.bin")
                adc_data.tofile(bin_path)
                record_chrip += 1
                print("saved frame")

                txt_path = os.path.join(save_dir, f"{xda.XsTimeStamp_nowMs()}.txt")
                with open(txt_path, 'w') as f:
                    f.write(s + "\n")

        print("\nStopping recording...")
        if not device.stopRecording():
            raise RuntimeError("Failed to stop recording. Aborting.")

        print("Closing log file...")
        if not device.closeLogFile():
            raise RuntimeError("Failed to close log file. Aborting.")

        print("Removing callback handler...")
        device.removeCallbackHandler(callback)

        print("Closing port...")
        control.closePort(mtPort.portName())

        print("Closing XsControl object...")
        control.close()

    except RuntimeError as error:
        print(error)
        sys.exit(1)
    except:
        print("An unknown fatal error has occured. Aborting.")
        sys.exit(1)
    else:
        print("Successful exit.")