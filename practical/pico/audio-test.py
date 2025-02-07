

import os

from nptdms import TdmsFile

import nidaqmx
from nidaqmx.constants import (
    READ_ALL_AVAILABLE,
    AcquisitionType,
    LoggingMode,
    LoggingOperation,
)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai0")
    task.timing.cfg_samp_clk_timing(44000.0, sample_mode=AcquisitionType.FINITE, samps_per_chan=44000)
    task.in_stream.configure_logging(
        "TestData.tdms", LoggingMode.LOG_AND_READ, operation=LoggingOperation.OPEN_OR_CREATE
    )
    for i in range(3):
        task.read(READ_ALL_AVAILABLE)

with TdmsFile.open("TestData.tdms") as tdms_file:
    for group in tdms_file.groups():
        for channel in group.channels():
            data = channel[:]
            print("Read data from TDMS file: [" + ", ".join(f"{value:f}" for value in data) + "]")

