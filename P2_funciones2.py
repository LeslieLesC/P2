# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from sys import stdout
import numpy.fft as fft
import nidaqmx
import numpy as np
import numpy.fft as fft
import os
import nidaqmx.constants as constants
import nidaqmx.stream_writers

"""
funciones para el PID2

"""
def save_folder(foldername,filename,vectorname, vector):

    if not os.path.exists(foldername):
        os.mkdir(filename)

    np.save(os.path.join(foldername, vectorname+filename),vector)


class PIDController:

    def __init__(self, setpoint, kp=1.0, ki=0.0, kd=0.0):

        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.last_error = 0
        self.p_term = 0
        self.i_term = 0
        self.d_term = 0

    def calculate(self, feedback_value):
        error = self.setpoint - feedback_value

        delta_error = error - self.last_error

        self.p_term = self.kp * error
        self.i_term += error
        self.d_term = delta_error

        self.last_error = error

        return self.p_term + (self.ki * self.i_term) + (self.kd * self.d_term)

def config_co_channel(task, device, duty_cycle, freq):
    """
    define el canal digital continuo con frecuencia en Hz
    """
    do = task.co_channels.add_co_pulse_chan_freq(counter=device,
                                            duty_cycle=duty_cycle,
                                            freq=freq,
                                            units=nidaqmx.constants.FrequencyUnits.HZ)
    task.timing.cfg_implicit_timing(sample_mode=constants.AcquisitionType.CONTINUOUS)
    return do

def config_ai_channel(task, device, max_range_value, min_range_value,name, samplerate,samples):
    """
    define el canal analógico
    """
    task.ai_channels.add_ai_voltage_chan(device,max_val=max_range_value, min_val=min_range_value,
                                         name_to_assign_to_channel=name,
                                         terminal_config=constants.TerminalConfiguration.RSE)#, "Voltage")#,AIVoltageUnits.Volts)#,max_val=10., min_val=-10.)
    task.timing.cfg_samp_clk_timing(samplerate,samps_per_chan=samples,
                                    sample_mode=constants.AcquisitionType.CONTINUOUS)

    return
