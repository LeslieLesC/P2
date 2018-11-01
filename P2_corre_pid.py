# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:15:03 2018

@author: Marco
"""

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import datetime
import time
import matplotlib.pylab as pylab
from scipy import signal
from sys import stdout
import numpy.fft as fft
import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import os
from P2_funciones import pid_daqmx
from P2_funciones import save_to_np_file
from P2_funciones import load_from_np_file
import nidaqmx.constants as constants
import nidaqmx.stream_writers
import time




def callback(i, input_buffer, output_buffer_duty_cycle, output_buffer_frequency, buffer_chunks, initial_pid_duty_cycle, initial_pid_frequency, callback_variables, pid_variables):
    
        # Parametros de entrada del callback
        vector_mean_ch1 = callback_variables[0]
        path_vector_mean_ch1 = callback_variables[1]
        path_duty_cycle = callback_variables[2]
        path_input_buffer = callback_variables[3]
        
        # Parametros PID
        valor_esperado = pid_variables[0]
        vector_error = pid_variables[1]
        constantes_pid = pid_variables[2]
        kp = constantes_pid[0]
        ki = constantes_pid[1]
        kd = constantes_pid[2]
        #####################################
    
        # Leo del buffer y 
        ch1_array = input_buffer[i,:,0]          
        mean_ch1 = np.mean(ch1_array)
        vector_mean_ch1[i] = mean_ch1
        error = mean_ch1 - valor_esperado
        vector_error[i] = error

        j = i-1
        if j == -1:
            j == buffer_chunks-1
        
        # Algoritmo PID
        output_buffer_duty_cycle_i = output_buffer_duty_cycle[j] + kp*error + ki*np.sum(vector_error) + kd*(vector_error[i]-vector_error[j])
        
        if output_buffer_duty_cycle_i > 0.99:
            output_buffer_duty_cycle_i = 0.99
        if output_buffer_duty_cycle_i < 0.01:
            output_buffer_duty_cycle_i = 0.01  

        if i == buffer_chunks-1:
           save_to_np_file(path_duty_cycle,output_buffer_duty_cycle)               
           save_to_np_file(path_vector_mean_ch1,vector_mean_ch1)   
           save_to_np_file(path_input_buffer,input_buffer)   
                   
        output_buffer_frequency_i = initial_pid_frequency

        return output_buffer_duty_cycle_i, output_buffer_frequency_i



##

# Variables 
buffer_chunks = 100
initial_pid_duty_cycle = 0.5
initial_pid_frequency = 200
ai_channels = 1
ai_samples = 1000
ai_samplerate = 50000

# Variables Callback
vector_mean_ch1 = np.zeros(buffer_chunks) 
path_vector_mean_ch1 = ''
path_duty_cycle = ''
path_input_buffer = ''

# Variables Callback PID
valor_esperado = 4.6
vector_error = np.zeros(buffer_chunks)
kp = 0.1
ki = 0.5
kd = 0.3
constantes_pid = [kp,ki,kd]

##
callback_variables = {}
callback_variables[0] = vector_mean_ch1
callback_variables[1] = path_vector_mean_ch1
callback_variables[2] = path_duty_cycle
callback_variables[3] = path_input_buffer

pid_variables = {}
pid_variables[0] = valor_esperado
pid_variables[1] = vector_error
pid_variables[2] = constantes_pid


parametros = {}
parametros['buffer_chunks'] = buffer_chunks
parametros['ai_channels'] = ai_channels    
parametros['ai_samples'] = ai_samples
parametros['ai_samplerate'] = ai_samplerate
parametros['initial_pid_duty_cycle'] = initial_pid_duty_cycle
parametros['initial_pid_frequency'] = initial_pid_frequency
parametros['callback'] = callback    
parametros['callback_variables'] = callback_variables
parametros['pid_variables'] = pid_variables

input_buffer, output_buffer_duty_cycle, output_buffer_frequency = pid_daqmx(parametros)