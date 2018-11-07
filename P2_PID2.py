# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:06:14 2018

@author: Marco
"""
import numpy as np
import matplotlib.pyplot as plt
import threading
import datetime
import matplotlib.pylab as pylab
from scipy import signal
from sys import stdout
import numpy.fft as fft
import matplotlib.pyplot as plt
import numpy.fft as fft
import os
import  P2_funciones2 
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.stream_writers
import time

#%%
digital_channel = 'Dev1/ctr0'
freq = 20
duty_cycle = 0.5
analog_channel = 'Dev1/ctr0'
max_range_value =2.
min_range_value = -2.
name = ch1
samplerate_ai = 50000
samples = 1000

chunks_buffer = 100

chunks_input_buffer = chunks_buffer
chunks_output_buffer = chunks_buffer

vector_valor_medido = np.zeros(100000)
vector_duty_cycle = np.zeros(100000)

output_channels = 1
input_channels = 1

valor_esperado = 4.6
kp = 1.
ki = 0.
kd = 0.
# Defino los buffers
input_buffer = np.zeros([chunks_input_buffer,samples])
output_buffer = np.zeros(chunks_output_buffer)

# Semaforos
semaphore1 = threading.Semaphore(0)
semaphore2 = threading.Semaphore(0)

pid = PIDController(valor_esperado,kp,ki,kd)


while True:

    signal = read()

    actuator = lazo.calculate(signal)

    write(actuator)
# Defino el thread que envia la señal
def producer():

    with nidaqmx.Task() as task_do:

        do = config_co_channel(task_do,device,freq,duty_cycle)
        digi_s = nidaqmx.stream_writers.CounterWriter(task_do.out_stream)
        task_do.start()

        i = 0
        while producer_exit[0] is False:

            semaphore2.acquire()

            digi_s.write_one_sample_pulse_frequency(frequency=200.0,duty_cycle=output_buffer[i])

            i = i+1
            i = i%chunks_output_buffer


# Defino el thread que adquiere la señal
def consumer():

    output_buffer_ant = 0.5

    with nidaqmx.Task() as task_ai:
        config_ai_channel(task_ai,max_range_value,min_range_value,name,samplerate_ai,samples)

        i = 0
        j = 0
        m = 0
        while consumer_exit[0] is False:

            if semaphore1._value > chunks_input_buffer:
                print('Hay overrun en la lectura! \n')

            if semaphore2._value > chunks_input_buffer:
                print('Hay overrun en la escritura! \n')

            semaphore1.release()

            b=task_ai.read(number_of_samples_per_channel=samples)
            medicion = np.asarray(b)
            input_buffer[j,:] = medicion
            semaphore1.acquire()

            j = j+1
            j = j%chunks_input_buffer


            ## Inicio Callback
            valor_medio_medido = np.mean(medicion)
            vector_valor_medido[m] = valor_medio_medido
            # error = valor_medio_medido - valor_esperado

            output_buffer[i] = pid.calculate(valor_medio_medido)

            # output_buffer[i] = output_buffer_ant + 0.1*error
            if output_buffer[i] > 0.99:
                output_buffer[i] = 0.99
            if output_buffer[i] < 0.01:
                output_buffer[i] = 0.01
            # output_buffer_ant = output_buffer[i]
            vector_duty_cycle[m] = output_buffer[i]

            # vector_duty_cycle[m] = output_buffer_ant

            ## Fin callback

            semaphore2.release()

            i = i+1
            i = i%chunks_output_buffer

            m = m+1
            m = m%len(vector_duty_cycle)

    return vector_duty_cycle,vector_valor_medido

# Variables de salida de los threads
producer_exit = [False]
consumer_exit = [False]

# Inicio los threads
print (u'\n Inicio barrido \n Presione Ctrl + c para interrumpir.')
t1 = threading.Thread(target=producer, args=[])
t2 = threading.Thread(target=consumer, args=[])

t1.start()
t2.start()

# Salida de la medición
while not producer_exit[0] or not consumer_exit[0]:
    try:
        time.sleep(0.2)
    except KeyboardInterrupt:
        consumer_exit[0] = True
        producer_exit[0] = True
        time.sleep(0.2)
        print ('\n \n Medición interrumpida \n')



plt.plot(np.transpose(input_buffer[:,:]))

fig,ax = plt.subplots(nrows=1,ncols=1)
ax1 = ax.twinx()
ax.plot(vector_duty_cycle,color='blue')
ax1.plot(vector_valor_medido,color='red')
ax1.set_ylim([4,5])


#%%

folder = 'PID'
save_folder(folder,'archivo','vectoduty',vector_duty_cycle)


#%%
# carpeta_salida = 'PID'
# sub_archivo = 'con_correccion_perturbacion_seno_10hz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
#
# #%%
# carpeta_salida = 'PID'
# sub_archivo = 'sin_correccion_perturbacion_seno_1hz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
#
# #%%
# carpeta_salida = 'PID'
# sub_archivo = 'con_correccion_perturbacion_seno_1hz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
#
# #%%
# carpeta_salida = 'PID'
# sub_archivo = 'sin_correccion_perturbacion_seno_100mhz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
#
# #%%
# carpeta_salida = 'PID'
# sub_archivo = 'con_correccion_perturbacion_seno_100mhz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
#
# #%%
# carpeta_salida = 'PID'
# sub_archivo = 'sin_correccion_perturbacion_seno_500mhz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
#
# #%%
# carpeta_salida = 'PID'
# sub_archivo = 'con_correccion_perturbacion_seno_500mhz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
#
# #%%
# carpeta_salida = 'PID'
# sub_archivo = 'sin_correccion_perturbacion_seno_300mhz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
#
# #%%
# carpeta_salida = 'PID'
# sub_archivo = 'con_correccion_perturbacion_seno_300mhz'
#
# if not os.path.exists(carpeta_salida):
#     os.mkdir(carpeta_salida)
#
# np.save(os.path.join(carpeta_salida, 'vector_duty_cycle'+sub_archivo),vector_duty_cycle)
# np.save(os.path.join(carpeta_salida, 'vector_valor_medido'+sub_archivo),vector_valor_medido)
