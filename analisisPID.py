# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

folder = 'PID2\\'
duty = 'experimento2_duty_cycle.bin'
mean = 'experimento2_mean_data.bin'
parentDirectory = os.path.abspath(os.getcwd())
file_duty = os.path.join(parentDirectory, folder, duty)
file_mean = os.path.join(parentDirectory, folder, mean)

def load_from_np_file(filename):

    f = open(filename, 'rb')
    arr = np.load(f)
    while True:
        try:
            arr = np.append(arr,np.load(f),axis=0)
        except:
            break
    f.close()

    return arr

arr1 = load_from_np_file(file_duty)
arr2 = load_from_np_file(file_mean)

plt.plot(arr1)
plt.plot(arr2)

plt.show()
