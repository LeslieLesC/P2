# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:01:23 2018

@author: Marco
"""
import numpy as np

def hola(a,b,c):
    

    
    return a+b




def algo(hola,c):
    
    a = 1
    b = 2
    
    out = hola(a,b,c)
    
    
    return out





algo(hola,3)

#%%

def hola(a,b,c):
    

    
    return a+b


parametros = {}
parametros['callback'] = hola
parametros['variables_callback'] = [1]

def algo(parametros):
    
    hola = parametros['callback']
    variables_callback = parametros['variables_callback']
    c = variables_callback[0]
    
    a = 1
    b = 2
    
    out = hola(a,b,c)
    
    
    return out


algo(parametros)

#%%


import numpy as np

a = np.array([1,2,3,4,5])


a = np.reshape(a,5,order='F')


#%%

import threading
import numpy as np

def funcion(parametros):
    
    variables = parametros['variables']   
    callback = parametros['callback']
    
    buffer = np.zeros(30)
    
    def producer_thread():
        for i in range(5):
            callback(i,buffer,variables)

    t1 = threading.Thread(target=producer_thread, args=[])
    
    t1.start()


variables = {}
variables[0] = np.zeros(10) 

def callback(i,buffer,variables):
    
    vector = variables[0]    
    vector[i] = i
    buffer[i] = i
    print(vector)
    
  
parametros = {}
parametros['callback'] = callback
parametros['variables'] = variables

    
funcion(parametros)

#%%
arr = np.array([1,2,3])

f_handle = open('hola1.npy', 'ab')
np.save(f_handle, arr*2)
f_handle.close()

a = np.load('hola1.npy')

f = open('hola1.npy', 'rb')
for _ in range(100):
    print(np.load(f))
    
    
#%%
    
    
    
def save_to_np_file(filename,arr):
    f_handle = open(filename, 'ab')
    np.save(f_handle, arr)
    f_handle.close()    
    

arr = np.array([1,2,3])
arr1 = np.array([1,2])

    
save_to_np_file('holas2.npy',arr)   


def load_from_np_file(filename):

    f = open(filename, 'rb')
    array = np.load(f)  
    while True:
        try:
            array = np.append(array,np.load(f),axis=0)
        except:
            break
    f.close()  

    return array      


array = load_from_np_file('holas2.npy')

   
        
        
        #print(np.load(f))
    
    
    
    
    
    
    
    
    
    
    
    
    