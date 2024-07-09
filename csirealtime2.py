#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot CSI(Linux 802.11n CSI Tool, nexmon_csi) in real time

Usage:
    1. python3 csirealtime.py
    2. python3 csiserver.py ../material/5300/dataset/sample_0x5_64_3000.dat 3000 500
"""

import socket
import threading

import csiread
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import time
import queue
cache_len = 100
cache_data1 = [np.nan] * cache_len
cache_data2 = [np.nan] * cache_len
cache_data3 = [np.nan] * cache_len

subcarrier_num = 256
Nrx=3
Ntx=1
cache_data4 = [np.nan] * subcarrier_num

nsc_pico = 56
cache_data5 = [np.nan] * nsc_pico
#csi_list=[np.nan] * cache_len

csi_list=[np.full((30,Nrx,Ntx), np.nan) for _ in range(cache_len)]
#np_csi=np.zeros((cache_len,30,Nrx,Ntx))
mutex = threading.Lock()
data_queue=queue.Queue()

class GetDataThread(threading.Thread):
    def __init__(self, device):
        super(GetDataThread, self).__init__()
        self.address_src = ('10.167.61.207', 10086)
        self.address_des = ('52.63.92.113', 10010)
        if device == 'intel':
            self.csidata = csiread.Intel(None, Nrx, Ntx
            )
        if device == 'nexmon':
            self.csidata = csiread.Nexmon(None, chip='4358', bw=80)
        if device == 'picoscenes':
            self.csidata = csiread.Picoscenes(None, {'CSI': [nsc_pico, 3, 1]})

    def run(self):
        self.update_background()

    def update_background(self):
        # config
        global cache_data1, cache_data2, cache_data3, cache_data4, cache_data5, mutex, csi_list
        count = 0

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(self.address_des)
            while True:
                data, address_src = s.recvfrom(4096)    # 65507
                msg_len = len(data)
                code = self.csidata.pmsg(data)
                if code == 0xbb:    # intel
                    scaled_csi_sm = self.csidata.get_scaled_csi()
                    tem_time=self.csidata.timestamp_low        
                    mutex.acquire()                   
                    csi_list.pop(0)
                    csi_list.append(scaled_csi_sm[0])
                    mutex.release()                   
                    count += 1
                    
                if code == 0xf100:  # nexmon
                    mutex.acquire()
                    cache_data4 = np.fft.fftshift(self.csidata.csi[0])
                    mutex.release()
                    count += 1
                if code == 0xf300:  # picoscenes
                    mutex.acquire()
                    cache_data5 = self.csidata.raw[0]["CSI"]["CSI"][:, 0, 0]
                    mutex.release()
                    count += 1
                # once we receive 100,start algorithm
                if count % cache_len == 0:
                    print(count)
                    np_csi=np.zeros((cache_len,30,Nrx,Ntx))
                    mutex.acquire()
                    np_csi=np.array(csi_list)                  
                    data_queue.put(np_csi)  
                    mutex.release()
                    print('receive %d bytes [msgcnt=%u]' % (msg_len, count))
                    
                    #plot_realtime()
                    
                    #time.sleep(1)
                                       
                    #csi_list.clear()
                    csi_list=[np.full((30,Nrx,Ntx), np.nan) for _ in range(cache_len)]
                    np_csi=np.zeros((cache_len,30,Nrx,Ntx))
                    count=0
def plot_realtime():

    fig, ax = plt.subplots()
    plt.title('csi-amplitude')
    plt.xlabel('packets')
    plt.ylabel('amplitude')
    ax.set_ylim(0, 40)
    ax.set_xlim(0, cache_len)
    x = np.arange(0, cache_len, 1)
    
    line1,  = ax.plot(x, np.abs(cache_data1), linewidth=1.0, label='subcarrier_15_0_0')

    plt.legend()

    def init():
        line1.set_ydata([np.nan] * len(x))

        return line1,

    def animate(i):
        global  mutex
       #print(csi_list)
        #mutex.acquire()
        np_csi=data_queue.get()
        
        line1.set_ydata(np.abs(np_csi[:,0,0,0]))
        #print(np.abs(np_csi[:,0,0,1]))
        #mutex.release()
        #np_csi=np.array(csi_list)
        return line1,

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=True)
    plt.show()
    
def realtime_plot_intel():
    #np_csi=np.array(csi_list)
    #print(np_csi.shape)
    #print(len(csi_list[0]))
    #print(type(np_csi))
  
     '''
    fig, ax = plt.subplots()
    plt.title('csi-amplitude')
    plt.xlabel('packets')
    plt.ylabel('amplitude')
    ax.set_ylim(0, 40)
    ax.set_xlim(0, cache_len)
    x = np.arange(0, cache_len, 1)

    line1,  = ax.plot(x, np.abs(cache_data1), linewidth=1.0, label='subcarrier_15_0_0')
    #line2,  = ax.plot(x, np.abs(cache_data2), linewidth=1.0, label='subcarrier_15_1_0')
    #line3,  = ax.plot(x, np.abs(cache_data3), linewidth=1.0, label='subcarrier_15_2_0')
    plt.legend()

    def init():
        line1.set_ydata([np.nan] * len(x))
        #line2.set_ydata([np.nan] * len(x))
        #line3.set_ydata([np.nan] * len(x))
        return line1,

    def animate(i):
        global cache_data1, cache_data2, cache_data3, mutex, csi_list
        print(csi_list)
        mutex.acquire()
        line1.set_ydata(np.abs(cache_data1))
        #line2.set_ydata(np.abs(cache_data2))
        #line3.set_ydata(np.abs(cache_data3))
        mutex.release()
        np_csi=np.array(csi_list)
        #print(csi_list)
        return line1,

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1000/25, blit=True)
    plt.show()
'''

def realtime_plot_nexmon():
    fig, ax = plt.subplots()
    plt.title('csi-amplitude')
    plt.xlabel('subcarrier')
    plt.ylabel('amplitude')
    ax.set_ylim(0, 4000)
    ax.set_xlim(0, subcarrier_num)
    x = np.arange(0, subcarrier_num)

    line4,  = ax.plot(x, np.abs(cache_data4), linewidth=1.0, label='subcarrier_256')
    plt.legend()

    def init():
        line4.set_ydata([np.nan] * subcarrier_num)
        return line4,

    def animate(i):
        global cache_data4, mutex
        mutex.acquire()
        line4.set_ydata(np.abs(cache_data4))
        mutex.release()
      
        return line4,

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1000/25, blit=True)
    plt.show()


def realtime_plot_picoscenes():
    print("Warning: Picoscenes.pmsg method hasn't been READY!")
    fig, ax = plt.subplots()
    plt.title('csi-amplitude')
    plt.xlabel('subcarrier')
    plt.ylabel('amplitude')
    ax.set_ylim(0, 500)
    ax.set_xlim(0, nsc_pico)
    x = np.arange(0, nsc_pico)

    line5,  = ax.plot(x, np.abs(cache_data5), linewidth=1.0, label='subcarrier_56')
    plt.legend()

    def init():
        line5.set_ydata([np.nan] * nsc_pico)
        return line5,

    def animate(i):
        global cache_data5, mutex
        mutex.acquire()
        line5.set_ydata(np.abs(cache_data5))
        mutex.release()
        
        return line5,

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1000/25, blit=True)
    plt.show()


def realtime_plot(device):
    task = GetDataThread(device)
    task.start()
    plot_realtime()
    #print(res)
    #eval('realtime_plot_' + device)()


if __name__ == '__main__':
    realtime_plot('intel')
