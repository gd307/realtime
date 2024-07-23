#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot CSI (Linux 802.11n CSI Tool, nexmon_csi) in real time

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
import queue
import os  # For beep command on Linux and macOS
import pandas as pd
from datetime import datetime
from get_model import real_time_predict, load_model_and_encoder
from scipy.stats import mode
 
cache_len = 30
Nrx = 3
Ntx = 3

csi_list = [np.full((30, Nrx, Ntx), np.nan) for _ in range(cache_len)]
mutex = threading.Lock()
data_queue = queue.Queue()
plot_queue = queue.Queue()

# 预定义的坐标列表
alarm_coords = np.array([[2, 0]])

# 定义投票次数
vote_count = 3
votes = []

# 创建一个DataFrame来保存预测结果和时间
results_df = pd.DataFrame(columns=['Time', 'X', 'Y'])
output_file='loc.csv'
# 定义一个函数来播放警报声
def play_alarm_sound(frequency=1000, duration=1000):
    os.system(f'play -nq -t alsa synth {duration/1000} sine {frequency}')
    # 或者使用 beep
    # os.system(f'beep -f {frequency} -l {duration}')

class GetDataThread(threading.Thread):
    def __init__(self, device):
        super(GetDataThread, self).__init__()
        self.address_des = ('0.0.0.0', 10000)
        if device == 'intel':
            self.csidata = csiread.Intel(None, Nrx, Ntx)

    def run(self):
        self.update_background()

    def update_background(self):
        global mutex, csi_list
        count = 0

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(self.address_des)
            while True:
                data, address_src = s.recvfrom(4096)
                msg_len = len(data)
                code = self.csidata.pmsg(data)
                if code == 0xbb:  # intel
                    scaled_csi_sm = self.csidata.get_scaled_csi()
                    tem_time = self.csidata.timestamp_low
                    mutex.acquire()
                    csi_list.pop(0)
                    csi_list.append(scaled_csi_sm[0])
                    mutex.release()
                    count += 1

                # Once we receive 100, start algorithm
                if count % cache_len == 0:
                    np_csi = np.array(csi_list)
                    mutex.acquire()
                    data_queue.put(np_csi)
                    mutex.release()
                    print('receive %d bytes [msgcnt=%u]' % (msg_len, count))
                    csi_list = [np.full((30, Nrx, Ntx), np.nan) for _ in range(cache_len)]
                    count = 0

class ProcessThread(threading.Thread):
    def __init__(self):
        super(ProcessThread, self).__init__()
        self.model, self.label_encoder, self.reverse_label_encoder = load_model_and_encoder()

    def run(self):
        global votes, results_df
        while True:
            np_csi_now = data_queue.get()
            pre = real_time_predict(np_csi_now, self.model, self.label_encoder, self.reverse_label_encoder)
            votes.append(pre)
            #plot_queue.put(pre)
            
            # 如果达到投票次数
            if len(votes) >= vote_count:
                # 使用多数投票法
                major_vote = self.major_vote(votes)
                print(major_vote)
                plot_queue.put(major_vote)
                
                # 保存预测坐标和当前时间到DataFrame
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                for coord in major_vote:
                    results_df.loc[len(results_df)] = [current_time, coord[0], coord[1]]
                
                # 实时保存到Excel文件
                results_df.to_csv(output_file, index=False)
                
                votes = []  # 重置投票列表
                
    def major_vote(self, votes):
        """Majority voting method to determine the final prediction."""
        votes_array = np.array(votes)
        major_vote = mode(votes_array,axis=0,keepdims=True).mode[0]
        return major_vote


def realtime_plot(device):
    task = GetDataThread(device)
    task.start()
    process_thread = ProcessThread()
    process_thread.start()

if __name__ == '__main__':
    realtime_plot('intel')
