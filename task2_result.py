import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def type_normalize(value):
    try:
        res = float(value)
    except ValueError:
        res_vrem = int(value, 16)
        res = float(res_vrem)
    finally:
        return res 

def get_files_txt(name):
    inp = name + '.txt'
    with open(inp) as f:
        b = np.loadtxt(inp)
    return b

def get_files(name):
    inp = name + '.csv'
    with open(inp, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        read = [row for row in reader]
    vrem_arr = []
    for i in range(0, len(read), 2):
        vrem_arr.append(read[i])
    nump_arr_w1 = np.array(vrem_arr)
    return nump_arr_w1

def neiro(x, kolvo):
    
    ops.reset_default_graph()

    sess = tf.Session()

    x_ = tf.placeholder(name="input", shape=[None, kolvo], dtype=tf.float32)

    hidden_neurons = 8
    hidden_neurons_2 = 2
    w1 = tf.convert_to_tensor(get_files('w1'), dtype=tf.float32)
    b1 = tf.convert_to_tensor(get_files_txt('b1'), dtype=tf.float32)
    layer1 = tf.nn.softplus(tf.add(tf.matmul(x_, w1), b1))
    
    w2 = tf.convert_to_tensor(get_files('w2'), dtype=tf.float32)
    b2 = tf.convert_to_tensor(get_files_txt('b2'), dtype=tf.float32)
    layer2 = tf.nn.softplus(tf.add(tf.matmul(layer1, w2), b2))

    w3 = tf.convert_to_tensor(get_files('w3'), dtype=tf.float32)
    b3 = tf.convert_to_tensor(get_files_txt('b3'), dtype=tf.float32)

    nn_output = tf.nn.softplus(tf.add(tf.matmul(layer2, w3), b3))
    nn_res = np.around(sess.run(nn_output, feed_dict={x_: x}) * 6)
    arr_res = []
    for i in range(len(nn_res)):
        tek = []
        tek.append(str(i))
        tek.append(int(nn_res[i]))
        arr_res.append(tek)
    with open('RES.csv', 'w', newline='') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(arr_res)
    return

def user_db(db, mode, kolvo, value_sold):
    if mode == 'x':
        value = value_sold
        val_start = 0
        minmax = np.zeros((2, value))
        for i in range(value):
            if str(db[0][i]) != 'nan':
                minmax[0][i] = db[0][i]
                minmax[1][i] = db[0][i]
            else:
                minmax[0][i] = 1e-15
                minmax[1][i] = 1e+15
        for i in range(kolvo):
            for j in range(value):
                if db[i][j] < minmax[0][j]:
                    minmax[0][j] = db[i][j]
                if db[i][j] > minmax[1][j]:
                    minmax[1][j] = db[i][j]
        for i in range(kolvo):
            for j in range(value):
                if float(minmax[0][j]) == 0.0:
                    minmax[0][j] = 0.0001
                if float(minmax[1][j]) == 0.0:
                    minmax[1][j] = -0.0001
    elif mode == 'y':
        value = 63
        val_start = 62
    user_array = np.zeros((kolvo, value_sold))
    for i in range(kolvo):
        for j in range(val_start, value):
            if mode != 'y' and str(db[i][j]) != 'nan':
                if minmax[0][j] != minmax[1][j]:
                    znach = float(1 * (db[i][j] - minmax[0][j]) / (minmax[1][j] - minmax[0][j]))
                else:
                    znach = 1.0
                user_array[i][j - val_start] = znach
            elif mode != 'y' and str(db[i][j]) == 'nan':
                user_array[i][j - val_start] = 0.0
            else:
                user_array[i][j - val_start] = db[i][j] / 6
    return user_array

def get_id(data, kolvo):
    base = []
    base.append(data['ID'].tolist())
    data_array = np.array(base)
    with open('ID.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(data_array)
    return data_array
        
def get_data(data, mode, kolvo):
    value = 62
    if mode == 'x':
        is_y = 0
    elif mode == 'xy':
        is_y = 1
    base = []
    for i in range(value):
        index = 'x' + str(i)
        base.append(data[index].tolist())
    if is_y == 1:
        base.append(data['y'].tolist())
    data_array = np.zeros((kolvo, value + is_y))
    for i in range(kolvo):
        for j in range(value + is_y):
            znach = base[j][i]
            if znach == None or str(znach) == 'nan':
                znach = 'nan'           
            else:
                znach = type_normalize(znach)
            if str(znach) == 'inf':
                rezzy = str(base[j][i])[0:-2]
                rez_kon = str(base[j][i])[-2:]
                r_v1 = int(rezzy, 16)
                r_v2 = int(rez_kon, 16)
                r_1 = float(r_v1)
                r_2 = float(r_v2)
                znach = r_1 * pow(16, 2) + r_2
            data_array[i][j] = znach
    return data_array


def main():
    filename2 = 'test.csv'
    df2 = pd.read_csv(filename2, low_memory=False) 
    count = 50000
    test_value = 62
    database2 = get_data(df2, 'x', count)
    X2 = user_db(database2, 'x', count, test_value)
    neiro(X2, test_value)
        

if __name__ == '__main__':
    main()