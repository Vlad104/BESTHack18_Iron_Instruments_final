import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import datetime
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

def show(kolvo):
    print('Kol-vo lines: {}'.format(kolvo))
    count = 0
    with open("Y.txt", 'r') as f:
        y_data = f.read()
    with open("nn_output.txt", 'r') as f:
        nn_output_data = f.read()
    for i in range(len(y_data)):
        try:
            if y_data[i] == nn_output_data[i] and y_data[i] != '.' and y_data[i] != ' ' and y_data[i] != '\n' and y_data [i]!= '[' and y_data[i] != ']': 
                count += 1
        except:
            print('File error . . .')
            break
    print('Kol-vo sovpadeniy: {}'.format(count))
    print('{} %'.format(round(count / kolvo * 100, 2)))
    return

def show1(kolvo):
    print('Kol-vo lines: {} (csv)'.format(kolvo))
    count = 0
    with open('Y.csv', 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        Y_read = [row for row in reader]
        #print(Y_read)
    with open('nn_output.csv', 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        nn_output_read = [row for row in reader]
        #print(nn_output_read)
    forbidden = ('[', ']', '.', ' ', "'", '\n', '')
    for i in range(0, len(nn_output_read), 2):
        if (Y_read[i] == nn_output_read[i]) and (Y_read[i] not in forbidden):
            count += 1
    print('Kol-vo sovpadeniy (csv): {}'.format(count))
    print('{} % (csv)'.format(round(count / kolvo * 100, 2)))
    return

def neiro(x, y, kolvo):
    ops.reset_default_graph()

    sess = tf.Session()

    x_ = tf.placeholder(name="input", shape=[None, kolvo], dtype=tf.float32)
    y_ = tf.placeholder(name="output", shape=[None, 1], dtype=tf.float32)

    hidden_neurons = 10
    hidden_neurons_2 = 2
    w1 = tf.Variable(tf.random_uniform(shape=[kolvo, hidden_neurons]))#, minval=0.001, maxval = 5.999))
    b1 = tf.Variable(tf.constant(value=0.0, shape=[hidden_neurons], dtype=tf.float32))
    layer1 = tf.nn.softplus(tf.add(tf.matmul(x_, w1), b1))
    
    w2 = tf.Variable(tf.random_uniform(shape=[hidden_neurons, hidden_neurons_2]))#, minval=0.001, maxval=5.999))
    b2 = tf.Variable(tf.constant(value=0.0, shape=[hidden_neurons_2], dtype=tf.float32))
    layer2 = tf.nn.softplus(tf.add(tf.matmul(layer1, w2), b2))

    w3 = tf.Variable(tf.random_uniform(shape=[hidden_neurons_2, 1]))#, minval=0.001, maxval=5.999))
    b3 = tf.Variable(tf.constant(value=0.0, shape=[1], dtype=tf.float32))

    nn_output = tf.nn.softplus(tf.add(tf.matmul(layer2, w3), b3))
    gd = tf.train.GradientDescentOptimizer(0.0000001)
    loss = tf.reduce_mean(tf.square(nn_output - y_))
    train_step = gd.minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(100000):
        sess.run(train_step, feed_dict={x_: x, y_: y}) 

    with open('nn_output.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows((np.around(sess.run(nn_output, feed_dict={x_: x}) * 6)))
    with open('w1.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(sess.run(w1))    
    with open('w2.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(sess.run(w2))
    with open('w3.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(sess.run(w3))
    with open('b1.txt', 'w') as f:
        np.savetxt('b1.txt', sess.run(b1))   
    with open('b2.txt', 'w') as f:
        np.savetxt('b2.txt', sess.run(b2)) 
    with open('b3.txt', 'w') as f:
        np.savetxt('b3.txt', sess.run(b3)) 

    with open("nn_output.txt", 'w') as f:
        f.write(str(np.around(sess.run(nn_output, feed_dict={x_: x}) * 6)))
    #print(np.array(sess.run(nn_output, feed_dict={x_: x}) * 6))

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
            #print(znach)
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
    time_1 = datetime.datetime.now()
    filename = 'train.csv'
    #filename2 = 'test.csv'
    df = pd.read_csv(filename, low_memory=False) 
    #df2 = pd.read_csv(filename2, low_memory=False) 
    count = 50000
    test_value = 62
    database = get_data(df, 'xy', count)
    #database2 = get_data(df2, 'x', count)
    #ID = get_id(df2, count)
    X1 = user_db(database, 'x', count, test_value)
    #X2 = user_db(database2, 'x', count, test_value)
    Y1 = user_db(database, 'y', count, 1)
    with open('Y.txt', 'w') as f:
        f.write(str(Y1 * 6))
    with open('Y.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(Y1 * 6)
    with open('X.txt', 'w') as f:
        for i in range(count):
            f.writelines(str(X1[i]))
    #print()
    #print(X1)
    print('\nYour arrays: X({}, {}) and Y({}, 1) are ready!\n'.format(count, test_value, count))
    #print(Y1)
    neiro(X1, Y1, test_value)
    print()
    #print(Y1)
    show(count)
    print()
    show1(count)
    time_2 = datetime.datetime.now()
    print()
    print(time_1)
    print(time_2)
    #print(nump_arr_w1)
    #print(nump_arr_w1[0][0])
    #with open('ID.csv', 'w') as fp:
        #writer = csv.writer(fp, delimiter=',')
        #writer.writerows(data_array)

if __name__ == '__main__':
    main()
