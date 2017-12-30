#-*- coding: UTF-8 -*- 
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle
import random
import numpy as np
# from IPython import embed

#######################
# global settings
#######################

check_pictures = False

modelpath = 'model/'
datatype = 'rotate_mellow_resize'

width, height, resultSpace = (24, 32, 36)

rounds = 3000
bigsize = 1000
datasize = bigsize * 3
batchsize = 50
start = random.randint(0, datasize // batchsize) * batchsize

feature1, feature2, feature3 = (32, 64, 128)
core1, core2, core3 = (5, 3, 3)

#######################
# load data
#######################

print('loading data...', end = '')
with open('data/train_package_%s_%d' % (datatype, bigsize), 'rb') as f:
    inputX = pickle.load(f)
with open('data/train_ans_%s_%d' % (datatype, bigsize), 'rb') as f:
    data = pickle.load(f)
    inputY = []
    for t in data:
        vector = [0] * 36
        vector[t] = 1
        inputY.append(vector)
print('complete')

#######################
# check the pictures
#######################

if check_pictures:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    for i in range(20):
        ii = random.randint(0, datasize)
        print(ii, end = ' ')
        for j, t in enumerate(inputY[ii]):
            if t == 1:
                if j < 10:
                    print(j)
                else:
                    print(chr(j + 55))
        plt.imshow(np.array(inputX[ii]).reshape(height, width))
        plt.show()
    exit()

#######################
# network
#######################

def weight_variable(shape, *, name):
    initial = tf.truncated_normal(shape, stddev = 0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial, name = name)

def bias_variable(shape, *, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
    几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, width * height])
x_image = tf.reshape(x, [-1,width,height,1]) #将输入按照 conv2d中input的格式来reshape，reshape

"""
# 第一层
# 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
# 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*28*28*32
# 也就是单个通道输出为28*28，共有32个通道,共有?个批次
# 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*14*14*32
# 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*20*30*32
"""
W_conv1 = weight_variable([core1, core1, 1, feature1], name = 'W_conv1')  
# 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv1 = bias_variable([feature1], name = 'b_conv1')
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
# 第二层
# 卷积核5*5，输入通道为32，输出通道为64。
# 卷积前图像的尺寸为 ?*14*14*32， 卷积后为?*14*14*64
# 池化后，输出的图像尺寸为?*7*7*64
# 池化后，输出的图像尺寸为?*10*15*64
"""
feature2 = 64
W_conv2 = weight_variable([core2, core2, feature1, feature2], name = 'W_conv2')
b_conv2 = bias_variable([feature2], name = 'b_conv2')
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

feature3 = 128
W_conv3 = weight_variable([core3, core3, feature2, feature3], name = "W_conv3")
b_conv3 = bias_variable([feature3], name = 'b_conv3')
h_conv3 = tf.nn.elu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
# h_pool4 = max_pool_2x2(h_pool3)

# W_conv4 = weight_variable([5, 5, 64, feature3])
# b_conv4 = bias_variable([feature3])
# h_conv4 = tf.nn.elu(conv2d(h_pool3, W_conv4) + b_conv4)
# h_pool4 = max_pool_2x2(h_conv4)

# 第三层 是个全连接层,输入维数7*7*64, 输出维数为output_dimension
output_dimension = 1024
W_fc1 = weight_variable([(width // 8) * (height // 8) * feature3, output_dimension], name = 'W_fc1')
b_fc1 = bias_variable([output_dimension], name = 'b_fc1')
h_pool2_flat = tf.reshape(h_pool3, [-1, (width // 8) * (height // 8) * feature3])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层，输入output_dimension维，输出resultSpace维，也就是具体的分类
W_fc2 = weight_variable([output_dimension, resultSpace], name = 'W_fc2')
b_fc2 = bias_variable([resultSpace], name = 'b_fc2')
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 使用softmax作为多分类激活函数
y_ = tf.placeholder(tf.float32, [None, resultSpace])

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # 损失函数，交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 使用adam优化
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) # 变量初始化

def getBatch(start, batchsize):
    tmpX = inputX[start: start + batchsize]
    tmpY = inputY[start: start + batchsize]
    # tmpY = [[0] * 36] * batchsize
    return tmpX, tmpY

########################
# train
########################

print('start training...')
for i in range(rounds):
    batch = getBatch(start, batchsize)
    start += batchsize
    if start >= datasize: 
        start -= datasize
    if (i + 1) % 20 == 0:
        train_accuracy = accuracy.eval(feed_dict = {
            x: batch[0], y_: batch[1], keep_prob: 1.0 })
        print("step %d, training accuracy %g" % (i + 1, train_accuracy))
    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5 })

saver = tf.train.Saver(tf.global_variables())
saver.save(sess, modelpath + 'three-layer-model')
print('Model saved at "%s"' % (modelpath + 'three-layer-model'))

########################
# test
########################

# saver = tf.train.Saver(tf.global_variables())
# module_file = tf.train.latest_checkpoint(modelpath)
# saver.restore(sess, module_file)
# print('Model restore from "%s"' % (modelpath + 'three-layer-model'))

cnt = np.zeros(36, dtype = np.int32)
cor = np.zeros((36, 36), dtype = np.int32)
batchsize = 1000
print('start testing...(total groups: %d)' % (bigsize // batchsize))
for grp in range(bigsize // batchsize):
    print('testing group %d' % (grp))
    testbatch = getBatch(datasize + grp * batchsize, batchsize)
    preVec = tf.argmax(y_conv, 1).eval(feed_dict = {
        x: testbatch[0], y_: testbatch[1], keep_prob: 1.0
        })
    ansVec = tf.argmax(y_, 1).eval(feed_dict = {
        x: testbatch[0], y_: testbatch[1], keep_prob: 1.0
        })
    for (i, ans) in enumerate(ansVec):
        cnt[ans] += 1
        cor[ans][preVec[i]] += 1

acc = [0, 0]
for i in range(36):
    acc[0] += cnt[i]
    acc[1] += cor[i][i]
print('accuracy: %f' % (acc[1] / acc[0]))
with open(modelpath + 'result(%f).csv' % (acc[1] / acc[0]), 'w') as file:
    file.write('accuracy: %f' % (acc[1] / acc[0]))
    for i in range(36):
        t = 55
        if i < 10: t = 48
        file.write(',%c' % (chr(i + t)))
    file.write('\n')
    for i in range(36):
        t = 55
        if i < 10: t = 48
        file.write(chr(i + t))
        for j in range(36):
            acu = float('nan')
            if cnt[i] > 0:
                acu = cor[i][j] / cnt[i]
            file.write(',%f' % (acu * 100))
        file.write('\n')

########################
# apply
########################     

# imageNumPer = 500
# with open('data/type2_test2_submit.csv', 'w') as file:
#     file.write('pic_id,content\n')
#     for i in range(1, 20001):
#         file.write('type1_test2_%d.jpg,11111\n' % (i))
#     for start in range(0, 20000, imageNumPer):
#         print('Processing image from %d to %d' % (start, start + imageNumPer))
#         testbatch = getBatch(start * 5, imageNumPer * 5)
#         ansVec = y_conv.eval(feed_dict = {
#             x: testbatch[0], y_:testbatch[1], keep_prob: 1.0
#             })
#         ans = []
#         for t in ansVec:
#             t = [(a, i) for i, a in enumerate(t)]
#             ans.append(max(t)[1])
#         ansstr = ''
#         for i, t in enumerate(ans):
#             if i % 5 == 0:
#                 ansstr += 'type2_test2_%d.jpg,' % (start + i // 5 + 1)
#             if t < 10:
#                 ansstr += str(t)
#             else:
#                 ansstr += chr(t + 55)
#             if i % 5 == 4:
#                 ansstr += '\n'
#         file.write(ansstr)
#     for i in range(3, 7):
#         for j in range(1, 20001):
#             file.write('type%d_test2_%d.jpg,11111\n' % (i, j))
