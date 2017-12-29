import os
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
# import type2_preprocessor_v1 as prep

cnt = 1
# encode data
def encode_data(im):
    im = im.convert('1')
    # im.show()
    # plt.imshow(im.getdata())
    # plt.show()
    a = list(im.getdata())
    a = [int(i/255) for i in a]
    # print(a)
    data.append(a)

#cropdata
def cropdata(name):
    global cnt
    if cnt % 100 == 0:
        print(cnt)
    cnt += 1
    with Image.open(name) as im:
        # im = prep.Process(im) # find the cycles
        for j in range(0, 4):
#                 (10,29)(45,66)
            # box = (30*j,0,30*j+40,60)
            # box = (32 + 30 * j, 9, 68 + 30 * j, 45) # 36 * 36
            box = (32 * j, 0, 32 * (j + 1), 40)
            tmp = im.crop(box)
            #tmp.show()
            encode_data(tmp)

data = []
def cz_process(image_name):
    cropdata(image_name)

def main():
    # path = 'big_sample/bigdata/'
    path = 'data/train/'
    # path = 'data/type2_train_pre_v1/'
    prefix = ''
    # croppath = 'big_sample/yjh_bigdata/'
    output_path = 'data/'
    workers = 1
    data_len = 10000
    start = 0

    datalist = []
    for i in range(start, start + data_len):
    #     type2_train_1.jpg
        name = path + prefix + str(i) + '.jpg'
        datalist.append(name)

    if workers > 1:
        Pool(workers).map(cz_process, datalist)
    else:
        for image in datalist:
            cz_process(image)

    with open(output_path + 'train_package_%d' % (data_len), 'wb') as f:
        pickle.dump(data, f)

    # Process answer
    with open(path + 'ans.csv', 'r') as file:
        fileData = file.read().split('\n')[start : start + data_len]
        ansVector = []
        for s in fileData:
            ans = s.split(',')[1]
            for ch in ans:
                if ch.isdigit():
                    ansVector.append(ord(ch) - 48)
                else:
                    ansVector.append(ord(ch) - 55)
    with open(output_path + 'train_ans_%d' % (data_len), 'wb') as f:
        pickle.dump(ansVector, f)

start_time = time.time()
main()
print('Run for %.2f seconds' % (time.time() - start_time))