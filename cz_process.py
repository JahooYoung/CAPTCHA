import os
from PIL import Image, ImageDraw
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
# import type2_preprocessor_v1 as prep

path = 'data/train/'
prefix = ''
output_path = 'data/'
workers = 1
data_len = 1000
start = 0
data = [0] * (data_len * 4)

def rect_size(image):
    img = np.array(image.getdata()).reshape(40, 32)
    minx = 32
    miny = 40
    maxx = maxy = 0
    for y in range(40):
        for x in range(32):
            if img[y][x] == 255:
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)
    # image = image.copy()
    # ImageDraw.Draw(image).rectangle([(minx, miny), (maxx + 1, maxy + 1)], outline = 0)
    
    return (maxx + 1 - minx) * (maxy + 1 - miny)

def fan(a):
    return 255 - a

dx = [0, 0, -1, 1, -1, -1, 1, 1]
dy = [1, -1, 0, 0, -1, 1, -1, 1]
def mellow(image):
    img = np.array(image.getdata()).reshape(40, 32)
    nimg = np.zeros((40, 32), dtype = np.uint8)
    for y in range(40):
        for x in range(32):
            if img[y][x] == 255:
                nimg[y][x] = 255
                continue
            cnt = 0
            for k in range(8):
                tx = x + dx[k]
                ty = y + dy[k]
                if tx < 0 or ty < 0 or tx >= 32 or ty >= 40:
                    continue
                if img[ty][tx] == 255:
                    cnt += 1
            if cnt >= 4:
                nimg[y][x] = 255
    # print(list(img))
    # print(list(nimg))
    return Image.fromarray(nimg, mode = 'L').convert('1')

# encode data
def encode_data(im):
    im = im.convert('1')
    im = Image.eval(im, fan)
    # im.show()
    l, r = (-36, 36)
    while l + 1 < r:
        mid1 = l + (r - l + 1) // 3
        mid2 = r - (r - l + 1) // 3
        rs1 = rect_size(im.rotate(mid1, expand = False))
        rs2 = rect_size(im.rotate(mid2, expand = False))
        # print(mid1, rs1, mid2, rs2)
        if rs1 < rs2:
            r = mid2
        else:
            l = mid1
    if rect_size(im.copy().rotate(l, expand = False)) > rect_size(im.copy().rotate(r, expand = False)):
        l = r
    # print(l)
    im = im.rotate(l, expand = False)
    # plt.subplot(121)
    # plt.imshow(np.array(im.getdata()).reshape(40, 32))
    # im = mellow(im)
    # plt.subplot(122)
    # plt.imshow(np.array(im.getdata()).reshape(40, 32))
    # plt.show()
    # print(np.array(im.getdata()))
    a = list(im.getdata())
    a = [int(i/255) for i in a]
    # print(a)
    return a

#cropdata
cnt = 1
def cz_process(pr):
    # global cnt
    # if cnt % 100 == 0:
    #     print(cnt)
    # cnt += 1
    # print(pr)
    index, name = pr
    print(index, name)
    # print(data)
    with Image.open(name) as im:
        for j in range(0, 4):
            box = (32 * j, 0, 32 * (j + 1), 40)
            tmp = im.crop(box)
            data[index * 4 + j] = encode_data(tmp)

# cz_process((0, 'data/train/100.jpg'))
# exit()

def main():
    start_time = time.time()

    datalist = []
    for i in range(start, start + data_len):
    #     type2_train_1.jpg
        name = path + prefix + str(i) + '.jpg'
        datalist.append((i, name))

    if workers > 1:
        Pool(workers).map(cz_process, datalist)
    else:
        for image in datalist:
            cz_process(image)

    with open(output_path + 'train_package_rotate_%d' % (data_len), 'wb') as f:
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
    with open(output_path + 'train_ans_rotate_%d' % (data_len), 'wb') as f:
        pickle.dump(ansVector, f)

    print('Run for %.2f seconds' % (time.time() - start_time))

if __name__ == '__main__':
    main()