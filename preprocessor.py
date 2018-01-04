import os
import sys
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

path = 'data/train/'
prefix = ''
output_path = 'data/'
shared_dir = 'data/train_rotate_mellow_resize/'
datatype = 'rotate_mellow_resize'
workers = 4
data_len = 10000
start = 0
output_width = 20
output_height = 24


def rect_size(image):
    w, h = image.size
    img = np.array(image.getdata()).reshape(h, w)
    minx = w
    miny = h
    maxx = maxy = 0
    for y in range(h):
        for x in range(w):
            if img[y][x] == 255:
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)
    # image = image.copy()
    # rect = [(minx, miny), (maxx + 1, maxy + 1)]
    # ImageDraw.Draw(image).rectangle(rect, outline = 0)

    return (maxx + 1 - minx) * (maxy + 1 - miny)


def crop(image):
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

    box = (minx - 3, miny - 3, maxx + 3 + 1, maxy + 3 + 1)
    return image.crop(box).resize((output_width, output_height))


dx = [0, 0, -1, 1, -1, -1, 1, 1]
dy = [1, -1, 0, 0, -1, 1, -1, 1]
dx8 = [0, 1, 1, 1, 0, -1, -1, -1, 0]
dy8 = [1, 1, 0, -1, -1, -1, 0, 1, 1]


def mellow(image):
    w, h = image.size
    img = np.array(image.getdata()).reshape(h, w)
    nimg = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            cnt = 0
            flag = -1
            flagcnt = -1
            for k in range(9):
                tx = x + dx8[k]
                ty = y + dy8[k]
                if tx < 0 or ty < 0 or tx >= w or ty >= h:
                    continue
                if img[ty][tx] == 255 and k < 8:
                    cnt += 1
                if flag != img[ty][tx]:
                    flagcnt += 1
                    flag = img[ty][tx]

            if (cnt >= 4 and flagcnt <= 2) or (img[y][x] == 255 and cnt > 1):
                nimg[y][x] = 255

    return Image.fromarray(nimg, mode='L').convert('1')


def rotate(im):
    le = -36
    ri = 36
    while le + 1 < ri:
        mid1 = le + (ri - le + 1) // 3
        mid2 = ri - (ri - le + 1) // 3
        rs1 = rect_size(im.rotate(mid1, expand=False))
        rs2 = rect_size(im.rotate(mid2, expand=False))
        if rs1 < rs2:
            ri = mid2 - 1
        else:
            le = mid1 + 1
    rs1 = rect_size(im.rotate(le, expand=False))
    rs2 = rect_size(im.rotate(ri, expand=False))
    if rs1 > rs2:
        le = ri
    return im.rotate(le, expand=False)


def search_block(st, img, vis):
    h, w = img.shape
    q = [st]
    que = [st]
    vis[st[0]][st[1]] = 1
    while len(q) > 0:
        y, x = q.pop(0)
        for k in range(4):
            ty, tx = (y + dy[k], x + dx[k])
            if ty < 0 or tx < 0 or ty >= h or tx >= w:
                continue
            if vis[ty][tx] == 1 or img[ty][tx] == 0:
                continue
            vis[ty][tx] = 1
            q.append((ty, tx))
            que.append((ty, tx))
    if len(que) >= 8:
        return
    for y, x in que:
        img[y][x] = 0


def wipe_noise(image):
    w, h = image.size
    img = np.array(image.getdata(), dtype=np.uint8).reshape(h, w)
    vis = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if img[y][x] == 255 and vis[y][x] == 0:
                search_block((y, x), img, vis)

    return Image.fromarray(img, mode='L').convert('1')


def process_single(im):
    im = im.convert('1')
    im = Image.eval(im, lambda x: 255 - x)
    im = wipe_noise(im)
    # im = mellow(im)
    # im = rotate(im)
    im = mellow(im)
    im = crop(im)

    a = np.array(im.getdata(), dtype='uint8') // 255
    return a


def process_single_debug(im):
    im = im.convert('1')
    im = Image.eval(im, lambda x: 255 - x)
    plt.subplot(141)
    plt.imshow(np.array(im.getdata()).reshape(40, 32))
    im = wipe_noise(im)
    plt.subplot(142)
    plt.imshow(np.array(im.getdata()).reshape(40, 32))
    # im = mellow(im)
    # im = rotate(im)
    plt.subplot(143)
    plt.imshow(np.array(im.getdata()).reshape(40, 32))
    im = mellow(im)
    im = crop(im)
    plt.subplot(144)
    plt.imshow(np.array(im.getdata()).reshape(output_height, output_width))
    plt.show()

    a = np.array(im.getdata(), dtype='uint8') // 255
    return a


def process(pr):
    index, name = pr
    if index % 50 == 0:
        print(name)
    if os.path.exists(shared_dir + str(index)):
        return

    data = []
    with Image.open(name) as im:
        for j in range(0, 4):
            box = (32 * j, 0, 32 * (j + 1), 40)
            tmp = im.crop(box)
            data.append(process_single(tmp))
    with open(shared_dir + str(index), 'wb') as f:
        pickle.dump(data, f)


def process_debug(pr):
    index, name = pr
    print(name)

    with Image.open(name) as im:
        for j in range(0, 4):
            box = (32 * j, 0, 32 * (j + 1), 40)
            tmp = im.crop(box)
            process_single_debug(tmp)


def main():
    start_time = time.time()

    if not os.path.exists(shared_dir):
        os.mkdir(shared_dir)

    datalist = []
    for i in range(start, start + data_len):
        name = path + prefix + str(i) + '.jpg'
        datalist.append((i, name))

    if workers > 1:
        with Pool(workers) as p:
            p.map(process, datalist)
    else:
        for image in datalist:
            process(image)

    data = []
    for i in range(start, start + data_len):
        with open(shared_dir + str(i), 'rb') as f:
            tmp = pickle.load(f)
            for t in tmp:
                data.append(t)

    output_file = output_path + 'train_package_%s_%d' % (datatype, data_len)
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    # Process answer
    with open(path + 'ans.csv', 'r') as file:
        fileData = file.read().split('\n')[start: start + data_len]
        ansVector = []
        for s in fileData:
            ans = s.split(',')[1]
            for ch in ans:
                if ch.isdigit():
                    ansVector.append(ord(ch) - 48)
                else:
                    ansVector.append(ord(ch) - 55)

    output_file = output_path + 'train_ans_%s_%d' % (datatype, data_len)
    with open(output_file, 'wb') as f:
        pickle.dump(ansVector, f)

    print('Run for %.2f seconds' % (time.time() - start_time))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        process_debug((0, 'data/train/%s.jpg' % (sys.argv[1])))
    else:
        main()
