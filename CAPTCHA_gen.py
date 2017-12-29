from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

import random

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import math
import sys

Dir = './CAPTCHA/'

# 随机字母:
alphabet = []
for i in range(10):
    alphabet.append(chr(48 + i))
for i in range(26):
    if chr(65 + i) != 'I' and chr(65 + i) != 'Q':
        alphabet.append(chr(65 + i))
def rndChar():
    return alphabet[random.randint(0, len(alphabet) - 1)]

# 随机颜色1:
def rndColor(alpha = None):
    if alpha == None:
        return (0, 0, 0)
    else:
        return (0, 0, 0, alpha)
    # return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

# 随机颜色2:
def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

def rndColor3():
    return (random.randint(48, 191), random.randint(48, 191), random.randint(48, 191))

def generate():
    # width x height:
    width = 60 * 2
    height = 50
    image = Image.new('RGBA', (width, height), (255, 255, 255, 127))
    # 创建Font对象:
    font = ImageFont.truetype('C:\Windows\Fonts\Arial.ttf', 36)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)
    # 填充每个像素:
    # for x in range(width):
    #     for y in range(height):
    #         draw.point((x, y), fill=rndColor())
    # 输出文字:
    xlist = [0, 30, 60, 90]
    # while len(xlist) < 4:
    #     t = random.randint(width // 12, width // 6 * 4)
    #     mindis = width
    #     for i in xlist:
    #         mindis = min(mindis, abs(t - i))
    #     if len(xlist) == 0 or (mindis >= 16 and mindis <= 19):
    #         xlist.append(t)
    # xlist.sort()
    ans = ''
    for t in range(4):
        char = rndChar()
        ans += char
        tempImage = Image.new('RGBA', (45, 40), (255, 255, 255, 0))
        ImageDraw.Draw(tempImage).text((0, 0), char, font=font, fill=rndColor())
        tempImage = tempImage.rotate(random.randint(-35, 35), expand = False)
        tw, th = tempImage.size
        tw = random.randint(round(0.7 * tw), round(1.0 * tw))
        th = random.randint(round(0.7 * th), round(1.0 * th))
        tempImage = tempImage.resize((tw, th))
        x = xlist[t]
        y = random.randint(2, max(height - tempImage.size[1], 2))
        # print(x, y, char)
        image.paste(tempImage, (x, y), tempImage)
    # 干扰线
    # for t in range(2):
    #     dots = []
    #     for i in range(random.randint(2, 2)):
    #         dots.append((random.randint(width // 6, width - 1), random.randint(0, height - 1)))
    #     draw.line(dots, fill = rndColor(), width = 2)
    # 噪点
    # draw = ImageDraw.Draw(image)
    # for t in range(500):
    #     x = random.randint(0, width - 1)
    #     y = random.randint(0, height - 1)
    #     draw.point((x, y), fill = rndColor())
    # # 模糊:
    # image = image.filter(ImageFilter.GaussianBlur(2.3))
    # image = image.filter(ImageFilter.RankFilter(5, 19))
    # image = image.filter(ImageFilter.UnsharpMask)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)

    image.save(Dir + ans + '.png', 'png')
    # image.save('code.png', 'png')

import shutil, os

if not os.path.exists(Dir):
    os.mkdir(Dir)
for file in os.listdir(Dir):
    os.remove(Dir + file)
for i in range(24):
    generate()

######################
# Processor
######################

# for rep in range(3):
#     # if rep > 0:
#         # image = image.filter(ImageFilter.BoxBlur(0.1))
#     draw = ImageDraw.Draw(image)
#     imageList = list(image.getdata())
#     nsumarray = np.array([t[0] for t in imageList], dtype = np.int64)
#     nsumarray = nsumarray + np.array([t[1] for t in imageList])
#     nsumarray = nsumarray + np.array([t[2] for t in imageList])
#     mean = nsumarray.sum() / (width * height)
#     n2 = nsumarray * nsumarray
#     var = (n2).sum() / (width * height) - mean ** 2
#     var = math.sqrt(var)
#     print(mean, var)
#     for i, (r, g, b, a) in enumerate(imageList):
#         if r + g + b <= mean - 0.65 * var:
#             color = (0, 0, 0, 255)
#         else:
#             color = (255, 255, 255, 255)
#         draw.point((i % width, i // width), fill = color)
# image.save('code1.png', 'png')

# plt.subplot(211)
# lena = mpimg.imread('code.png') # 读取和代码处于同一目录下的 lena.png
# # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
# # lena.shape #(512, 512, 3)
# plt.imshow(lena) # 显示图片
# # plt.axis('off') # 不显示坐标轴
# plt.subplot(212)
# lena = mpimg.imread('code1.png') # 读取和代码处于同一目录下的 lena.png
# # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
# # lena.shape #(512, 512, 3)
# plt.imshow(lena) # 显示图片
# # plt.axis('off') # 不显示坐标轴
# plt.show()