import cv2
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue

isDebug = 0

def getClothesHSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist( [hsv], [1], None, [255], [0, 255])
    hist_v = cv2.calcHist( [hsv], [2], None, [255], [0, 255])

    cv2.normalize(hist, hist, 0, 180, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, 0, 255, cv2.NORM_MINMAX)

    # H
    max = -1
    h = 0
    for i in range(1, 180 - 1):
        if hist[i] > max:
            max = hist[i]
            h = i
    #print("h:", h)

    # S
    max = -1
    s = 0
    for i in range(1, 255 - 1):
        if hist_s[i] > max:
            max = hist_s[i]
            s = i
    #print("s:", s)

    # V
    max = -1
    v = 0
    for i in range(1, 255 - 1):
        if hist_v[i] > max:
            max = hist_v[i]
            v = i
    #print("v:", v)

    return [h, s, v]

# def getClothesColor(img):
#     color = "默认"
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist( [hsv], [0], None, [180], [0, 180])
#     hist_s = cv2.calcHist( [hsv], [1], None, [255], [0, 255])
#     hist_v = cv2.calcHist( [hsv], [2], None, [255], [0, 255])
#
#     cv2.normalize(hist, hist, 0, 180, cv2.NORM_MINMAX)
#     cv2.normalize(hist_s, hist_s, 0, 255, cv2.NORM_MINMAX)
#     cv2.normalize(hist_v, hist_v, 0, 255, cv2.NORM_MINMAX)
#
#     # H
#     max = -1
#     h = 0
#     for i in range(1, 180 - 1):
#         if hist[i] > max:
#             max = hist[i]
#             h = i
#     #print("h:", h)
#
#     # S
#     max = -1
#     s = 0
#     for i in range(1, 255 - 1):
#         if hist_s[i] > max:
#             max = hist_s[i]
#             s = i
#     #print("s:", s)
#
#     # V
#     max = -1
#     v = 0
#     for i in range(1, 255 - 1):
#         if hist_v[i] > max:
#             max = hist_v[i]
#             v = i
#     #print("v:", v)
#
#     if v >= 0 and v < 47:
#         color = "black"
#     elif s >= 0 and s < 43:
#         if v >= 47 and v < 217:
#             color = "grey"
#         else:
#             color = "white"
#
#     else:
#         if h > 0 and h < 6:
#             color = "scarlet"
#         elif h >= 6 and h < 11:
#             color = "red"
#         elif h >= 11 and h < 15:
#             color = "brown"
#         elif h >= 15 and h < 18:
#             color = "orange"
#         elif h >= 18 and h < 26:
#             color = "camel"
#         elif h >= 26 and h < 29:
#             color = "khaki"
#         elif h >= 29 and h < 32:
#             color = "beige"
#         elif h >= 32 and h < 36:
#             color = "yellow"
#         elif h >= 36 and h < 41:
#             color = "olive"
#         elif h >= 41 and h < 78:
#             color = "green"
#         elif h >= 78 and h < 84:
#             color = "turquoise"
#         elif h >= 84 and h < 100:
#             color = "cyan"
#         elif h >= 100 and h < 120:
#             color = "blue"
#         elif h >= 120 and h < 125:
#             color = "lilac"
#         elif h >= 125 and h < 138:
#             color = "lavender"
#         elif h >= 138 and h < 144:
#             color = "pansy"
#         elif h >= 144 and h < 156:
#             color = "mauve"
#         elif h >= 156 and h < 170:
#             color = "pink"
#         elif h >= 170 and h < 181:
#             color = "maroon"
#         else:
#             color = "red"
#     return color




# 改良后的算法

def resizeImg(img, ratio):
    shape = img.shape
    height = shape[0]
    width = shape[1]
    size = (int(width * ratio), int(height * ratio))
    shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return shrink


# new way to tell the color
colorWhiteMin = 253
colorWhiteMax = 255

colors = {
        'scarlet': [[0.0, 1.0, 1.0], [0.0000, 1.0000, 0.9333]],
        'red': [[0.0000, 1.0000, 0.8039], [0.0000, 0.5512, 0.8039], ],
        'brown': [[0.0, 0.745, 0.647], [0.0, 0.7482, 0.5451]],
        'orange': [[0.1078, 1.0, 1.0], [0.1078, 1.0, 0.9333]],
        'camel': [[0.0931, 0.1429, 0.9333], [0.0926, 0.1412, 1.0000]],
        'khaki': [[0.1500, 0.4167, 0.9412], [0.1533, 0.4392, 1.0], [0.1540, 0.4412, 0.9333]],
        'beige': [[0.1667, 0.1020, 0.9608]],
        'yellow': [[0.1667, 1.0, 1.0], [0.1667, 1.0, 0.9333], [0.1667, 1.0, 0.8039]],
        'olive': [[0.2284, 0.5608, 1.0], [0.2289, 0.5630, 0.9333], [0.2211, 0.7569, 1.0], [0.2213, 0.7563, 0.9333]],
        'green': [[0.3333, 1.0000, 1.0000], [0.3333, 1.0000, 0.9333], [0.3333, 1.0000, 0.8039],
                  [0.3333, 1.0000, 0.5451]],
        'turquoise': [[0.5024, 1.0000, 0.8196], [0.4939, 0.6555, 0.8196], [0.4833, 0.7143, 0.8784],
                      [0.5065, 1.0000, 1.0000], [0.5063, 1.0000, 0.9333]],
        'cyan': [[0.5000, 1.0000, 0.8039], [0.5000, 1.0000, 0.5451]],
        'blue': [[0.6667, 1.0000, 1.0000], [0.6667, 1.0000, 0.9333], [0.5418, 1.0000, 1.0000]],
        'lavender': [[0.6667, 0.0800, 0.9804], ],
        'purple': [[0.7692, 0.8667, 0.9412], [0.7528, 0.8118, 1.0000], [0.7534, 0.8151, 0.9333],
                   [0.7212, 0.4886, 0.8588], [0.7535, 0.8146, 0.8039]],
        'thistle': [[0.8333, 0.1157, 0.8471], [0.8333, 0.1176, 1.0000], [0.8333, 0.1176, 0.9333]],
        'pink': [[0.9709, 0.2471, 1.0000], [0.9640, 0.2902, 1.0000], [0.9638, 0.2899, 0.9333], [0.9099, 0.9216, 1.0000],
                 [0.9098, 0.9244, 0.9333], [0.9167, 0.5882, 1.0000], [0.9774, 0.3176, 1.0000]],
        'maroon': [[0.9375, 0.7273, 0.6902], [0.8949, 0.7986, 0.5451], [0.8953, 0.8000, 0.8039]]
}

colorsRange = {
        'scarlet': [0, 6],
        'red': [6, 11],
        'brown': [11, 15],
        'orange': [15, 20],
        # 'camel': [18, 26],
        # 'khaki': [26, 29],
        # 'beige': [29, 32],
        'yellow': [20, 36],
        'olive': [36, 41],
        'green': [41, 78],
        'turquoise': [78, 84],
        'cyan': [84, 100],
        'blue': [100, 120],
        'lavender': [120, 138],
        'purple': [138, 156],
        'pink': [156, 170],
        'maroon': [170, 181]
}


def normalizeHSV(hsv):
    t = [0.0, 0.0, 0.0]
    divArg = [180, 255, 255]
    for i in range(0, 3):
        t[i] = float(float(hsv[i]) / divArg[i])
    return t


# 计算颜色距离的方法效果不是很好 最终还是根据颜色区间+个别颜色校正来选取颜色
def calculateColorDistance(t):

    # 根据错误的图片识别样例 做出的颜色校正
    if t[0] >= 0 and t[1] >= 0.08 and t[2] >= 0.8 \
            and t[0] <= 0.006 and t[1] <= 0.14 and t[2] <= 1:
        return 'pink'
    if t[0] >= 0.96 and t[1] >= 0.08 and t[2] >= 0.8 \
            and t[0] <= 1 and t[1] <= 0.14 and t[2] <= 1:
        return 'pink'

    if t[0] >= 0.054 and t[1] >= 0.46 and t[2] >= 0.63 \
            and t[0] <= 0.062 and t[1] <= 0.50 and t[2] <= 0.68:
        return 'brown'

    if t[0] >= 0.57 and t[1] >= 0.12 and t[2] >= 0.74 \
            and t[0] <= 0.585 and t[1] <= 0.167 and t[2] <= 0.83:
        return 'blue'

    if t[0] >= 0.83 and t[1] >= 0.05 and t[2] >= 0.8 \
            and t[0] <= 0.9 and t[1] <= 0.07 and t[2] <= 0.9:
        return 'lavender'

    if t[2] <= (46 / 255):
        return 'black'

    if t[1] <= (43 / 255):
        if t[2] >= (220 / 255):
            return 'white'
        else:
            return 'gray'

    color = 'scarlet'
    # 通过颜色区间计算出像素点归属于哪个颜色
    for k in colorsRange:
        if t[0] * 180 >= colorsRange[k][0] and t[0] * 180 < colorsRange[k][1]:
            color = k

    # 颜色校正
    if color == 'scarlet' and t[1] <= 0.7:
        color = 'lightRed'

    if color == 'orange' and t[1] <= 0.6:
        color = 'camel'

    if color == 'yellow' and t[0] < 0.159 and t[1] <= 0.5 and t[2] <= 0.85:
        color = 'khaki'

    if color == 'yellow' and t[1] <= 0.27 and t[2] > 0.85:
        color = 'beige'

    return color

def getColor(hsv, i, j):
    if isDebug == 1:
        print(hsv[i][j])
    t = normalizeHSV(hsv[i][j])

    if isDebug == 1:
        print(t)
    color = calculateColorDistance(t)
    return color

def isWhite(color):
    for i in range(0, 3):
        if color[i] >= colorWhiteMin and color[i] <= colorWhiteMax:
            return True
    return False


# 填充白色背景 标记为 visited=1 被访问
offsetX = [0, 1, 0, -1]
offsetY = [1, 0, -1, 0]
def fillColor(img, width, height, visited, start):
    q = Queue()
    q.put([0, 0])
    q.put([height - 1, 0])
    q.put([0, width - 1])
    q.put([height - 1, width - 1])
    q.put([(height - 1), int((width - 1) / 2)])
    q.put([0, int((width - 1) / 2)])
    while not q.empty():
        i, j = q.get()
        if isWhite(img[i][j]):
            img[i][j] = [255, 0, 0]
        else:
            continue
        # print([i, j])
        m = i
        n = j
        for k in range(0, 4):
            i = m + offsetX[k]
            j = n + offsetY[k]
            if i >= 0 and j >= 0 and i < height and j < width and [i, j] not in visited:
                q.put([i, j])
                visited.append([i, j])

    return visited


# 获取mainColor及其HSV
def getClothesColor2(img):
    colorsNum = {
        'black': 0,
        'white': 0,
        'gray': 0,
        'scarlet': 0,
        'red': 0,
        'brown': 0,
        'orange': 0,
        'camel': 0,
        'khaki': 0,
        'beige': 0,
        'yellow': 0,
        'olive': 0,
        'green': 0,
        'turquoise': 0,
        'cyan': 0,
        'blue': 0,
        'lavender': 0,
        'purple': 0,
        'thistle': 0,
        'pink': 0,
        'maroon': 0,
        'lightRed':0
    }

    shape = img.shape
    height = shape[0]
    width = shape[1]

    visited = []
    visited = fillColor(img, width, height, visited, [0, 0])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    count = [0] * 10
    for i in range(0, height):
        for j in range(0, width):
            if [i, j] not in visited:
                c = getColor(hsv, i, j)
                if isDebug == 1:
                    print(c)
                colorsNum[c] += 1
    max = -1
    for k in colorsNum:
        if colorsNum[k] > max:
            max = colorsNum[k]
            mainColor = k
    # print('mainColor: ' + mainColor)

    mainColorHSV = [0.0, 0.0, 0.0]

    for i in range(0, height):
        for j in range(0, width):
            if [i, j] not in visited:
                c = getColor(hsv, i, j)
                if c == mainColor:
                    mainColorHSV = hsv[i][j]
                    # mainColorHSV = normalizeHSV(hsv[i][j])
                    break

    return [mainColor, mainColorHSV]
