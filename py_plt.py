# coding = utf-8

import numpy as np
import cv2

import matplotlib.pyplot as plt
from pylab import *


x = [0, 10, 20, 40, 80, 120, 160, 200]
y = [0, 0.7843, 0.8333, 0.8627, 0.9020, 0.9216, 0.9608, 0.9902]

x1 = [0, 10, 20, 40, 80, 120, 160, 200]
y1 = [0, 0.1863, 0.1863, 0.1765, 0.1765, 0.1569, 0.1471, 0.1176]

plt.figure(1)
plt.subplot(211)
plt.xlabel('Train Sample Number')
plt.ylabel('Probability')
#添加标题
plt.title('TE')

plt.text(10, 0.70, r'Accuracy = 78.43%')
plt.text(40, 0.80, r'Accuracy = 86.27%')
plt.text(80, 0.85, r'Accuracy = 90.20%')
plt.text(120, 0.90, r'Accuracy = 92.16%')
plt.text(160, 0.90, r'Accuracy = 96.08%')
plt.text(190, 0.99, r'Accuracy = 99.02%')

plt.text(10, 0.18, r'Manual Identify Rate = 18.63%')
plt.text(40, 0.17, r'Manual Identify Rate = 17.65%')
plt.text(80, 0.17, r'Manual Identify Rate = 17.65%')
plt.text(120, 0.15, r'Manual Identify Rate = 15.69%')
plt.text(160, 0.14, r'Manual Identify Rate = 14.71%')
plt.text(190, 0.11, r'Manual Identify Rate = 11.76%')

plt.axis([0, 200, 0, 1.0])

plt.plot(x, y, 'bs')
plt.plot(x, y, 'r--')

plt.plot(x1, y1, 'g^')
plt.plot(x1, y1, 'r--')
plt.show()