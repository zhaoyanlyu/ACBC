from pylab import *
import pylab

from scipy.stats import *
import numpy as np
import math
"""
a_norm = norm(10, 3)
array_r = np.zeros(100000)
for r in range(len(array_r)):
    array_r[r] = 1 / norm.rvs(10)

boxes = np.linspace(array_r.min(), array_r.max(), 101)
counts = np.zeros(len(boxes)-1)
for left_edge in range(len(boxes) - 1):
    for element in range(len(array_r)):
        if boxes[left_edge] < array_r[element] <= boxes[left_edge + 1]:
            counts[left_edge] += 1
for i in range(len(counts)):
    counts[i] /= len(array_r)
plot(boxes[:-1], counts, color='blue', linewidth=1.0)
"""

mean = 24.488
var = 10

figure(figsize=(10, 6), dpi=100)
subplot(1, 1, 1)
X = linspace(0.001, 0.08, 1000)

Y = np.zeros(1000)
for i in range(len(Y)):
    Y[i] = (math.e ** (- (((1/X[i]) - mean)**2) / (2 * (var**2)))) / (((2 * math.pi) ** (1/2)) * (X[i]**2) * var)
max_spot = Y.argmax()


plot([1/mean, 1/mean], [0, Y.max()*1.2], color='k', linestyle='--', linewidth=2.0, label='1/mean')
plot([X[max_spot], X[max_spot]], [0, Y.max()*1.2], color='k', linestyle=':', linewidth=2.0, label='peak')

plot(X, Y, color='k', linewidth=2.0)

print(X[max_spot])

annotate(r'$\frac{1}{24.488}=0.03510$',
         xy=(1/mean, Y.max()*1.05), xycoords='data',
         xytext=(+60, +30), textcoords='offset points', fontsize=18,
         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
annotate(r'$peak=0.03445$',
         xy=(X[max_spot], Y.max()*1.05), xycoords='data',
         xytext=(-140, +30), textcoords='offset points', fontsize=18,
         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
legend(loc='upper right')
xlabel('h')
show()
