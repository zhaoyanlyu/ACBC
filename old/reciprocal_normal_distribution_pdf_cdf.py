from pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from scipy.stats import *
import numpy as np

mean = 24.488
var = 3.8


X = linspace(0.001, 0.08, 1000)

pdf = np.zeros(1000)
for i in range(len(pdf)):
    pdf[i] = (math.e ** (- (((1/X[i]) - mean)**2) / (2 * (var**2)))) / (((2 * math.pi) ** (1/2)) * (X[i]**2) * var)
max_spot = pdf.argmax()

cdf = np.zeros(1000)

print(pdf)

for i in range(1, len(pdf)):
    cdf[i] = cdf[i-1] + pdf[i]*((0.08-0.001)/1000)
print(cdf)

host = host_subplot(111, axes_class=AA.Axes)
par2 = host.twinx()
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right", axes=par2)

p1 = host.plot(X, pdf, color='k', linewidth=2.0, linestyle='-')
p2 = par2.plot(X, cdf, color='k', linewidth=2.0, linestyle='--')
xlabel('h')
show()