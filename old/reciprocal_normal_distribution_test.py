from pylab import *

from scipy.stats import *
import numpy as np

# norm(exp, sq_div)
a_norm = norm(24.488, 3.81)


array_r = np.zeros(100000)
for r in range(len(array_r)):
    array_r[r] = 1 / a_norm.rvs()

boxes = np.linspace(array_r.min(), array_r.max(), 101)
counts = np.zeros(len(boxes)-1)
for left_edge in range(len(boxes) - 1):
    for element in range(len(array_r)):
        if boxes[left_edge] < array_r[element] <= boxes[left_edge + 1]:
            counts[left_edge] += 1
max_spot = counts.argmax()




figure(figsize=(10, 6), dpi=100)
subplot(1, 1, 1)
plot([1/24.488, 1/24.488], [0, counts.max()*1.1], color='k', linestyle='--', linewidth=2.0, label='1/mean')
plot([boxes[max_spot], boxes[max_spot]], [0, counts.max()*1.1], color='k', linestyle=':', linewidth=2.0, label='peak')

annotate(r'$\frac{1}{24.488}=0.03510$',
         xy=(1/24.488, counts.max()*1.05), xycoords='data',
         xytext=(+60, 0), textcoords='offset points', fontsize=18,
         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
annotate(r'$peak=$'+str(round(boxes[max_spot], 5)),
         xy=(boxes[max_spot], counts.max()*1.05), xycoords='data',
         xytext=(-60, +20), textcoords='offset points', fontsize=18,
         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))

legend(loc='upper right')
xlabel('h')

plot(boxes[:-1], counts, color='k', linewidth=2.0)
show()


