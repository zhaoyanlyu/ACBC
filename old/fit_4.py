from pylab import *
from scipy.stats import *
import numpy as np
import scipy.integrate as integrate

mean = 24.488
var = 3.81
test_parameter = 10
begin = 0.001
end = 0.08
resolution = 1000

X = linspace(begin, end, resolution)
# pdf of reciprocal normal distribution
pdf = np.zeros(resolution)
for i in range(len(pdf)):
    pdf[i] = (math.e ** (- (((1/X[i]) - mean)**2) / (2 * (var**2)))) / (((2 * math.pi) ** (1/2)) * (X[i]**2) * var)
max_spot = pdf.argmax()


def f1(x):
    return ((math.e**(- (((1/x) - mean)**2) / (2 * (var**2)))) / (((2 * math.pi)**(1/2)) * (x**2) * var))\
           * x
result_mean, abs_err_1 = integrate.quad(f1, -inf, +inf)

result_var = 1 / (pdf.max() * (2*math.pi) ** (1/2))

a_norm = norm(result_mean, result_var)
a_norm_pdf = a_norm.pdf(X)

# error calculations

# total error rate calculation
error_area = 0
for x_spot in range(resolution):
    error_area += abs(a_norm_pdf[x_spot] - pdf[x_spot]) * (abs(begin-end) / resolution)
print("error area = "+str(error_area))

# 95% of values error rate calculation
error_area_95 = 0
left_edge = 0
right_edge = resolution - 1
for x_spot in range(resolution):
    if X[x_spot] < result_mean-2*result_var <= X[x_spot+1]:
        left_edge = x_spot
        break
for x_spot in range(resolution):
    if X[x_spot] < result_mean+2*result_var <= X[x_spot+1]:
        right_edge = x_spot
        break
print("left edge = "+str(left_edge)+"; right edge"+str(right_edge))
for x_spot in range(left_edge, right_edge):
    error_area_95 += abs(a_norm_pdf[x_spot] - pdf[x_spot]) * (abs(begin-end) / resolution)
print("error area 95% = "+str(error_area_95))

# average quadratic error calculation
average_quadratic_error = 0
for x_spot in range(resolution):
    average_quadratic_error += (a_norm_pdf[x_spot] - pdf[x_spot])**2
average_quadratic_error /= resolution

print("average quadratic error"+str(average_quadratic_error))

# plot the figure
figure(figsize=(10, 6), dpi=100)
subplot(1, 1, 1)
plot([X[left_edge], X[left_edge]], [0, pdf.max()*1.1], color='k', linestyle='--')
plot([X[right_edge], X[right_edge]], [0, pdf.max()*1.1], color='k', linestyle='--')
plot(X, pdf, color='k', linewidth=2.0, label='original')
plot(X, a_norm_pdf, color='k', linewidth=2.0, label='fit', linestyle='--')
annotate("", xy=(X[left_edge], pdf.max()*1.05),
         xytext=(X[right_edge], pdf.max()*1.05),
         arrowprops=dict(arrowstyle='<->', connectionstyle='arc3, rad=.3'))
annotate(str(round(X[left_edge], 6)), xy=(X[left_edge], 0),
         xytext=(X[left_edge], -10),
         arrowprops=dict(arrowstyle='-', connectionstyle='arc3'))
annotate(str(round(X[right_edge], 6)), xy=(X[right_edge], 0),
         xytext=(X[right_edge], -10),
         arrowprops=dict(arrowstyle='-', connectionstyle='arc3'))
text(0.04, 80, '95%')
legend(loc='best')
fill_between(X, a_norm_pdf, pdf, facecolor='k', alpha=0.2)
text(0.055, 80, 'error rate='+str(round(error_area*100, 1))+'%')
text(0.055, 75, 'error rate 95%='+str(round(error_area_95*100, 1))+'%')
text(0.055, 70, 'average quadratic error='+str(round(average_quadratic_error, 2)))
text(0.055, 65, 'mean:'+str(mean)+'  var:'+str(var))
ylim(0, 100)
xlabel('h')
show()
