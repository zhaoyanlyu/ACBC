from pylab import *
from scipy.stats import *
import numpy as np

mean = 24.488
var = 3.81
test_parameter = 10
test_numbers = 100000
begin = 0.001
end = 0.08
resolution = 1000

fake_tests = np.zeros(test_numbers)

X = linspace(begin, end, resolution)
# pdf of reciprocal normal distribution
pdf = np.zeros(resolution)
for i in range(len(pdf)):
    pdf[i] = (math.e ** (- (((1/X[i]) - mean)**2) / (2 * (var**2)))) / (((2 * math.pi) ** (1/2)) * (X[i]**2) * var)
max_spot = pdf.argmax()
# fake some tests for normal distribution generation
pdf_sum = 0
for x_spot in range(resolution):
    pdf_sum += pdf[x_spot]
pdf_adjusted = np.zeros(resolution)
for x_spot in range(resolution):
    pdf_adjusted[x_spot] = pdf[x_spot]/pdf_sum
temp_test_sum = 0
for i in range(test_numbers):
    fake_tests[i] = np.random.choice(X, p=pdf_adjusted)

# fit faked tests to a normal distribution
mu, std = norm.fit(fake_tests)

print("mean = "+str(mu)+"; var = "+str(std))
norm_fit = norm.pdf(X, mu, std)

# error calculations

# total error rate calculation
error_area = 0
for x_spot in range(resolution):
    error_area += abs(norm_fit[x_spot] - pdf[x_spot]) * (abs(begin-end) / resolution)
print("error area = "+str(error_area))

# 95% of values error rate calculation
error_area_95 = 0
left_edge = 0
right_edge = resolution - 1
for x_spot in range(resolution):
    if X[x_spot] < mu-2*std <= X[x_spot+1]:
        left_edge = x_spot
        break
for x_spot in range(resolution):
    if X[x_spot] < mu+2*std <= X[x_spot+1]:
        right_edge = x_spot
        break
print("left edge = "+str(left_edge)+"; right edge"+str(right_edge))
for x_spot in range(left_edge, right_edge):
    error_area_95 += abs(norm_fit[x_spot] - pdf[x_spot]) * (abs(begin-end) / resolution)
print("error area 95% = "+str(error_area_95))

# average quadratic error calculation
average_quadratic_error = 0
for x_spot in range(resolution):
    average_quadratic_error += (norm_fit[x_spot] - pdf[x_spot])**2
average_quadratic_error /= resolution

print("average quadratic error"+str(average_quadratic_error))

# plot the figure
figure(figsize=(10, 6), dpi=100)
subplot(1, 1, 1)
plot([X[left_edge], X[left_edge]], [0, pdf.max()*1.1], color='k', linestyle='--')
plot([X[right_edge], X[right_edge]], [0, pdf.max()*1.1], color='k', linestyle='--')
plot(X, pdf, color='k', linewidth=2.0, label='original', linestyle='-')
plot(X, norm_fit, color='k', linewidth=2.0, label='fit', linestyle='--')
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
fill_between(X, norm_fit, pdf, facecolor='k', alpha=0.2)
text(0.06, 80, 'error rate='+str(round(error_area*100, 1))+'%')
text(0.06, 75, 'error rate 95%='+str(round(error_area_95*100, 1))+'%')
text(0.06, 70, 'average quadratic error='+str(round(average_quadratic_error, 2)))
text(0.06, 65, 'mean:'+str(mean)+'  var:'+str(var))
ylim(0, 100)
xlabel('h')
show()

