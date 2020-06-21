from pylab import *
from scipy.stats import *
import collections


class Scenario:
    size_of_file = 8000 # Mbit
    size_of_chunk = 80 # Mbit
    chunk_number = int(np.ceil((size_of_file/size_of_chunk)))
    bandwidth_wireless = 40 # Mbit
    trace_length = 1000
    mean = 28 # average speed
    var = 3.8 # speed standard deviation

    begin = 0.001
    end = 0.08
    resolution = 1000
    pdf = np.zeros(resolution)

    result_mean = 0
    result_std = 1

    chunk_mean = 0
    chunk_std = 1

    error_area = 0

    X = linspace(begin, end, resolution)

    # calculate time distribution, norm fit, using fit method 5
    def fit_5(self):
        for i in range(len(self.pdf)):
            self.pdf[i] = (math.e ** (- (((1 / self.X[i]) - self.mean) ** 2) / (2 * (self.var ** 2)))) / (
        ((2 * math.pi) ** (1 / 2)) * (self.X[i] ** 2) * self.var)
        max_spot = self.pdf.argmax()

        self.result_mean = self.X[max_spot] # mean of travel time
        self.result_std = 1 / (self.pdf.max() * (2 * math.pi) ** (1 / 2)) # standard deviation
        self.chunk_mean = \
            int(np.floor((self.bandwidth_wireless * self.trace_length * self.result_mean) / self.size_of_chunk))
        self.chunk_std = \
            ((self.bandwidth_wireless * self.trace_length * self.result_std) / self.size_of_chunk) ** 2
    # total error rate calculation
    def error_rate(self):
        a_norm = norm(self.result_mean, self.result_std)
        a_norm_pdf = a_norm.pdf(self.X)
        error_area = 0
        for x_spot in range(self.resolution):
            self.error_area += abs(a_norm_pdf[x_spot] - self.pdf[x_spot]) * (abs(self.begin - self.end) / self.resolution)
        print("error area = " + str(self.error_area))

    def __init__(self,
                 size_of_file=8000,
                 size_of_chunk=80,
                 bandwidth_wireless=40,
                 trace_length=1000,
                 mean=28,
                 var=3.8,
                 begin=0.001,
                 end=0.08,
                 resolution=1000):
        self.size_of_file = size_of_file
        self.size_of_chunk = size_of_chunk
        self.chunk_number = int(np.ceil((size_of_file/size_of_chunk)))
        self.bandwidth_wireless = bandwidth_wireless
        self.trace_length = trace_length
        self.mean = mean
        self.var = var
        self.begin = begin
        self.end = end
        self.resolution = resolution
        self.X = linspace(begin, end, resolution)



class MAP:
    # Threshold: t
    # The probability of each chunk downloaded should higher than it.
    threshold = 0.8
    # Number of total chunk: K
    total_chunk = 100
    # Number of total ENs: I
    total_EN = 5

    # Yi = Yi-1 + Xi = ∑Xj, j = 1:i
    # Xi ~ P(triangle (0.5)， avg=10)
    # Expectation of Xi
    expectation_Xi = 10

    standard_deviation_Xi = 1

    # Due to triangle distribution, the scale of X should be twice the mean
    x_scale = 2 * expectation_Xi

    # Set an array to store the pdf of Yi
    Yi_pdf = np.zeros((total_EN, x_scale * (2 ** total_EN)))
    # Set an array for phi_i_k
    phi_i_k = zeros((total_EN, total_chunk * total_EN))
    # Set a dictionary, key is the number of chunk, value is the ENs need to store the chunk;
    dictionary_of_EN = {key: '' for key in range(total_chunk)}
    # Set an array to store the value of probability of each chunk been downloaded
    prob_of_k = np.zeros(total_chunk)

    # Price of fog
    price_of_fog = 1.0
    last_chunk = total_chunk

    a_norm = norm(expectation_Xi, standard_deviation_Xi)
    a_norm_cdf = a_norm.cdf(np.arange(0, x_scale, 1))
    a_norm_pdf = a_norm.pdf(np.arange(0, x_scale, 1))

    EN_start = 0

    # Due to do the math will take up too much resource and do it once is enough,
    #  so if the math is done, turn the flag to True
    flag_math_is_done = False
    flag_pof_is_done = False
    EN_unused = -1

    def algrithm_1(self):
        # Algorithm 1
        for k in range(self.total_chunk):
            # Sort the array according to one column;
            # Save the column number in EN_order from big to small;
            EN_order = np.argsort(-self.phi_i_k[:, k])
            for i in range(self.total_EN - self.EN_start):
                if self.prob_of_k[k] < self.threshold:
                    self.prob_of_k[k] += self.phi_i_k[EN_order[i], k]
                    self.dictionary_of_EN[k] += str(EN_order[i])
                else:
                    break
            # If all ENs cache can't match threshold, revise.
            if self.prob_of_k[k] < self.threshold:
                self.prob_of_k[k] = 0
                self.dictionary_of_EN[k] = ''
                self.last_chunk = k - 1
                break

    def do_the_math(self):
        # The array of pdf of Yi in each EN
        expectation_Xi_cal_Yi = self.expectation_Xi
        squired_deviation_Xi_cal_Yi = self.standard_deviation_Xi ** 2
        for i in range(self.EN_start, self.total_EN):
            standard_deviation_Xi_cal_Yi = squired_deviation_Xi_cal_Yi ** (1/2)
            temp_norm = norm(expectation_Xi_cal_Yi, standard_deviation_Xi_cal_Yi)
            self.Yi_pdf[i, 0:expectation_Xi_cal_Yi + self.expectation_Xi + 1] = \
                temp_norm.pdf(np.arange(0, expectation_Xi_cal_Yi + self.expectation_Xi + 1, 1))
            expectation_Xi_cal_Yi += self.expectation_Xi
            squired_deviation_Xi_cal_Yi += self.standard_deviation_Xi ** 2

        # The probability of chunk k downloaded at ENi: øi(k)
        # When i = 0, ø0(k) = P(X >= k) = 1 - cdf(x = k)
        for k in range(self.total_chunk):
            if k < self.x_scale:
                self.phi_i_k[self.EN_start, k] = 1 - self.a_norm_cdf[k]
            else:
                self.phi_i_k[self.EN_start, k] = 0
        # When i > 0, use the formula
        for i in range(1 + self.EN_start, self.total_EN):
            for k in range(self.total_chunk):
                for n in range(k):
                    # print('execute: i = '+str(i)+', k = '+str(k)+', n = '+str(n))
                    if self.total_chunk >= self.x_scale:
                        x_cdf_array = np.hstack((self.a_norm_cdf, np.ones(self.total_chunk - self.x_scale)))
                    else:
                        x_cdf_array = self.a_norm_cdf
                    self.phi_i_k[i, k] += (1 - (x_cdf_array[k - n])) * (self.Yi_pdf[i - 1, n])
        # Reshape the øi(k) array
        phi_i_k_temp = np.zeros((self.total_EN - self.EN_start, self.total_chunk))
        for i in range(self.total_EN - self.EN_start):
            for k in range(self.total_chunk):
                phi_i_k_temp[i, k] = self.phi_i_k[i, k]
        self.phi_i_k = phi_i_k_temp.copy()
        self.algrithm_1()
        self.flag_math_is_done = True

    def __init__(self, threshold=0.8, total_chunk=100, total_EN=5, expectation_Xi=10, standard_deviation_Xi=1):
        self.threshold = threshold
        self.total_chunk = total_chunk
        self.last_chunk = self.total_chunk
        self.total_EN = total_EN
        self.expectation_Xi = int(expectation_Xi)
        self.standard_deviation_Xi = standard_deviation_Xi
        self.dictionary_of_EN = {key: '' for key in range(total_chunk)}
        self.prob_of_k = np.zeros(total_chunk)
        self.x_scale = 2 * expectation_Xi
        self.a_norm = norm(expectation_Xi, standard_deviation_Xi)
        self.a_norm_cdf = self.a_norm.cdf(np.arange(0, self.x_scale, 1))
        self.a_norm_pdf = self.a_norm.pdf(np.arange(0, self.x_scale, 1))
        self.Yi_pdf = np.zeros((total_EN, self.x_scale * (2 ** total_EN)))
        self.phi_i_k = zeros((total_EN, total_chunk * total_EN))
        self.dictionary_of_EN = {key: '' for key in range(total_chunk)}
        self.flag_math_is_done = False
        self.flag_pof_is_done = False

    def calculate_pof(self):
        if self.flag_math_is_done is False:
            self.do_the_math()
        whole_dictionary = ''
        for i in range(self.total_chunk):
            whole_dictionary += self.dictionary_of_EN[i]
        numerator = len(whole_dictionary)
        denominator = self.last_chunk
        print(numerator)

        self.price_of_fog = numerator/denominator
        self.flag_pof_is_done = True

    def get_prob_of_k(self):
        if self.flag_math_is_done is False:
            self.do_the_math()
        return(self.prob_of_k)

    def get_dictionary(self):
        if self.flag_math_is_done is False:
            self.do_the_math()
        return(self.dictionary_of_EN)

    def get_price_of_fog(self):
        if self.flag_pof_is_done is False:
            self.calculate_pof()
        return self.price_of_fog

    def get_Yi(self, number_i_EN):
        if self.flag_math_is_done is False:
            self.do_the_math()
        return(self.Yi_pdf[number_i_EN])

    def get_phi_i_k(self, number_i_EN):
        if self.flag_math_is_done is False:
            self.do_the_math()
        return(self.phi_i_k[number_i_EN])
    def get_last_chunk(self, last_chunk):
        if self.flag_math_is_done is False:
            self.do_the_math()
        return(self.last_chunk)
    def get_download_percentage(self):
        if self.flag_math_is_done is False:
            self.do_the_math()
        return(self.last_chunk/self.total_chunk)
    def get_summery_of_deployment(self):
        if self.flag_math_is_done is False:
            self.do_the_math()
        # the string of whole dictionary elements
        dictionary_str = ''
        for key in range(self.total_chunk):
            dictionary_str += self.dictionary_of_EN[key]
            # print(self.dictionary_of_EN[key])
        # count howmany chunks does each EN cache
        frequency_dictionary = collections.Counter(dictionary_str)
        print('Max spot = '+str(max(frequency_dictionary, key=frequency_dictionary.get))+', chunk number: '+str(max(frequency_dictionary.values())))

        last_EN = (max(frequency_dictionary.keys()))
        if last_EN == str(self.total_EN - 1):
            print('All ENs are used. Download percentage = '+str(self.get_download_percentage()))
        elif self.get_download_percentage() == 1.0:
            print('All chunks are downloaded! Last EN = ' + str(last_EN))
        else:
            print('Needs to be optimized! Last EN = ' + str(last_EN) + '. Download percentage = ' + str(self.get_download_percentage()))


# MAP1 = MAP()
# print(MAP1.get_price_of_fog())

example_scenario = Scenario(trace_length=800, size_of_file=20000, mean=24.488, var=3.81)
example_scenario.fit_5()
print(example_scenario.result_mean)

example_MAP = MAP(total_chunk=example_scenario.chunk_number, expectation_Xi=example_scenario.chunk_mean, standard_deviation_Xi=example_scenario.chunk_std, total_EN=20, threshold = 0.9)
example_MAP.do_the_math()

# print(example_MAP.get_dictionary())
print('PoF = '+ str(example_MAP.get_price_of_fog()))
example_MAP.get_summery_of_deployment()

x1 = np.arange(example_MAP.total_chunk)
figure(figsize=(13, 6), dpi=100)
subplot(1, 1, 1)
for i in range(example_MAP.total_EN):
    plot(x1, example_MAP.get_phi_i_k(i), color='k')
plot(x1, example_MAP.get_prob_of_k(), color='k', linewidth=2.0)

# threshold line
plot([0, example_scenario.chunk_number], [example_MAP.threshold, example_MAP.threshold], color='k', linewidth=1.0, linestyle='-.')


# spot all cached dots
x2 = np.zeros(0)
y2 = np.zeros(0)
for item in example_MAP.dictionary_of_EN.items():
    for times in range(len(item[1])):
        x2 = np.append(x2, item[0])
    for EN in range(len(item[1])):
        y2 = np.append(y2, example_MAP.get_phi_i_k(int(item[1][EN]))[item[0]])
# plot(x2, y2, '.', color='k')

# spot upward spikes
upward_spike = zeros(0)
temp_length = len(example_MAP.dictionary_of_EN[0])
for item in example_MAP.dictionary_of_EN.items():
    if item[0] == 0:
        continue
    else:
        if len(item[1]) > temp_length:
            upward_spike = append(upward_spike, item[0])
    temp_length = len(item[1])
y3 = np.ones(len(upward_spike))
# plot([upward_spike, upward_spike], [0, y3[0]], color='k', linestyle=':', linewidth=2.0)

xlabel('chunk number')
ylabel('probability')
xlim(0, example_MAP.total_chunk)
show()





