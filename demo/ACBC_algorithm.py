from pylab import *
from scipy.stats import *
import collections

# np.set_printoptions(threshold=np.nan)

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
            self.error_area += \
                abs(a_norm_pdf[x_spot] - self.pdf[x_spot]) * (abs(self.begin - self.end) / self.resolution)
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
        self.fit_5()


class MAP:

    EN_start = 0
    chunk_start = 0

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
    dictionary_array_str = np.zeros(0)
    # Set an array to store the value of probability of each chunk been downloaded
    prob_of_k = np.zeros(total_chunk)

    # Price of fog
    price_of_fog = 1.0
    last_chunk = total_chunk

    a_norm = norm(expectation_Xi, standard_deviation_Xi)
    a_norm_cdf = a_norm.cdf(np.arange(0, x_scale, 1))
    a_norm_pdf = a_norm.pdf(np.arange(0, x_scale, 1))

    last_EN = -1

    # Due to do the math will take up too much resource and do it once is enough,
    #  so if the math is done, turn the flag to True
    flag_math_is_done = False
    flag_pof_is_done = False

    def algrithm_1(self):
        # Algorithm 1
        for k in range(self.chunk_start, self.total_chunk):
            # Sort the array according to one column;
            # Save the column number in EN_order from big to small;
            EN_order = np.argsort(-self.phi_i_k[:, k])
            # print(EN_order)
            for i in range(self.total_EN - self.EN_start):
                if self.prob_of_k[k] < self.threshold:
                    self.prob_of_k[k] += self.phi_i_k[EN_order[i], k]
                    self.dictionary_of_EN[k] += str(EN_order[i])
                    self.dictionary_of_EN[k] += ' '
                else:
                    break
            # If all ENs cache can't match threshold, give up.
            if self.prob_of_k[k] < self.threshold:
                self.prob_of_k[k] = 0
                self.dictionary_of_EN[k] = ''
                self.last_chunk = k - 1
                break
        # print(self.prob_of_k)

    def do_the_math(self):
        # The array of pdf of Yi in each EN
        # Calculate expectation and deviation of normal distribution, simplized part.
        """
        print('\nDoing the math, EN_start = '
              + str(self.EN_start)
              + ', chunk_start = '
              + str(self.chunk_start))
        """
        expectation_Xi_cal_Yi = self.expectation_Xi
        squired_deviation_Xi_cal_Yi = self.standard_deviation_Xi ** 2
        for i in range(self.EN_start, self.total_EN):
            standard_deviation_Xi_cal_Yi = squired_deviation_Xi_cal_Yi ** (1/2)
            temp_norm = norm(expectation_Xi_cal_Yi, standard_deviation_Xi_cal_Yi)
            self.Yi_pdf[i, :expectation_Xi_cal_Yi + self.expectation_Xi + 1] = \
                temp_norm.pdf(np.arange(0, expectation_Xi_cal_Yi + self.expectation_Xi + 1, 1))
            expectation_Xi_cal_Yi += self.expectation_Xi
            squired_deviation_Xi_cal_Yi += self.standard_deviation_Xi ** 2
        # The probability of chunk k downloaded at ENi: øi(k)
        # When i = 0, ø0(k) = P(X >= k) = 1 - cdf(x = k)
        for k in range(self.chunk_start, self.total_chunk):
            if k - self.chunk_start < self.x_scale:
                self.phi_i_k[self.EN_start, k] = 1 - self.a_norm_cdf[k - self.chunk_start]
            else:
                self.phi_i_k[self.EN_start, k] = 0

        # When i > 0, use the formula
        if self.total_chunk - self.chunk_start >= self.x_scale:
            x_cdf_array = np.hstack((self.a_norm_cdf,
                                     np.ones(self.total_chunk - self.chunk_start - self.x_scale)))
        else:
            x_cdf_array = self.a_norm_cdf

        for i in range(self.EN_start + 1, self.total_EN):
            for k in range(self.chunk_start, self.total_chunk):
                for n in range(self.chunk_start, k):
                    # print('execute: i = '+str(i)+', k = '+str(k)+', n = '+str(n))
                    self.phi_i_k[i, k] +=\
                        (1 - (x_cdf_array[k - n])) * (self.Yi_pdf[i - 1, n - self.chunk_start])
        # Reshape the øi(k) array
        phi_i_k_temp = np.zeros((self.total_EN, self.total_chunk))
        for i in range(self.EN_start, self.total_EN):
            for k in range(self.chunk_start, self.total_chunk):
                phi_i_k_temp[i, k] = self.phi_i_k[i, k]
        # Shape of phi_i_k: ((total_EN) * (total_chunk))
        self.phi_i_k = phi_i_k_temp.copy()
        self.algrithm_1()
        self.flag_math_is_done = True
        self.get_summery_of_deployment()

    def __init__(self,
                 threshold=0.8,
                 total_chunk=100,
                 total_EN=5,
                 expectation_Xi=10,
                 standard_deviation_Xi=1,
                 EN_start=0,
                 chunk_start=0):
        self.threshold = threshold
        self.total_chunk = total_chunk
        self.last_chunk = self.total_chunk
        self.total_EN = total_EN
        self.EN_start = EN_start
        self.chunk_start = chunk_start
        self.expectation_Xi = int(expectation_Xi)
        self.standard_deviation_Xi = standard_deviation_Xi
        self.dictionary_of_EN = {key: '' for key in range(chunk_start, total_chunk)}
        self.prob_of_k = np.zeros(total_chunk)
        self.x_scale = 2 * expectation_Xi
        self.a_norm = norm(expectation_Xi, standard_deviation_Xi)
        self.a_norm_cdf = self.a_norm.cdf(np.arange(0, self.x_scale, 1))
        self.a_norm_pdf = self.a_norm.pdf(np.arange(0, self.x_scale, 1))
        self.Yi_pdf = np.zeros((total_EN, self.x_scale * (2 ** (total_EN))))
        self.phi_i_k = zeros((total_EN, (total_chunk) * (total_EN)))
        self.dictionary_of_EN = {key: '' for key in range(self.total_chunk)}
        self.dictionary_array_str = np.zeros(0)
        self.last_EN = -1
        self.flag_math_is_done = False
        self.flag_pof_is_done = False
        self.do_the_math()

    def calculate_pof(self):
        if self.flag_math_is_done is False:
            self.do_the_math()
        numerator = len(self.dictionary_array_str)
        denominator = self.last_chunk - self.chunk_start + 1
        print('Numerator = ' + str(numerator))
        print('Denominator = ' + str(denominator))
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

    def get_summery_of_deployment(self, return_value=NaN, print_result=False):
        # the string of whole dictionary elements
        dictionary_str = ''
        for key in range(self.chunk_start, self.total_chunk):
            dictionary_str += self.dictionary_of_EN[key]
        temp = ''
        for letter in range(len(dictionary_str)):
            if dictionary_str[letter] != ' ':
                temp += dictionary_str[letter]
            else:
                self.dictionary_array_str = append(self.dictionary_array_str, temp)
                temp = ''
        # count howmany chunks does each EN cache
        frequency_dictionary = collections.Counter(self.dictionary_array_str)
        dictionary_array_int = map(int, self.dictionary_array_str)
        self.last_EN = str(max(dictionary_array_int))

        if print_result == True:
            print('\n\n-----------------SUMMERY START-----------------')
            print(frequency_dictionary)
            print('Max spot = '
                  + str(max(frequency_dictionary, key=frequency_dictionary.get))
                  + ', chunk number: '
                  + str(max(frequency_dictionary.values())))
            print('Last chunk = '
                  + str(self.last_chunk)
                  + ', last EN = ' + str(self.last_EN))
            # print('PoF = ' + str(example_MAP.get_price_of_fog()))

            if self.last_EN == str(self.total_EN - 1) and print_result == True :
                print('All ENs are used. Download percentage = '
                      + str(self.get_download_percentage())
                      + '. Last chunk: ' + str(self.last_chunk))
            elif self.get_download_percentage() == 1.0 and print_result == True:
                print('All chunks are downloaded! Last EN = ' + str(self.last_EN))
            elif print_result == True:
                print('Needs to be optimized! Last EN = '
                      + str(self.last_EN)
                      + '. Download percentage = '
                      + str(self.get_download_percentage())
                      + '. Last chunk = '+ str(self.last_chunk))
            print('------------------SUMMERY END------------------\n\n')

        if return_value == 'PeakSpot':
            return int(max(frequency_dictionary, key=frequency_dictionary.get))
        elif return_value == 'PeakchunkNumber':
            return int(max(frequency_dictionary.values()))
        elif return_value == 'MostFrequencychunkNumber':
            return argmax(self.get_phi_i_k(int(max(frequency_dictionary, key = frequency_dictionary.get))))
        elif return_value == 'LastEN':
            return int(self.last_EN)


def generate_random_chunk(example_MAP):
    # Feedback structure.
    # Select feedback EN, which is the EN have the most chunk cached.
    # print('Generating random chunk......')
    feedback_EN = int(example_MAP.get_summery_of_deployment(return_value='PeakSpot'))
    # print('Feedback EN is EN ' + str(feedback_EN))
    # Get all chunks that cached in this EN, as feedback chunks.
    feedback_chunks = np.zeros(0)
    for item in example_MAP.dictionary_of_EN.items():
        if str(feedback_EN) in item[1]:
            feedback_chunks = append(feedback_chunks, item[0])
    # print('Feedback chunks are:')
    # print(feedback_chunks)

    # Get the probability that each chunk is downloaded in this feedback EN.
    probability_list = np.zeros(len(feedback_chunks))
    for i in range(len(feedback_chunks)):
        probability_list[i] = example_MAP.get_phi_i_k(feedback_EN)[int(feedback_chunks[i])]
    # Adjust to probability that sum = 1
    probability_sum_temp = np.sum(probability_list)
    for i in range(len(feedback_chunks)):
        probability_list[i] = probability_list[i] / probability_sum_temp
    # print(probability_list)
    # Randomly geneat a chunk that simulate the last chunk downloaded in feedback EN.
    end_chunk = np.random.choice(feedback_chunks, p=probability_list)
    # print('The last chunk downloaded in feedback EN is ' + str(end_chunk) + ' (random generated)')
    return end_chunk, feedback_EN


def run(example_scenario, example_MAP):
    # Set the scenario and run for the first time

    # Entire solution buffers:
    # Include a dictionary that map the chunks and the ENs that cache these chunks
    whole_dictionary = {key: '' for key in range(example_MAP.total_chunk)}
    for item in example_MAP.get_dictionary().items():
        if item[1] != '':
            whole_dictionary[item[0]] = item[1]
    # Include an array of prob_of_k to measure the probability of each chunk that coule be downloaded
    whole_prob_of_k = zeros(example_MAP.total_chunk)
    for i in range(example_MAP.total_chunk):
        if example_MAP.get_prob_of_k()[i] != 0:
            whole_prob_of_k[i] = example_MAP.get_prob_of_k()[i]
    # Include an array of dictionary_array_str to save all dictionary_array_str
    whole_dictionary_array_str = zeros(0)
    whole_dictionary_array_str = append(whole_dictionary_array_str, example_MAP.dictionary_array_str)
    # Include an array of all feedback chunks
    whole_feedback_chunks = np.zeros(0)
    # Include an array of all feedback ENs
    whole_feedback_ENs = np.zeros(0)

    end_chunk = 0
    feedback_EN = 0
    # If not finished, run the MAP over and over again
    while (int(example_MAP.last_EN) != int(example_MAP.total_EN - 1) and
                   int(example_MAP.last_chunk) != int(example_MAP.total_chunk)):

        # example_MAP.get_summery_of_deployment(print_result=True)
        # print('Run again...')
        end_chunk, feedback_EN = generate_random_chunk(example_MAP)
        whole_feedback_chunks = np.append(whole_feedback_chunks, int(end_chunk))
        whole_feedback_ENs = np.append(whole_feedback_ENs, int(feedback_EN))
        # print(end_chunk, feedback_EN)
        # end_chunk = example_MAP.get_summery_of_deployment(return_value='MostFrequencychunkNumber')

        # print('end_chunk=' + str(end_chunk) + '; feedback_EN=' + str(feedback_EN))
        example_MAP = MAP(total_chunk=example_scenario.chunk_number,
                          expectation_Xi=example_scenario.chunk_mean,
                          standard_deviation_Xi=example_scenario.chunk_std,
                          total_EN=20,
                          threshold=0.9,
                          EN_start=int(feedback_EN + 1),
                          chunk_start=int(end_chunk + 1))
        # Update solution buffers
        # Update whole_dictionary
        for item in example_MAP.get_dictionary().items():
            if item[1] != '':
                whole_dictionary[item[0]] = item[1]
        # Update whole_prob_of_k
        for i in range(example_MAP.total_chunk):
            if example_MAP.get_prob_of_k()[i] != 0:
                whole_prob_of_k[i] = example_MAP.get_prob_of_k()[i]
        # Update whole_dictionary_array_str
        whole_dictionary_array_str = append(whole_dictionary_array_str, example_MAP.dictionary_array_str)

    #example_MAP.get_summery_of_deployment(print_result=True)

    whole_array_EN = np.zeros([example_MAP.total_EN, example_MAP.total_chunk])
    whole_array_EN[whole_array_EN==0] = NaN
    for item in whole_dictionary.items():
        temp_EN_number = ''
        for letter in item[1]:
            if letter != ' ':
                temp_EN_number +=letter
            else:
                temp_EN_number = int(temp_EN_number)
                whole_array_EN[temp_EN_number][item[0]] = item[0]
                temp_EN_number = ''
    whole_dictionary_EN = {key: np.zeros(0) for key in range(example_MAP.total_EN)}
    for EN in range(example_MAP.total_EN):
        for chunk in range(example_MAP.total_chunk):
            if 0 <= whole_array_EN[EN][chunk] <= example_MAP.total_chunk:
                whole_dictionary_EN[EN] = np.append(whole_dictionary_EN[EN], whole_array_EN[EN][chunk])


    print('Finish!!!!')
    return whole_dictionary, whole_dictionary_EN, whole_feedback_ENs, whole_feedback_chunks

file_dictionary = {1:10000, 2:12000, 3:15000, 4:18000, 5:20000, 6:24000}