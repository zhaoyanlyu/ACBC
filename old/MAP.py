from pylab import *
from scipy.stats import *
# In the library pylab, have a defination of 'norm' refer to the distance between vectors, while in the library scipy.stats, there is a defination of 'norm' refer to normal distribution. So we should import pylab first if we want to use norm as normal distribution.


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

    # Due to triangle distribution, the scale of X should be twice the mean
    x_scale = 2 * expectation_Xi

    # Set an array to store the pdf of Yi
    Yi_pdf = np.zeros((total_EN, x_scale * total_EN))
    # Set an array for phi_i_k
    phi_i_k = zeros((total_EN, total_chunk * total_EN))
    # Set a dictionary, key is the number of chunk, value is the ENs need to store the chunk;
    dictionary_of_EN = {key: '' for key in range(total_chunk)}
    # Set an array to store the value of probability of each chunk been downloaded
    prob_of_k = np.zeros(total_chunk)

    # Price of fog
    price_of_fog = 1.0

    # Due to do the math will take up too much resource and do it once is enough,
    #  so if the math is done, turn the flag to True
    flag_math_is_done = False
    flag_pof_is_done = False

    def do_the_math(self):
        # The array of pdf of Yi in each EN
        for i in range(self.total_EN):
            x_range = np.arange(0, 1, 1 / self.x_scale)

            #triangular distribution
            triang_pdf = triang.pdf(x_range, 0.5)
            triang_pdf /= self.x_scale  # Adjust to less then 1.

            self.Yi_pdf[i, :self.x_scale] = triang_pdf

            for ii in range(i): # ii is index of ENs, EN1, EN2, ... ENii. From 1 to i
                self.Yi_pdf[i, :(ii + 2) * self.x_scale] = np.hstack(
                    (np.convolve(self.Yi_pdf[i, :(ii + 1) * self.x_scale], triang_pdf), np.zeros(1)))

        # The probability of chunk k downloaded at ENi: øi(k)
        # When i = 0, ø0(k) = P(X >= k) = 1 - cdf(x = k)
        for k in range(self.total_chunk):
            x_range = np.arange(0, 1, 1 / self.x_scale)
            if k < self.x_scale:
                self.phi_i_k[0, k] = 1 - triang.cdf(x_range, 0.5)[k]
            else:
                self.phi_i_k[0, k] = 0
        # When i > 0, use the formula
        for i in range(1, self.total_EN):
            for k in range(self.total_chunk):
                for n in range(k):
                    # print('execute: i = '+str(i)+', k = '+str(k)+', n = '+str(n))
                    x_range = np.arange(0, 1, 1 / self.x_scale)
                    x_cdf_array = np.hstack((triang.cdf(x_range, 0.5), np.ones(self.total_chunk - self.x_scale)))
                    self.phi_i_k[i, k] += (1 - (x_cdf_array[k - n])) * (self.Yi_pdf[i - 1, n])
                    # print(phi_i_k[i, :])
        # Reshape the øi(k) array
        phi_i_k_temp = np.zeros((self.total_EN, self.total_chunk))
        for i in range(self.total_EN):
            for k in range(self.total_chunk):
                phi_i_k_temp[i, k] = self.phi_i_k[i, k]
        self.phi_i_k = phi_i_k_temp.copy()

        # Algorithm 1
        for k in range(self.total_chunk):
            # Sort the array according to one column;
            # Save the column number in EN_order from big to small;
            EN_order = np.argsort(-self.phi_i_k[:, k])
            # print(EN_order)
            for i in range(self.total_EN):
                if self.prob_of_k[k] < self.threshold:
                    self.prob_of_k[k] += self.phi_i_k[EN_order[i], k]
                    self.dictionary_of_EN[k] += str(EN_order[i])
                else:
                    break
            # If all ENs cache can't match threshold, abanden.
            if self.prob_of_k[k] < self.threshold:
                self.prob_of_k[k] = 0
                self.dictionary_of_EN[k] = ''
                break

        self.flag_math_is_done = True

    def __init__(self, threshold=0.8, total_chunk=100, total_EN=5, expectation_Xi=10):
        self.threshold = threshold
        self.total_chunk = total_chunk
        self.total_EN = total_EN
        self.expectation_Xi = expectation_Xi
        self.dictionary_of_EN = {key: '' for key in range(total_chunk)}
        self.prob_of_k = np.zeros(total_chunk)
        self.x_scale = 2 * expectation_Xi
        self.flag_math_is_done = False
        self.flag_pof_is_done = False

    def calculate_pof(self):
        if self.flag_math_is_done is False:
            self.do_the_math()
        whole_dictionary = ''
        for i in range(self.total_chunk):
            whole_dictionary += self.dictionary_of_EN[i]
        numerator = len(whole_dictionary)
        denominator = self.total_chunk
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

# MAP1 = MAP()
# print(MAP1.get_price_of_fog())

example = MAP(total_chunk=100)
example.do_the_math()

x1 = np.arange(100)
figure(figsize=(13, 6), dpi=100)
subplot(1, 1, 1)
for i in range(5):
    plot(x1, example.get_phi_i_k(i))
plot(x1, example.get_prob_of_k())
show()

"""
examples = []
for i in range(10, 13):
    x = MAP(total_EN=i)
    examples.append(x)

x1 = np.arange(100)
figure(figsize=(13, 6), dpi=100)
subplot(1, 1, 1)
for i in range(10, 13):
    plot(x1, examples[i-10].get_prob_of_k(), label='total_EN = '+str(i))
    print('EN = '+str(i)+', PoF = '+str(examples[i-10].get_price_of_fog()))
for i in range(10, 13):
    print(examples[i-10].get_dictionary())
legend(loc='best')
show()
"""
