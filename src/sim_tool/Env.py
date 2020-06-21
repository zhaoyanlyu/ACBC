import numpy as np
import scipy.stats as stats
import scipy as sp
import matplotlib.pyplot as plt


class Env:
    def __init__(self, size_file=25000*8., size_chunk=10*8.,
                 bw_channel=40., trace_len=400.,
                 speed_mu=40., speed_sigma=5.):
        """

        :param size_file: float; Mbit;
        :param size_chunk: float; Mbit;
        :param bw_channel: float; Mbit/s;
        :param trace_len: float; m;
        :param speed_mu: float; m/s;
        :param speed_sigma: float; m^2/s^2
        """
        # Setup parameters
        self.size_file = size_file
        self.size_chunk = size_chunk
        self.bw_channel = bw_channel
        self.trace_len = trace_len
        self.speed_mu = speed_mu
        self.speed_sigma = speed_sigma

        # Parameters
        self.tot_chunk = int(np.ceil(size_file/size_chunk))
        print(f"tot_chunk: {self.tot_chunk}")

        # Helper parameters
        self.resolution = 1000
        self.pdf = np.zeros(self.resolution)
        self.begin = 0.001
        self.end = 0.08
        self.X = np.linspace(self.begin, self.end, self.resolution)
        self.fit_5()

    def fit_5(self):
        """
        The Gaussian speed does not give Gaussian dwell time for user in a trace. We use this fit function to
        approximate it into a Gaussian.
        :return:
        """
        for i in range(len(self.pdf)):
            self.pdf[i] = (np.e ** (- (((1 / self.X[i]) - self.speed_mu) ** 2) /
                                    (2 * (self.speed_sigma ** 2)))) /\
                          (((2 * np.pi) ** (1 / 2)) * (self.X[i] ** 2) * self.speed_sigma)
        max_spot = self.pdf.argmax()

        self.time_mu = self.X[max_spot]  # mean of travel time
        self.time_sigma = 1 / (np.max(self.pdf) * (2 * np.pi) ** (1 / 2)) # standard deviation
        self.chunk_mu = \
            int(np.floor((self.bw_channel * self.trace_len * self.time_mu) / self.size_chunk))
        self.chunk_sigma = \
            ((self.bw_channel * self.trace_len * self.time_sigma) / self.size_chunk) ** 2
        print(f"chunk_mu: {self.chunk_mu}, chunk_sigma: {self.chunk_sigma}")

    # total error rate calculation
    def get_error_rate(self):
        a_norm = stats.norm(self.time_mu, self.time_sigma)
        a_norm_pdf = a_norm.pdf(self.X)
        self.error_area = 0
        for x_spot in range(self.resolution):
            self.error_area += abs(a_norm_pdf[x_spot] - self.pdf[x_spot]) *\
                               (abs(self.begin - self.end) / self.resolution)
        print("error area = " + str(self.error_area))

    def plot_pdf(self):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        plt.plot(self.X, self.pdf)
        plt.show()


if __name__ == '__main__':

    speed_mu = 20
    speed_sigma = speed_mu/10
    env = Env(size_file=25000*8, size_chunk=10*8, bw_channel=100, trace_len=400,
              speed_mu=speed_mu, speed_sigma=speed_sigma)
    env.plot_pdf()
