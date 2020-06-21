import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
# from scipy.stats import *
import collections
import math
from tqdm import trange
from datetime import datetime


class RICH:
    def __init__(self, tau_begin=0.8, tau_decay_step=0.1, tau_end=0.4, tot_chunk=100, tot_EN=5, mu_xi=10., sigma_xi=1., EN_start=0):
        """

        :param tau: float; The probability of chunk being downloaded on all EN must above this threshold.
        :param tot_chunk: int; Total number of chunk.
        :param tot_EN: int; Total number of EN.
        :param mu_xi: float; The mean of the number of chunks could be downloaded from each EN.
        :param sigma_xi: float; the square root of variance of the number of chunks could be downloaded from each EN.
        :param EN_start: int; The index of EN that start download.
        """

        self.tau_begin = tau_begin
        self.tau_decay_step = tau_decay_step
        self.tau_end = tau_end
        self.tot_chunk = tot_chunk
        self.tot_EN = tot_EN
        self.mu_xi = mu_xi
        self.sigma_xi = sigma_xi
        self.EN_start = EN_start

        self.chunk_dict = {key: [] for key in range(tot_chunk)}  # The key is the chunk, and the value is the list of
        # ENs that this chunk should be saved to.
        self.prob_of_k = np.zeros(tot_chunk)  # The total probability of chunk k could being downloaded by this
        # prediction.
        self.x_scale = 2 * mu_xi
        self.xi_norm = stats.norm(mu_xi, sigma_xi)
        self.xi_norm_cdf = self.xi_norm.cdf(np.arange(0, self.x_scale, 1))
        self.xi_norm_pdf = self.xi_norm.pdf(np.arange(0, self.x_scale, 1))

        # self.yi_pdf = np.zeros((tot_EN, self.x_scale * (2 ** tot_EN)), dtype=np.float32)  # Empty list for the PDF of Yi
        self.yi_pdf = np.zeros((tot_EN, self.x_scale * tot_EN * 10), dtype=np.float32)  # Empty list for the PDF of Yi
        self.phi_i_k = np.zeros((tot_EN, tot_chunk * tot_EN), dtype=np.float32)  # Empty list for Phi_ik

        self.last_chunk = self.tot_chunk

        self.run_MAP()

    def run_MAP(self):
        # The array of pdf of Yi in each EN
        temp_mu_xi = self.mu_xi
        temp_sigma_xi = self.sigma_xi

        with trange(self.tot_EN-self.EN_start, unit="EN", ncols=150) as tq1:
            tq1.set_description("Building Yi")
            for idx_EN in range(self.EN_start, self.tot_EN):
                temp_norm = stats.norm(temp_mu_xi, temp_sigma_xi)
                self.yi_pdf[idx_EN, 0:(temp_mu_xi+self.mu_xi+1)] = temp_norm.pdf(np.arange(0, temp_mu_xi+self.mu_xi+1, 1))
                temp_mu_xi += self.mu_xi
                temp_sigma_xi = math.sqrt(temp_sigma_xi**2 + self.sigma_xi**2)
                tq1.update(1)

        # The probability of chunk k downloaded at ENi: øi(k)
        # When i = 0, ø0(k) = P(X >= k) = 1 - cdf(x = k)
        with trange(self.tot_EN-self.EN_start, unit='EN', ncols=150) as tq2:
            tq2.set_description_str("Building Phi_ik")
            for idx_chunk in range(self.tot_chunk):
                if idx_chunk < self.x_scale:
                    self.phi_i_k[self.EN_start, idx_chunk] = 1 - self.xi_norm_cdf[idx_chunk]
                else:
                    self.phi_i_k[self.EN_start, idx_chunk] = 0
            tq2.update(1)

            # When i > 0, use the formula
            for idx_EN in range(1 + self.EN_start, self.tot_EN):
                for kth_chunk in range(self.tot_chunk):
                    for idx_chunk_before_kth_chunk in range(kth_chunk):
                        # print('execute: i = '+str(i)+', k = '+str(k)+', n = '+str(n))
                        if self.tot_chunk >= self.x_scale:
                            x_cdf_array = np.hstack((self.xi_norm_cdf, np.ones(int(self.tot_chunk - self.x_scale))))
                        else:
                            x_cdf_array = self.xi_norm_cdf
                        self.phi_i_k[idx_EN, kth_chunk] += (1 - (x_cdf_array[kth_chunk - idx_chunk_before_kth_chunk])) *\
                                                           (self.yi_pdf[idx_EN - 1, idx_chunk_before_kth_chunk])
                tq2.update(1)

        # Reshape the øi(k) array
        phi_i_k_temp = np.zeros((self.tot_EN - self.EN_start, self.tot_chunk))
        for idx_EN in range(self.tot_EN - self.EN_start):
            for kth_chunk in range(self.tot_chunk):
                phi_i_k_temp[idx_EN, kth_chunk] = self.phi_i_k[idx_EN, kth_chunk]
        self.phi_i_k = phi_i_k_temp.copy()

        # Allocate chunks to EN according to phi_i_k
        new_tau = self.tau_begin
        for kth_chunk in range(self.tot_chunk):
            # Sort the array according to one column;
            # Save the column number in EN_order from big to small;
            EN_idx_sorted = np.argsort(-self.phi_i_k[:, kth_chunk])

            for idx_EN in range(self.tot_EN - self.EN_start):
                if self.prob_of_k[kth_chunk] < new_tau:
                    self.prob_of_k[kth_chunk] += self.phi_i_k[EN_idx_sorted[idx_EN], kth_chunk]
                    self.chunk_dict[kth_chunk] += [EN_idx_sorted[idx_EN]]
                else:
                    break
            # If all ENs cache can't match threshold, revise.
            if self.prob_of_k[kth_chunk] < new_tau:
                self.prob_of_k[kth_chunk] = 0
                self.chunk_dict[kth_chunk] = []
                self.last_chunk = kth_chunk - 1
                break

            # Update threshold \tau
            if kth_chunk % 10 == 0 and new_tau-self.tau_decay_step >= self.tau_end:
                new_tau -= self.tau_decay_step

        # build the cache_dict whose key is EN and values are chunks that are going to be cached in this EN.
        self.cache_dict = {i: [] for i in range(self.tot_EN)}
        for index_chunk in self.chunk_dict.keys():
            for index_EN in self.chunk_dict[index_chunk]:
                self.cache_dict[index_EN].append(index_chunk)

        print(f"{datetime.now()}: I MAP Done.")


    def calculate_pof(self):
        whole_dictionary = ''
        for i in range(self.tot_chunk):
            whole_dictionary += self.chunk_dict[i]
        numerator = len(whole_dictionary)
        denominator = self.last_chunk

        self.price_of_fog = numerator/denominator


    def get_price_of_fog(self):
        tot_copies_of_chunk = 0
        for idx_chunk in range(self.tot_chunk):
            tot_copies_of_chunk += len(self.chunk_dict[idx_chunk])
        num_types_of_chunk = self.last_chunk
        self.price_of_fog = tot_copies_of_chunk/num_types_of_chunk
        return self.price_of_fog

    def get_Yi(self, number_i_EN):
        return(self.yi_pdf[number_i_EN])

    def get_phi_i_k_for_ENi(self, number_i_EN):
        return(self.phi_i_k[number_i_EN])

    def get_download_percentage(self):
        return(self.last_chunk/self.tot_chunk)

    def get_summery_of_deployment(self):
        # the string of whole dictionary elements
        dictionary_str = ''
        for key in range(self.tot_chunk):
            dictionary_str += self.chunk_dict[key]
            # print(self.dictionary_of_EN[key])
        # count howmany chunks does each EN cache
        frequency_dictionary = collections.Counter(dictionary_str)
        print('Max spot = '+str(max(frequency_dictionary, key=frequency_dictionary.get))+', chunk number: '+str(max(frequency_dictionary.values())))

        last_EN = (max(frequency_dictionary.keys()))
        if last_EN == str(self.tot_EN - 1):
            print('All ENs are used. Download percentage = '+str(self.get_download_percentage()))
        elif self.get_download_percentage() == 1.0:
            print('All chunks are downloaded! Last EN = ' + str(last_EN))
        else:
            print('Needs to be optimized! Last EN = ' + str(last_EN) + '. Download percentage = ' + str(self.get_download_percentage()))

    def get_FEN(self):
        if self.last_chunk == self.tot_chunk:  # If the prediction area could cover the whole trip...
            return self.tot_EN
        else:
            num_chunk_on_EN_list = []
            for index_EN in range(self.tot_EN):
                num_chunk_on_EN_list.append(len(self.cache_dict[index_EN]))
            self.FEN = np.argmax(np.asarray(num_chunk_on_EN_list))
            return self.FEN

    def get_lastEN(self):
        self.last_EN = 0
        for index_EN in range(self.tot_EN):
            if len(self.cache_dict[index_EN]) != 0:
                self.last_EN += 1
            else:
                break
        return self.last_EN



if __name__ == '__main__':
    # MAP1 = MAP()
    # print(MAP1.get_price_of_fog())
    tau_begin = 0.9
    tau_decay_step = 0.1
    tau_end = 0.4
    tot_chunk = 100
    tot_EN = 10
    mu_xi = 10
    sigma_xi = 3
    EN_start = 0

    map = RICH(tau_begin=tau_begin, tau_decay_step=tau_decay_step, tau_end=tau_end,
               tot_chunk=tot_chunk, tot_EN=tot_EN, mu_xi=mu_xi, sigma_xi=sigma_xi, EN_start=EN_start)

    print(f'PoF = {map.get_price_of_fog()}')
    print(f'last chunk: {map.last_chunk}')
    # exit()
    # -- Plot setup
    SMALL_SIZE = 18
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 24


    csfont = {'fontname':'Comic Sans MS'}
    hfont = {'fontname':'Helvetica'}
    tfont = {'fontname': 'Times New Roman'}
    plt.rcParams['font.family'] = 'Times New Roman'

    # matplotlib.rcParams['text.usetex'] = True

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    x1 = np.arange(map.tot_chunk)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    plt.xlim(0, map.tot_chunk)


    # -- Plot the phi_i_k for each EN
    for i in range(map.tot_EN):
        plt.plot(x1, map.get_phi_i_k_for_ENi(i), color='gray', linestyle=':', linewidth=2.0)

    # -- Plot the total probability for each chunk
    plt.plot(x1, map.prob_of_k, color='red', linewidth=3)

    # -- Plot the threshold line
    tau_line = []
    new_tau = tau_begin
    for k in range(tot_chunk):
        tau_line.append(new_tau)
        if k % 10 == 0 and new_tau - tau_decay_step >= tau_end:
            new_tau -= tau_decay_step
    plt.plot(x1, tau_line, color='blue', linewidth=2.0, linestyle='--')


    # -- Spot all cached dots
    '''
    x2 = np.zeros(0)
    y2 = np.zeros(0)
    for item in map.chunk_dict.items():
        for times in range(len(item[1])):
            x2 = np.append(x2, item[0])
        for EN in range(len(item[1])):
            y2 = np.append(y2, map.get_phi_i_k_for_ENi(int(item[1][EN]))[item[0]])
    plot(x2, y2, '.', color='k')
    '''

    # Spot upward spikes
    '''
    upward_spike = zeros(0)
    temp_length = len(map.chunk_dict[0])
    for item in map.chunk_dict.items():
        if item[0] == 0:
            continue
        else:
            if len(item[1]) > temp_length:
                upward_spike = append(upward_spike, item[0])
        temp_length = len(item[1])
    y3 = np.ones(len(upward_spike))
    plot([upward_spike, upward_spike], [0, y3[0]], color='k', linestyle=':', linewidth=2.0)
    '''


    # -- Annotations
    '''
    ax1.annotate(r"$\varphi_{1k}$",
                 xy=(8, 0.8), xycoords='data',
                 xytext=(4, 0.7), textcoords='data',
                 va='center', ha='center',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=-0.2',
                                 fc='w')
                 )
    ax1.annotate(r"$\varphi_{2k}$",
                 xy=(18, 0.7), xycoords='data',
                 xytext=(14, 0.6), textcoords='data',
                 va='center', ha='center',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=-0.2',
                                 fc='w')
                 )
    ax1.annotate(r"$\varphi_{7k}$",
                 xy=(69, 0.42), xycoords='data',
                 xytext=(73, 0.52), textcoords='data',
                 va='center', ha='center',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=-0.2',
                                 fc='w')
                 )

    ax1.annotate(r"$\varphi_{5k}$",
                 xy=(48, 0.5), xycoords='data',
                 xytext=(52, 0.6), textcoords='data',
                 va='center', ha='center',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle='arc3,rad=-0.2',
                                 fc='w')
                 )

    ax1.annotate(r"64",
                 xy=(64, 0.), xycoords='data',
                 xytext=(67, -0.1), textcoords='data',
                 va='center', ha='center',
                 arrowprops=dict(arrowstyle='-',
                                 connectionstyle='arc3,rad=0',
                                 fc='w')
                 )

    ax1.annotate(r"threshold $\tau$",
                 xy=(80, 0.91), xycoords='data')
    '''
    # plt.xlabel('chunk index',)
    plt.xlabel(r'$k$',)
    # plt.ylabel('probability of chunk \nbeing downloaded by user from caches')
    plt.ylabel(r"$\varphi_{k}=\sum_{i}^{N}{\varphi_{ik}}$")

    # plt.tight_layout()

    plt.show()
