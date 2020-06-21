import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd


FONT_CS = 'Comic Sans MS'
FONT_HE = 'Helvetica'
FONT_TI = 'Times New Roman'


def fast_plot_MRI(map):

    fig, ax1 = plt.subplots(figsize=(20, 6), dpi=100)
    if map.last_chunk >= map.tot_chunk - 10:
        xmax = map.tot_chunk
    else:
        xmax = map.last_chunk+10

    plt.xlim(0, xmax)
    x1 = np.arange(map.tot_chunk)

    for i in range(map.tot_EN):
        plt.plot(x1, map.get_phi_i_k_for_ENi(i), color='gray', linestyle=':', linewidth=2.0)

    # -- Plot the total probability for each chunk
    plt.plot(x1, map.prob_of_k, color='red', linewidth=3)

    # -- Plot the threshold line
    plt.plot([0, map.tot_chunk], [map.tau, map.tau], color='blue', linewidth=2.0, linestyle='--')

    plt.xlabel(r'$k$',)
    # plt.ylabel('probability of chunk \nbeing downloaded by user from caches')
    plt.ylabel(r"$\varphi_{k}=\sum_{i}^{N}{\varphi_{ik}}$")
    # plt.tight_layout()
    plt.show()


def log_results(log_file_dir, dict_to_save):
    """
    Save the running results to the log file.
    :param log_file_dir: The path to save the log file
    :param dict_to_save: dict; The data for each item must be a single number, String, bool, or a list with length of 1.
    :return: None
    """
    # Make every data as a list with length of 1.
    for key in dict_to_save:
        if not isinstance(dict_to_save[key], list):
            dict_to_save[key] = [dict_to_save[key]]
        else:
            if len(dict_to_save[key]) != 1:
                exit(f'{datetime.now().replace(microsecond=0)} E Wrong log dict format. Check the code comments. ({key}: {dict_to_save[key]})')

    if os.path.exists(log_file_dir):
        df = pd.read_csv(log_file_dir)
        df_new = pd.DataFrame(dict_to_save)
        df = df.append(df_new, sort=False)
    else:
        df = pd.DataFrame(dict_to_save)

    df.to_csv(log_file_dir, index=False)