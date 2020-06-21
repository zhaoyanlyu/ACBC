import numpy as np
from pathos.multiprocessing import Pool
import os

import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import src.sim_tool.evaluate as evaluate
from src.sim_tool.Env import Env
from src.model.MAP import MAP
import src.utility.plot_util as plot_util


def run_MAP(dict_env, env, tau, tot_user=100, num_thread=1):

    map = MAP(tau=tau, tot_chunk=env.tot_chunk, tot_EN=dict_env['tot_EN'],
              mu_xi=env.chunk_mu, sigma_xi=env.chunk_sigma, EN_start=0)
    pof = map.get_price_of_fog()

    # plot_util.fast_plot_MRI(map)

    def one_user_fn(user_id):
        quest_dict = evaluate.user_quests(size_file=dict_env['size_file'], size_chunk=dict_env['size_chunk'],
                                          bw_channel=dict_env['bw_channel'], trace_len=dict_env['trace_len'],
                                          speed_mu=dict_env['speed_mu'], speed_sigma=dict_env['speed_sigma'],
                                          tot_EN=dict_env['tot_EN'])
        hit_arr, \
        tot_hit, \
        tot_miss,\
        PoF, = evaluate.test_quests_on_cache(quest_dict=quest_dict, cache_dict=map.cache_dict,
                                                 tot_EN=dict_env['tot_EN'],
                                                 size_file=dict_env['size_file'], size_chunk=dict_env['size_chunk'])
        avg_hit_elem = tot_hit
        avg_miss_elem = tot_miss
        avg_PoF_elem = PoF

        return [avg_hit_elem, avg_miss_elem, avg_PoF_elem]

    pool = Pool(num_thread)
    outputs = np.asarray(list(pool.map(one_user_fn, range(tot_user))))
    pool.close()
    pool.join()
    avg_hit = np.average(outputs[:, 0])
    avg_miss = np.average(outputs[:, 1])
    avg_PoF = np.average(outputs[:, 2])

    return avg_hit, avg_miss, avg_PoF


if __name__ == '__main__':
    log_path = "../../results/log.csv"

    tau = 0.9

    for idx_test, speed_mu in enumerate(range(15, 34)):  # 15-33
        dict_env = {'speed_mu': speed_mu,
                    'speed_sigma': speed_mu / 8,
                    'bw_channel': 100,
                    'trace_len': 300,
                    'tot_EN': 50,
                    'size_chunk': 8 * 8,
                    'size_file': 5000 * 8,
                    'tot_user': 100}
        env = Env(size_file=dict_env['size_file'],
                  size_chunk=dict_env['size_chunk'],
                  bw_channel=dict_env['bw_channel'],
                  trace_len=dict_env['trace_len'],
                  speed_mu=dict_env['speed_mu'],
                  speed_sigma=dict_env['speed_sigma'])

        avg_hit, avg_miss, avg_PoF = run_MAP(dict_env=dict_env, env=env,
                                             tau=tau, tot_user=dict_env['tot_user'], num_thread=2)

        result_dict = {'algorithm': 'MAP'}
        for key in dict_env.keys():
            result_dict[key] = [dict_env[key]]
        result_dict['tau'] = [tau]
        result_dict['avg_hit'] = [avg_hit]
        result_dict['avg_miss'] = [avg_miss]
        result_dict['avg_PoF'] = [avg_PoF]

        plot_util.log_results(os.path.join(os.getcwd(),
                                           log_path), result_dict)










