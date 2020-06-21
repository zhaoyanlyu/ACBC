import numpy as np
from pathos.multiprocessing import Pool
import os

import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import src.sim_tool.evaluate as evaluate
from src.sim_tool.Env import Env
from src.model.MAP import MAP
import src.utility.plot_util as plot_util


def run_distributed(dict_env, env, overflow, tot_user=1):
    # Distributed caching algorithm
    assert overflow >= 0
    cache_dict = {}
    tot_chunk = int(np.ceil(dict_env['size_file'] / dict_env['size_chunk']))
    split = tot_chunk // dict_env['tot_EN']
    for index_EN in range(dict_env['tot_EN']):
        iEN_chunk_begin = split * index_EN
        iEN_chunk_end = min(split * (index_EN + 1), tot_chunk)
        chunk_overflow = int(split * overflow)
        if index_EN == 0:  # First EN
            iEN_chunk_end = min(iEN_chunk_end + chunk_overflow // 2, tot_chunk)
        elif index_EN == dict_env['tot_EN'] - 1:  # Last EN
            iEN_chunk_begin = max(iEN_chunk_begin - chunk_overflow // 2, 0)
        else:
            iEN_chunk_begin = max(iEN_chunk_begin - chunk_overflow // 2, 0)
            iEN_chunk_end = min(iEN_chunk_end + chunk_overflow // 2, tot_chunk)

        cache_dict[index_EN] = list(range(int(iEN_chunk_begin), int(iEN_chunk_end)))

    avg_hit = 0
    avg_miss = 0
    avg_PoF = 0

    for user in range(dict_env['tot_user']):
        quest_dict = evaluate.user_quests(size_file=dict_env['size_file'], size_chunk=dict_env['size_chunk'],
                                          bw_channel=dict_env['bw_channel'], trace_len=dict_env['trace_len'],
                                          speed_mu=dict_env['speed_mu'], speed_sigma=dict_env['speed_sigma'],
                                          tot_EN=dict_env['tot_EN'])
        hit_arr, \
        tot_hit, \
        tot_miss, \
        PoF, = evaluate.test_quests_on_cache(quest_dict=quest_dict, cache_dict=cache_dict,
                                             tot_EN=dict_env['tot_EN'],
                                             size_file=dict_env['size_file'], size_chunk=dict_env['size_chunk'])

        avg_hit += tot_hit / dict_env['tot_user']
        avg_miss += tot_miss / dict_env['tot_user']
        avg_PoF += PoF / dict_env['tot_user']

    return avg_hit, avg_miss, avg_PoF



if __name__ == '__main__':
    log_path = "../../results/log.csv"

    overflow = 10.

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

        avg_hit, avg_miss, avg_PoF = run_distributed(dict_env=dict_env, env=env, overflow=overflow, tot_user=dict_env['tot_user'])

        result_dict = {'algorithm': 'distributed'}
        for key in dict_env.keys():
            result_dict[key] = [dict_env[key]]
        result_dict['avg_hit'] = [avg_hit]
        result_dict['avg_miss'] = [avg_miss]
        result_dict['avg_PoF'] = [avg_PoF]
        result_dict['overflow'] = [overflow]

        plot_util.log_results(os.path.join(os.getcwd(),
                                           log_path), result_dict)










