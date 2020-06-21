import numpy as np
from pathos.multiprocessing import Pool
import os

import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import src.sim_tool.evaluate as evaluate
from src.sim_tool.Env import Env
from src.model.RICH import RICH
import src.utility.plot_util as plot_util


def run_RICH(dict_env, env, tau_begin, tau_decay_step, tau_end, tot_user=1, num_thread=1):

    def one_user_fn(user_id):
        quest_dict = evaluate.user_quests(size_file=dict_env['size_file'], size_chunk=dict_env['size_chunk'],
                                          bw_channel=dict_env['bw_channel'], trace_len=dict_env['trace_len'],
                                          speed_mu=dict_env['speed_mu'], speed_sigma=dict_env['speed_sigma'],
                                          tot_EN=dict_env['tot_EN'])
        cache_dict = {i: [] for i in range(dict_env['tot_EN'])}

        EN_start = 0
        chunk_start = 0
        num_feedback = 0
        finish_code = 0

        # RICH algorithm
        while(True):
            map = RICH(tau_begin=tau_begin, tau_decay_step=tau_decay_step, tau_end=tau_end,
                       tot_chunk=env.tot_chunk-chunk_start, tot_EN=dict_env['tot_EN']-EN_start,
                      mu_xi=env.chunk_mu, sigma_xi=env.chunk_sigma, EN_start=0)
            FEN = map.get_lastEN()
            for index_EN in range(FEN):
                cache_dict[index_EN+EN_start] = [i+chunk_start for i in map.cache_dict[index_EN]]
            if map.last_chunk == map.tot_chunk:  # Prediction finish.
                finish_code = 0  # All chunk cached
                break
            elif map.get_lastEN() == dict_env['tot_EN']:
                finish_code = 1  # Run out of EN
                break
            elif len(quest_dict[FEN+EN_start]) == 0:
                finish_code = 2  # Cached on more ENs than required but still not finish predicting all chunks.
                break
            else:
                num_feedback += 1
                chunk_start = quest_dict[FEN+EN_start][-1] + 1
                EN_start += FEN+1

        hit_arr, \
        tot_hit, \
        tot_miss,\
        PoF = evaluate.test_quests_on_cache(quest_dict=quest_dict, cache_dict=cache_dict,
                                                 tot_EN=dict_env['tot_EN'],
                                                 size_file=dict_env['size_file'], size_chunk=dict_env['size_chunk'])

        avg_hit_elem = tot_hit
        avg_miss_elem = tot_miss
        avg_num_feedback_elem = num_feedback
        avg_PoF_elem = PoF

        return [avg_hit_elem, avg_miss_elem, avg_PoF_elem, avg_num_feedback_elem, finish_code]
        # print(tot_hit)
        # print(tot_miss)

    pool = Pool(num_thread)
    outputs = np.asarray(list(pool.map(one_user_fn, range(tot_user))))
    pool.close()
    pool.join()
    avg_hit = np.average(outputs[:, 0])
    avg_miss = np.average(outputs[:, 1])
    avg_PoF = np.average(outputs[:, 2])
    avg_num_feedback = np.average(outputs[:, 3])
    finish_code_list = np.array(outputs[:, 4]).reshape([-1])

    return avg_hit, avg_miss, avg_PoF, avg_num_feedback, finish_code_list


if __name__ == '__main__':
    log_path = "../../results/log.csv"

    tau_begin = 0.9
    tau_decay_step = 0.1
    tau_end = 0.7

    for idx_test, speed_mu in enumerate(range(15, 34)):
        dict_env = {'speed_mu': speed_mu,
                    'speed_sigma': speed_mu / 8,
                    'bw_channel': 100,
                    'trace_len': 300,
                    'tot_EN': 50,
                    'size_chunk': 8 * 8,
                    'size_file': 5000 * 8,
                    'tot_user': 1000}

        env = Env(size_file=dict_env['size_file'],
                  size_chunk=dict_env['size_chunk'],
                  bw_channel=dict_env['bw_channel'],
                  trace_len=dict_env['trace_len'],
                  speed_mu=dict_env['speed_mu'],
                  speed_sigma=dict_env['speed_sigma'])

        avg_hit, \
        avg_miss,\
        avg_PoF, \
        avg_num_feedback,\
        finish_code_list = run_RICH(dict_env=dict_env, env=env,
                                    tau_begin=tau_begin, tau_decay_step=tau_decay_step, tau_end=tau_end,
                                    tot_user=dict_env['tot_user'], num_thread=50)

        result_dict = {'algorithm': 'RICH'}
        for key in dict_env.keys():
            result_dict[key] = [dict_env[key]]
        result_dict['tau_begin'] = [tau_begin]
        result_dict['tau_decay_step'] = [tau_decay_step]
        result_dict['tau_end'] = [tau_end]
        result_dict['avg_hit'] = [avg_hit]
        result_dict['avg_miss'] = [avg_miss]
        result_dict['avg_PoF'] = [avg_PoF]
        result_dict['avg_num_feedback'] = [avg_num_feedback]
        result_dict['finish_code_list'] = finish_code_list

        plot_util.log_results(os.path.join(os.getcwd(),
                                           log_path), result_dict)
