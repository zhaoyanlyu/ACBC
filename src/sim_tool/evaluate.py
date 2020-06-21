import numpy as np


def get_time_dwell_in_ENi(speed_mu, speed_sigma, trace_len):
    """

    :param speed_mu: m/s
    :param speed_sigma: m^2/s^2
    :param trace_len: m
    :return: time_dwell: s
    """

    speed = np.random.normal(loc=speed_mu, scale=speed_sigma)
    if speed <= 5:
        speed = 5
    time_dwell = trace_len / speed
    return time_dwell


def user_quests(size_file, size_chunk,
                bw_channel, trace_len,
                speed_mu, speed_sigma,
                tot_EN):

    quest_dict = {i: [] for i in range(tot_EN)}
    tot_chunk = int(np.ceil(size_file/size_chunk))

    chunk_begin = 0
    for index_EN in range(tot_EN):
        time_dwell = get_time_dwell_in_ENi(speed_mu, speed_sigma, trace_len)
        num_chunk_quest = time_dwell * bw_channel // size_chunk
        chunk_end = chunk_begin + int(num_chunk_quest)
        if chunk_end >= tot_chunk:
            chunk_end = tot_chunk
        chunk_quest = list(range(chunk_begin, chunk_end))
        quest_dict[index_EN] = chunk_quest
        chunk_begin = chunk_end
        if chunk_begin == int(tot_chunk):
            break
    return quest_dict


def test_quests_on_cache(quest_dict, cache_dict,
                         tot_EN,
                         size_file, size_chunk):
    assert len(quest_dict.keys()) == len(cache_dict.keys())
    assert len(quest_dict.keys()) == tot_EN

    tot_chunk = int(np.ceil(size_file / size_chunk))
    # Count hits
    hit_arr = np.zeros(tot_chunk)
    tot_hit = 0
    for index_EN in range(tot_EN):
        for quest_chunk in quest_dict[index_EN]:
            if quest_chunk in cache_dict[index_EN]:
                hit_arr[quest_chunk] = 1
                tot_hit += 1
    tot_miss = tot_chunk - tot_hit

    # PoF
    last_cached_chunk_id = 0
    copies_of_chunk = 0
    for index_EN in cache_dict.keys():
        num_chunk_on_ENi = len(cache_dict[index_EN])
        copies_of_chunk += num_chunk_on_ENi
        if num_chunk_on_ENi != 0:
            last_cached_chunk_id = cache_dict[index_EN][-1]

    PoF = copies_of_chunk/(last_cached_chunk_id+1)

    return hit_arr, tot_hit, tot_miss, PoF


if __name__ == '__main__':
    print(get_time_dwell_in_ENi(24, 2.4, 400))
    print(user_quests(size_file=5000*8, size_chunk=10*8,
                      bw_channel=100, trace_len=400,
                      speed_mu=20, speed_sigma=2,
                      tot_EN=25))




