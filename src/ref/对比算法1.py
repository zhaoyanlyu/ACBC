"""
有待适配
"""

import numpy as np

#-*-coding:utf-8
"""
 每个函数只需要传入请求队列，缓存容量大小即可获得请求命中率
"""

#FIFO算法
def fifo(seq, size):
    bs = []
    hit = 0
    for page in seq:
        if page not in bs:
            if len(bs) < size:
                bs.append(page)
            else:
                bs.pop(0)
                bs.append(page)
        else:
            hit+=1
    print("FIFO内存容量为{}时的命中率为{:.2%}".format(size,float(hit)/len(seq)))
    return bs


def lru(seq,size):
    ps = []
    hit=0
    for page in seq:
        if page not in ps:
            if len(ps) < size:
                ps.append(page)
            else:
                ps.pop(0)
                ps.append(page)
        else:
            hit += 1
            ps.append(ps.pop(ps.index(page)))  # 弹出后插入到最近刚刚访问的一端
    print("LRU内存容量为{}时的命中率为{:.2%}".format(size, float(hit) /len(seq)))
    return ps

# def lfu(seq,size):
#     """
#     LFU（Least Frequently Used）最近最少使用算法。它是基于“如果一个数据在最近一段时间内使用次数很少，
#     那么在将来一段时间内被使用的可能性也很小”的思路。
#     """
#     ps={}
#     hit=0
#     for i, page in enumerate(seq):
#         if page not in ps:
#             if len(ps) < size:  # 内存还未满
#                 ps[page] = 1
#             else:
#                 data = pd.Series(ps)
#                 data = data.sort_values()
#                 min_index = data.keys()[0]
#                 ps.pop(min_index)
#                 ps[page] = 1
#         else:
#             ps[page] += 1
#             hit+=1
#     print("LFU内存为{}k时的命中率为{:.2%}".format(size, float(hit) /len(seq)))
#     return ps.keys()
def lfu(seq,size):
    """
    LFU（Least Frequently Used）最近最少使用算法。它是基于“如果一个数据在最近一段时间内使用次数很少，
    那么在将来一段时间内被使用的可能性也很小”的思路。
    """
    ps,bad, bad_i = {},1 << 31 - 1, 0
    hit=0
    for i, page in enumerate(seq):
        if page not in ps:
            if len(ps) < size:  # 内存还未满
                ps[page] = 1
            else:
                for j, v in ps.items():
                    if v < bad:
                        bad, bad_i = v, j
                ps.pop(bad_i)
                ps[page] = 1
                bad, bad_i = 2 ** 32 - 1, 0
        else:
            ps[page] += 1
            hit+=1
    print("LFU内存容量为{}时的命中率为{:.2%}".format(size, float(hit)/len(seq)))


if __name__ == "__main__":
    # 加载数据
    # 这里举例子说明：有1000各请求【1-1000】缓存容量大小为200
    Request = [i for i in range(1, 1001)]
    print(Request)

    tot_chunk = 500
    tot_user = 10
    full_quest_list = []
    for index_user in range(tot_user):
        start_point = int(np.random.rand() * tot_chunk)


    F_BS=fifo(Request,200)
    Lru_BS =lru(Request,200)
    Lfu_BS = lfu(Request,200)

