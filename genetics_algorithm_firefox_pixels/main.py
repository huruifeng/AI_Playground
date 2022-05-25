##########################
## Ruifeng Hu
## 05-11-2022
## Lexington, MA
## hurufieng.cn@hotmail.com
##########################

##--------------------------------------------
# Using the Genetic Algorithm(GA) to draw
#   the FireFox logo
# https://www.cfanz.cn/resource/detail/ZzmQvALXBMEDN
##------------------------------------------
import random
import pickle
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def comp_similarity(indv, target):
    """计算相似度，计算两个array的差值平方和
    :param indv: 图片像素
    :param target: 目标图片像素
    """
    score = np.sum(np.square((indv - target)))
    return score

def save_img(pixel, out):
    fig = plt.figure(figsize=(5.3,5.3))
    plt.axis("off")
    plt.tight_layout()
    plt.imsave(out, arr=pixel)
    plt.close()


def init_genes(id, x, y, z):
    """初始化个体基因
    :param id: 一个数字，与其他初始化相区别
    :param x: 图片高
    :param y: 图片长
    :param z: 4表示rgba, 3表示rgb
    """
    np.random.seed(id)
    pixel = np.random.random((x, y, z))
    return pixel


def init_indv(id, x, y, z, target):
    """初始化个体数据，初始话像素基因，与目标相似度分数
    :param id: 一个数字，与其他初始化相区别
    :param x: 图片高
    :param y: 图片长
    :param z: 4表示rgba, 3表示rgb
    :param target: 目标像素
    """
    pixel = init_genes(id, x, y, z)
    score = comp_similarity(pixel, target)
    indv = {}
    indv['score'] = score
    indv["gene"] = pixel
    return indv


def init_pop(target, p=15, jobs=5):
    """初始化群体数据
    :param target: 目标像素
    :param p: 群体大小
    :param jobs: 进程数
    """
    x, y, z = target.shape
    f_init = partial(init_indv, x=x, y=y, z=z,target=target)
    with Pool(jobs) as pl:
        for i,v in enumerate(pl.map(f_init, list(range(p)))):
            data_pool[i] = v


def breed(p1, p2, mutation, width=5):
    """初始化群体数据
    :param p1: 个体1
    :param p2: 个体2
    :param mutation: 突变率
    :param width: 变异宽度
    """
    x, y, z = p1.shape
    new_p = p1.copy()
    x1_idx = random.sample(range(x), int(x/2))
    y1_idx = random.sample(range(y), int(y/2))
    x2_idx = list(set(range(x)) - set(x1_idx))
    y2_idx = list(set(range(y)) - set(y1_idx))

    temp1 = p2[x1_idx]
    temp1[:,y1_idx] = p1[x1_idx][:, y1_idx]
    new_p[x1_idx] = temp1

    temp2 = p2[x2_idx]
    temp2[:,y2_idx] = p1[x2_idx][:, y2_idx]
    new_p[x2_idx] = temp2

    m_x = random.sample(range(x), int(x*mutation))
    m_y = random.sample(range(y), int(x*mutation))

    indv_m = new_p.copy()
    for i,j in zip(m_x, m_y):
        channel = random.randint(0,z-1)
        center_p = new_p[i,j][channel]
        sx = list(range(max(0, i-width), min(i+width, x-1)))
        sy = list(range(max(0, j-width), min(j+width, y-1)))
        mtemp = indv_m[sx]
        normal_rgba = np.random.normal(center_p,.01, size=mtemp[:,sy].shape[:2])
        normal_rgba[normal_rgba>1] = 1
        normal_rgba[normal_rgba<0] = 0
        mtemp[:,sy, channel] = normal_rgba
        indv_m[sx] = mtemp
    return new_p, indv_m

def crossover(g, target, pair, mutation):
    pi = data_pool.keys()
    males = random.sample(pi, int(len(data_pool)/2))
    females = set(pi) - set(males)
    mm = random.sample(males, pair)
    fm = random.sample(females, pair)
    f_mate = lambda pair: breed(data_pool[pair[0]]["gene"], data_pool[pair[1]]["gene"], mutation)
    for idx, sps in enumerate(map(f_mate, zip(mm,fm))):
        n_indv = f"g{g}_{idx}_n"
        m_indv = f"g{g}_{idx}_m"
        data_pool[n_indv] = {}
        data_pool[m_indv] = {}
        data_pool[n_indv]["gene"] = sps[0]
        data_pool[n_indv]["score"] = comp_similarity(sps[0], target)
        data_pool[m_indv]["gene"] = sps[1]
        data_pool[m_indv]["score"] = comp_similarity(sps[1], target)

def evol(godie):
    scores = [(k, v['score']) for k, v in data_pool.items()]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for i in range(godie):
        del data_pool[scores[i][0]]
    best_id, best_score = scores[-1][0], scores[-1][1]
    best_indv = data_pool[best_id]
    return best_id, best_indv

def main(args):
    pkl = args.evol_info # 50000 + 20000
    img = args.img
    p = args.population
    pair = args.pair
    generations = args.generation
    mutation = args.mutation
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    global data_pool
    data_pool = {}
    einfo = {"data": data_pool, "g": 0}
    target = mpimg.imread(img)
    if args.c:
        h =  open(pkl, "rb")
        einfo = pickle.load(h)
        data_pool = einfo["data"]
        h.close()
    else:
        init_pop(target, p=p, jobs=5)
    fcross = partial(crossover, target=target, pair=pair, mutation=mutation)
    for i in range(0+einfo["g"], einfo["g"]+500001):
        if i % 5000 == 0:
            einfo["data"] = data_pool
            einfo["g"] = i
            with open(pkl, "wb") as h:
                pickle.dump(einfo, h)
        fcross(i)
        best_id, best_indv = evol(pair*2)
        print(i, best_id, best_indv['score'])
        if i % 500 == 0:
            pixel = best_indv['gene']
            out = f"{outdir}/{best_id}_{best_indv['score']:.1f}.png"
            save_img(pixel, out)

if __name__ == '__main__':
    main()