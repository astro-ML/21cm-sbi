import concurrent.futures
import math
from time import sleep
from concurrent.futures import as_completed
from multiprocessing import get_context
import torch
import numpy as np

test = [10,20,30,40,50,60,70]

def is_prime(n, id):
    sleep(1)
    return np.random.rand(n), id

def afterstuff(res, id):
    return np.sum(res), id

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(is_prime, p, id) for id,p in enumerate(test)]
        for fut in as_completed(futures): 
            res, id = fut.result()
            res, id = afterstuff(res, id)
            print(res,id)


if __name__ == '__main__':
    main()