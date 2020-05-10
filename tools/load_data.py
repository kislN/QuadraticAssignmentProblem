import numpy as np
from os.path import join
import os

def get_case(path):

    with open(path, 'r') as file:
        data = file.readlines()
        n = int(data.pop(0)[:-1])
        D = []
        F = []
        for _ in range(n):
            string = data.pop(0)[:-1].replace('  ', ' ').split(' ')[1:]
            D.append(np.asarray(string, dtype=np.int32))
        D = np.asarray(D)
        data.pop(0)
        for _ in range(n):
            if _ == n-1:
                string = data.pop(0).replace('  ', ' ').split(' ')[1:]
                F.append(np.asarray(string, dtype=np.int32))
                break
            string = data.pop(0)[:-1].replace('  ', ' ').split(' ')[1:]
            F.append(np.asarray(string, dtype=np.int32))
        F = np.asarray(F)

    dct = { 'n': n,
            'D': D,
            'F': F,
            }

    return dct

def get_all() -> list:
    path = './data/benchmarks'
    cases = []
    for file_name in sorted(os.listdir(path)):
        cases.append(get_case(join(path, file_name)))
        print(file_name, ' is loaded!')
    cases = sorted(cases, key=lambda x: x['n'])
    return cases

