import numpy as np

def get_data(path):
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

    return n, D, F


if __name__ == '__main__':
    get_data('data/benchmarks/tai20a')
