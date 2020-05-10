from algorithms.local_search import LocalSearch
from algorithms.iterated import Iterated
from algorithms.guided import Guided
from tools.load_data import get_case
import numpy as np
import timeit

paths = ['./data/benchmarks/tai20a', './data/benchmarks/tai40a', './data/benchmarks/tai60a',
         './data/benchmarks/tai80a', './data/benchmarks/tai100a']

for path in paths[0:1]:
    data = get_case(path)
    algo = LocalSearch(*data.values())
    method = 'first_improvement'
    print(method)
    print(path.rsplit('/', maxsplit=1)[-1], ' result:')
    start = timeit.default_timer()
    result = algo.run(method, eye=True)
    end = timeit.default_timer()
    print(result)
    print(f"Cost: {algo.cost_fun}")
    print('Time spent: {:.4f} seconds\n'.format(end-start))

    # method = 'best_improvement'
    # print(method)
    # print(path.rsplit('/', maxsplit=1)[-1], ' result:')
    # start = timeit.default_timer()
    # result = algo.run(method)
    # end = timeit.default_timer()
    # print(result)
    # print(f"Cost: {algo.cost_fun}")
    # print('Time spent: {:.4f} seconds\n'.format(end-start))

    for i in range(100):
        np.random.seed(i)
        algo = Guided(*data.values())
        method = 'first_improvement'
        print('Guided ', method)
        print('i: ', i)
        print(path.rsplit('/', maxsplit=1)[-1], ' result:')
        start = timeit.default_timer()
        end = timeit.default_timer()
        print(result)
        print(f"Cost: {algo.cost_fun}")
        print('Time spent: {:.4f} seconds\n'.format(end - start))