from algorithms.local_search import LocalSearch
from algorithms.iterated import Iterated
from algorithms.guided import Guided
from tools.load_data import get_case
import timeit

paths = ['./data/benchmarks/tai20a', './data/benchmarks/tai40a', './data/benchmarks/tai60a',
         './data/benchmarks/tai80a', './data/benchmarks/tai100a']

for path in paths:
    data = get_case(path)
    algo = Guided(*data.values())
    print(path.rsplit('/', maxsplit=1)[-1], ' result:')
    start = timeit.default_timer()
    result = algo.run('first_improvement')
    end = timeit.default_timer()
    print(result)
    print(f"Cost: {algo.cost_fun}")
    print('Time spent: {:.4f} seconds\n'.format(end-start))
