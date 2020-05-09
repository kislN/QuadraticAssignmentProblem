from lab3.QuadraticAssignmentProblem.algorithms.my_local_search import LocalSearch
from lab3.QuadraticAssignmentProblem.tools.load_data import get_data
import timeit

paths = ['./data/benchmarks/tai20a', './data/benchmarks/tai40a', './data/benchmarks/tai60a',
         './data/benchmarks/tai80a', './data/benchmarks/tai100a']

for path in paths:
    data = get_data(path)
    algo = LocalSearch(*data)
    print(path.rsplit('/', maxsplit=1)[-1], ' result:')
    start = timeit.default_timer()
    result = algo.run('best_improvement')
    end = timeit.default_timer()
    print(result)
    print(f"Cost: {algo.cost_fun}")
    print('Time spent: {:.4f} seconds\n'.format(end-start))
