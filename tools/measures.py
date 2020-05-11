import time
import pandas as pd
import numpy as np

from algorithms.local_search import LocalSearch
from algorithms.iterated import Iterated
from algorithms.guided import Guided
from tools.load_data import get_case
# from lab3.QuadraticAssignmentProblem.tools.time_limit import time_limit, TimeoutException
from timeit import default_timer
from tqdm import tqdm

def get_local_result(path, method, stochastic_iters=100):
    data = get_case(path)
    path = path.rsplit('/', maxsplit=1)[-1]
    algorithm = LocalSearch(*data.values())
    if method == 'stochastic_2_opt':
        start = default_timer()
        result = algorithm.run(method, stochastic_iters=stochastic_iters)
        end = default_timer()
    else:
        start = default_timer()
        result = algorithm.run(method)
        end = default_timer()

    answer = {
        'file': path,
        'algorithm': type(algorithm).__name__,
        'method': method,
        'result': result,
        'cost': algorithm.cost_fun,
        'time': end - start,
        'experiment': algorithm
    }
    return answer


def get_iterated_result(path, method, epoches, swap_num=None, stochastic_iters=100):
    data = get_case(path)
    path = path.rsplit('/', maxsplit=1)[-1]
    algorithm = Iterated(*data.values(), swap_num)
    if method == 'stochastic_2_opt':
        start = default_timer()
        result = algorithm.run(method, epoches=epoches, stochastic_iters=stochastic_iters)
        end = default_timer()
    else:
        start = default_timer()
        result = algorithm.run(method, epoches=epoches)
        end = default_timer()

    answer = {
        'file': path,
        'algorithm': type(algorithm).__name__,
        'method': method,
        'swap_num': swap_num,
        'epoches': epoches,
        'result': result,
        'cost': algorithm.cost_fun,
        'time': end - start,
        'experiment': algorithm
    }
    return answer


def get_guided_result(path, method, epoches, mu, stochastic_iters=100):
    data = get_case(path)
    path = path.rsplit('/', maxsplit=1)[-1]
    algorithm = Guided(*data.values())
    if method == 'stochastic_2_opt':
        start = default_timer()
        result = algorithm.run(method, epoches=epoches, mu=mu,
                               stochastic_iters=stochastic_iters)
        end = default_timer()
    else:
        start = default_timer()
        result = algorithm.run(method, epoches=epoches, mu=mu)
        end = default_timer()

    answer = {
        'file': path,
        'algorithm': type(algorithm).__name__,
        'method': method,
        'mu': mu,
        'epoches': epoches,
        'result': result,
        'cost': algorithm.cost_fun,
        'time': end - start,
        'experiment': algorithm
    }
    return answer


def get_results(func, iterations, **kwargs):
    cost_list = []
    time_list = []
    result_list = []

    for _ in tqdm(range(iterations)):
        result = func(**kwargs)
        cost_list.append(result['cost'])
        time_list.append(result['time'])
        result_list.append(result['result'])

    result = {
        'costs': np.asarray(cost_list),
        'times': np.asarray(time_list),
        'results': np.asarray(result_list),
        'info': result
    }
    return result


def get_time_table(path, iterations, epoches=None, mu=1):
    df = pd.DataFrame(columns=['file', 'algorithm', 'method', 'mean_time', 'mean_result'])

    algos = [get_local_result, get_iterated_result, get_guided_result]
    methods = ['first_improvement', 'best_improvement', 'stochastic_2_opt']

    iterated_dict = {
        'epoches': epoches
    }

    guided_dict = {
        'mu': mu,
        'epoches': epoches
    }

    for algo in algos:
        for method in methods:
            if algo == get_local_result:
                result = get_results(algo, iterations, method=method, path=path)
            if algo == get_iterated_result:
                result = get_results(algo, iterations, method=method, path=path, **iterated_dict)
            if algo == get_guided_result:
                result = get_results(algo, iterations, method=method, path=path, **guided_dict)

            mean_time = result['times'].mean()
            mean_cost = result['costs'].mean()
            df = df.append(pd.Series([result['info']['file'], result['info']['algorithm'],
                      result['info']['method'], mean_time, int(mean_cost)], index=df.columns),
                      ignore_index=True)
    filename = result['info']['file']
    df.to_csv(f'./data/output/{filename}_time.csv')
    return df


def get_best_table(path, iterations, epoches=None, mu=1):

    df = pd.DataFrame(columns=['file', 'algorithm', 'method', 'best_result'])

    algos = [get_local_result, get_iterated_result, get_guided_result]
    methods = ['first_improvement', 'best_improvement', 'stochastic_2_opt']

    iterated_dict = {
        'epoches': epoches
    }

    guided_dict = {
        'mu': mu,
        'epoches': epoches
    }

    for algo in algos:
        best_result_by_method = []
        best_cost_by_method = []
        for method in methods:
            if algo == get_local_result:
                result = get_results(algo, iterations, method=method, path=path)
            if algo == get_iterated_result:
                result = get_results(algo, iterations, method=method, path=path, **iterated_dict)
            if algo == get_guided_result:
                result = get_results(algo, iterations, method=method, path=path, **guided_dict)

            min_idx = np.argmin(result['costs'])
            best_cost = int(result['costs'][min_idx])
            best_result = result['results'][min_idx]
            best_cost_by_method.append(best_cost)
            best_result_by_method.append(best_result)
            df = df.append(pd.Series([result['info']['file'], result['info']['algorithm'],
                      result['info']['method'], best_cost], index=df.columns),
                      ignore_index=True)

        best_idx = np.argmin(best_cost_by_method)
        print(f'Best cost{best_cost_by_method[best_idx]}')
        best_result = best_result_by_method[best_idx].tolist()
        best_result = " ".join(str(x) for x in best_result)
        name = path.rsplit('/', maxsplit=1)[-1].capitalize()+'.sol'
        with open(f'./data/output/to_send/{name}_{result["info"]["algorithm"]}', 'w') as file:
            file.write(best_result)
        file.close()

    filename = result['info']['file']

    df.to_csv(f'./data/output/{filename}_best.csv')
    return df

