import time
import pandas as pd
import numpy as np

from algorithms.local_search import LocalSearch
from algorithms.iterated import Iterated
from algorithms.guided import Guided
from tools.time_limit import time_limit, TimeoutException


def get_time(cases, algorithms=[LocalSearch, Iterated, Guided],
             methods=['stochastic_2_opt', 'first_improvement', 'best_improvement'],
             iterations=3, lim_sec=300, file_name='time.csv'):
    df = pd.DataFrame(columns=['Method', 'Number of items', 'Mean', 'Median', 'Min', 'Max', 'Variance'])
    for i, case in enumerate(cases):
        for algo in algorithms:
            algorithm = algo(*case.values())
            for method in methods:
                time_list = []
                flag = 1
                for _ in range(iterations):
                    try:
                        with time_limit(lim_sec):
                            t0 = time.time()
                            algorithm.run(method)
                            t1 = time.time()
                            time_list.append(t1 - t0)
                    except TimeoutException as e:
                        df = df.append(pd.Series([algo.__name__ + '/' + method, case['n'], str(lim_sec) + ' seconds have passed!',
                                                  np.nan, np.nan, np.nan, np.nan], index=df.columns), ignore_index=True)
                        flag = 0
                        break
                if flag:
                    time_list = np.array(time_list)
                    df = df.append(pd.Series([algo.__name__ + '/' + method, case['n'], np.mean(time_list), np.median(time_list),
                                              np.min(time_list), np.max(time_list), np.var(time_list)],
                                             index=df.columns), ignore_index=True)

    df.to_csv('./data/output/' + file_name)
    return df

