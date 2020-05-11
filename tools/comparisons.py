import pandas as pd
import numpy as np

from algorithms.local_search import LocalSearch
from algorithms.iterated import Iterated
from algorithms.guided import Guided
from tools.time_limit import time_limit, TimeoutException


def run_tests(cases, algorithms=[LocalSearch, Iterated, Guided],
             methods=['stochastic_2_opt', 'first_improvement', 'best_improvement'],
             iterations=3, lim_sec=300, file_name='solutions.csv'):

    df = pd.DataFrame(columns=['Method', 'Number of items', 'Best cost', 'Mean cost', 'Best solution'])

    for method in methods:
        for i, case in enumerate(cases):
            answer = case['optimal_profit']

            try:
                with time_limit(lim_sec):
                    method_ans = method(knapsack['capacity'], knapsack['weights'], knapsack['costs'])
                    profit = method_ans[0]
                    weight = method_ans[1]
                    X = method_ans[2]
                    df = df.append(pd.Series([method.__name__, i, knapsack['n'], answer, profit, weight,
                                              X, answer == profit], index=df.columns), ignore_index=True)
            except TimeoutException as e:
                df = df.append(pd.Series([method.__name__, i, knapsack['n'], answer, np.nan, np.nan, np.nan,
                                          str(lim_sec) + ' seconds have passed!'], index=df.columns), ignore_index=True)

    df.to_csv('./data/output/' + file_name)
    return df