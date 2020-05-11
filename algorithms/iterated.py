import numpy as np
import copy
from random import sample
from algorithms.local_search import LocalSearch


class Iterated(LocalSearch):
    def __init__(self, n, D, F, swap_num=None):
        super(Iterated, self).__init__(n, D, F)
        if swap_num is None:
            self.swap_num = max(1, n // 5)
        else:
            self.swap_num = swap_num

    def run(self, method, eye=False, epoches=100, **kwargs) -> np.array:
        self.solution = self.initial_solution(eye=eye)
        self.cost_fun = self.cost_function()
        self.plot_list.append(self.cost_fun)
        self.run_local(method, **kwargs)
        self.plot_list.append(self.cost_fun)
        best_solution = copy.deepcopy(self.solution)
        best_cost = self.cost_fun
        for _ in range(epoches):
            self.perturbation()
            self.run_local(method, **kwargs)
            self.plot_list.append(self.cost_fun)
            if self.cost_fun < best_cost:
                best_cost = self.cost_fun
                best_solution = self.solution
        self.solution = best_solution
        self.cost_fun = best_cost
        self.plot_list.append(self.cost_fun)
        return np.argsort(self.solution)

    def run_local(self, method, **kwargs):
        method = getattr(LocalSearch, method)
        while True:
            result = method(self, **kwargs)
            if result is not None:
                r = result['r']
                s = result['s']
                self.solution[[r, s]] = self.solution[[s, r]]
                self.cost_fun += result['delta']
                kwargs['except_fac'] = result['r']
            else:
                return None

    def perturbation(self):
        k = sample(range(self.n), 2 * self.swap_num)
        self.solution[k] = self.solution[np.flip(k)]
        self.cost_fun = self.cost_function()
