import numpy as np
import copy
from algorithms.local_search import LocalSearch

class Guided(LocalSearch):

    def __init__(self, n, D, F):
        super(Guided, self).__init__(n, D, F)
        self.alpha = None
        self.mu = None
        self.penalty = np.zeros((self.n, self.n))

    def run(self, method, eye=False, epoches=100, mu=0.1, **kwargs) -> np.array:
        self.mu = mu
        self.solution = self.initial_solution(eye=eye)
        self.cost_fun = self.cost_function()
        self.alpha = self.calculate_alpha()
        self.plot_list.append(self.cost_fun)
        best_solution = copy.deepcopy(self.solution)
        best_cost_fun = self.cost_fun
        for _ in range(epoches):
            self.run_local(method, **kwargs)
            self.plot_list.append(self.cost_fun)
            if best_cost_fun > self.cost_fun:
                best_cost_fun = self.cost_fun
                best_solution = copy.deepcopy(self.solution)
            utils = self.utils()
            max_fac = np.argmax(utils)
            self.penalty[max_fac, self.solution[max_fac]] += 1
        self.solution = best_solution
        self.cost_fun = best_cost_fun
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
                kwargs['except_fac'] = result['r']
            else:
                self.cost_fun = self.cost_function()
                return None

    def calculate_alpha(self) -> float:
        sum_d = 0
        sum_f = 0
        for i in range(self.n):
            for j in range(self.n):
                sum_d += self.D[self.solution[i], self.solution[j]]
                sum_f += self.F[i,j]
        return (sum_d * sum_f) / self.n**4

    def feature_cost(self, feature) -> int:
        fun_sum = 0
        for i in range(self.n):
            fun_sum += self.F[feature, i] * self.D[self.solution[feature], self.solution[i]]
        return fun_sum

    def util(self, feature) -> float:
        return self.feature_cost(feature) / (1 + self.penalty[feature, self.solution[feature]])

    def utils(self) -> np.array:
        return np.array([self.util(x) for x in range(self.n)])

    def delta_h(self, r, s) -> int:
        fun_sum = 0
        for k in range(self.n):
            if k != r and k != s:
                fun_sum += (self.F[k, r] + self.F[r, k]) * \
                           (self.D[self.solution[s], self.solution[k]] -
                            self.D[self.solution[r], self.solution[k]]) + \
                           (self.F[k, s] + self.F[s, k]) * \
                           (self.D[self.solution[r], self.solution[k]] -
                            self.D[self.solution[s], self.solution[k]])
        fun_sum += self.mu * self.alpha * (self.penalty[r, self.solution[s]] +
                                           self.penalty[s, self.solution[r]] -
                                           self.penalty[r, self.solution[r]] -
                                           self.penalty[s, self.solution[s]])
        return fun_sum

    def first_improvement(self, except_fac=-1, dlb=True):
        bits = np.zeros(self.n)
        for r in range(self.n):
            if r == except_fac:
                continue
            for s in range(self.n):
                if r != s:
                    if dlb and bits[s]:
                        continue
                    delta_h = self.delta_h(r, s)
                    if delta_h < 0:
                        return {'r': r, 's': s}
            bits[r] = 1
        return None

    def best_improvement(self, except_fac=-1, dlb=True):
        min_delta_h = 0
        min_result = None
        bits = np.zeros(self.n)
        for r in range(self.n):
            if r == except_fac:
                continue
            for s in range(self.n):
                if r != s:
                    if dlb and bits[s]:
                        continue
                    delta_h = self.delta_h(r, s)
                    if delta_h < min_delta_h:
                        min_delta_h = delta_h
                        min_result = {'r': r, 's': s}
            if min_result is not None:
                return min_result
            else:
                bits[r] = 1
        return None

