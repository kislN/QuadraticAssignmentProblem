import numpy as np
import copy
from algorithms.local_search import LocalSearch

class Guided(LocalSearch):
    def __init__(self, n, D, F):
        super(Guided, self).__init__(n, D, F)
        self.alpha = None
        self.penalty = np.zeros((self.n, self.n))
        self.mu = None

    def first_improvement(self, except_fac, dlb=False):
        if dlb:
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

    def best_improvement(self, except_fac, dlb=False):
        min_delta_h = 0
        min_result = None
        if dlb:
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

    def run_local(self, method, dlb=True):
        method_name = copy.deepcopy(method)
        method = getattr(Guided, method)
        except_fac = -1
        while True:
            if 'stochastic_2_opt' in method_name:
                result = method(self, iters=100)
            else:
                result = method(self, except_fac, dlb=dlb)
            if result is not None:
                r = result['r']
                s = result['s']
                self.solution[[r, s]] = self.solution[[s, r]]
                except_fac = result['r']
            else:
                self.cost_fun = self.cost_function()
                return


    def run(self, method, dlb=True, eye=False, iters=100, mu=0.1):
        self.mu = mu
        self.solution = self.initial_solution(eye=eye)
        self.cost_fun = self.cost_function()
        self.alpha = self.calculate_alpha()
        best_solution = copy.deepcopy(self.solution)
        best_cost_fun = self.cost_fun
        for _ in range(iters):
            self.run_local(method, dlb=dlb)
            if best_cost_fun > self.cost_fun:
                best_cost_fun = copy.deepcopy(self.cost_fun)
                best_solution = copy.deepcopy(self.solution)
            utils = self.utils()
            max_fac = np.argmax(utils)
            self.penalty[max_fac, self.solution[max_fac]] += 1
        self.solution = best_solution
        self.cost_fun = best_cost_fun
        return np.argsort(self.solution)

    def calculate_alpha(self):
        sum_d = 0
        sum_f = 0
        for i in range(self.n):
            for j in range(self.n):
                sum_d += self.D[self.solution[i], self.solution[j]]
                sum_f += self.F[i,j]
        return (sum_d * sum_f) / self.n**4

    def feature_cost(self, feature):
        fun_sum = 0
        for i in range(self.n):
            fun_sum += self.F[feature, i] * self.D[self.solution[feature], self.solution[i]]
        return fun_sum

    def util(self, feature):
        return self.feature_cost(feature) / (1 + self.penalty[feature, self.solution[feature]])

    def utils(self):
        return np.array([self.util(x) for x in range(self.n)])

    def delta_h(self, r, s) -> int:
        fun_sum = 0
        for k in range(self.n):
            if k != r and k != s:
                fun_sum += (self.F[k, r] + self.F[r, k]) * \
                           (self.D[self.solution[s], self.solution[k]] - self.D[self.solution[r], self.solution[k]]) + \
                           (self.F[k, s] + self.F[s, k]) * \
                           (self.D[self.solution[r], self.solution[k]] - self.D[self.solution[s], self.solution[k]])
        fun_sum += self.mu * self.alpha * (self.penalty[r, self.solution[s]] + self.penalty[s, self.solution[r]] -
                                      self.penalty[r, self.solution[r]] - self.penalty[s, self.solution[s]])
        return fun_sum

# n = 5
# D = np.array([[0, 50, 50, 94, 50],
#               [50, 0, 22, 50, 36],
#               [50, 22, 0, 44, 14],
#               [94, 50, 44, 0, 50],
#               [50, 36, 14, 50, 0]])
# F = np.array([[0, 0, 2, 0, 3],
#               [0, 0, 0, 3, 0],
#               [2, 0, 0, 0, 0],
#               [0, 3, 0, 0, 1],
#               [3, 0, 0, 1, 0]])
# test = Guided(n, D, F)
# print('F is: \n', F)
# print('D is: \n', D)
# print('\n', test.run('best_improvement', mu=1))
# print(test.cost_fun)