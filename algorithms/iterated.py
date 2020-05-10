import numpy as np
import copy
from random import sample
from algorithms.local_search import LocalSearch

class Iterated(LocalSearch):

    def __init__(self, n, D, F):
        super(Iterated, self).__init__(n, D, F)
        self.swap_num = max(1, n // 5)

    def run_local(self, method, dlb=True):
        method_name = copy.deepcopy(method)
        method = getattr(LocalSearch, method)
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
                self.cost_fun += result['delta']
                except_fac = result['r']
            else:
                return

    def run(self, method, dlb=True, eye=False, iters=100):
        self.solution = self.initial_solution(eye=eye)
        self.cost_fun = self.cost_function()
        self.run_local(method, dlb=dlb)
        best_solution = copy.deepcopy(self.solution)      # TODO: check deepcopy
        best_cost = self.cost_fun
        for _ in range(iters):
            self.perturbation()
            self.run_local(method, dlb=dlb)
            if self.cost_fun < best_cost:
                best_cost = self.cost_fun
                best_solution = self.solution
        self.solution = best_solution
        self.cost_fun = best_cost
        return np.argsort(self.solution)

    def perturbation(self):
        k = sample(range(self.n), 2 * self.swap_num)
        self.solution[k] = self.solution[np.flip(k)]
        self.cost_fun = self.cost_function()




#
# n = 4
# D = np.array([[0, 22, 53, 53],
#               [22, 0, 40, 62],
#               [53, 40, 0, 55],
#               [53, 62, 55, 0]])
# F = np.array([[0, 3, 0, 2],
#               [3, 0, 0, 1],
#               [0, 0, 0, 4],
#               [2, 1, 4, 0]])
# test = Iterated(n, D, F)
# print('F is: \n', F)
# print('D is: \n', D)
# print('\n', test.run('first_improvement'))
# print(test.cost_fun)


#
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
# test = Iterated(n, D, F)
# print('F is: \n', F)
# print('D is: \n', D)
# print('\n', test.run('best_improvement'))
# print(test.cost_fun)