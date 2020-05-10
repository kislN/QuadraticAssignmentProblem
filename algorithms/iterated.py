import numpy as np
import copy
from random import sample
from algorithms.local_search import LocalSearch

class Iterated(LocalSearch):

    def run_local(self, method, dlb=True, itertations=100):
        method_name = copy.deepcopy(method)
        method = getattr(LocalSearch, method)
        except_fac = -1
        while True:
            if 'stochastic_2_opt' in method_name:
                result = method(self, itertations)
            else:
                result = method(self, except_fac, dlb=dlb)
            if result is not None:
                r = result['r']
                s = result['s']
                self.adj_matrix[[r, s]] = self.adj_matrix[[s, r]]
                self.cost_fun += result['delta']
                except_fac = result['r']
            else:
                return np.where(self.adj_matrix.T == 1)[1]

    def run(self, method, dlb=True, eye=True, itertations=100, epoches=10):
        self.adj_matrix = self.initial_solution(self.n, eye=eye)
        self.cost_fun = self.cost_function()
        self.run_local(method, dlb=dlb, itertations=itertations)
        best_adj = copy.deepcopy(self.adj_matrix)
        best_cost = self.cost_fun
        for ind in range(epoches):
            self.perturbation()
            self.run_local(method, dlb=dlb, itertations=itertations)
            if self.cost_fun < best_cost:
                best_cost = self.cost_fun
                best_adj = self.adj_matrix

        self.adj_matrix = best_adj
        self.cost_fun = best_cost
        return np.where(self.adj_matrix.T == 1)[1]


    def perturbation(self, swap_num=1):
        assert 2*swap_num < self.n, 'You are asshole'
        k = sample(range(self.n), 2*swap_num)
        self.adj_matrix[k] = self.adj_matrix[np.flip(k)]
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