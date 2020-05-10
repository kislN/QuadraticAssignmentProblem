import numpy as np
import copy
# np.random.seed(42)

class LocalSearch:

    def __init__(self, n, D, F):
        self.n = n
        self.D = D
        self.F = F
        self.adj_matrix = None
        self.cost_fun = None


    def initial_solution(self, n, eye=False) -> np.array:
        if eye:
            return np.eye(n, dtype=int)
        adj_matrix = np.zeros((n, n), dtype=int)
        indexes = np.arange(n)
        np.random.shuffle(indexes)
        for i, ind in enumerate(indexes):
            adj_matrix[i][ind] = 1
        return adj_matrix


    def cost_function(self) -> int:
        fun_sum = 0
        loc = np.where(self.adj_matrix == 1)[1]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    fun_sum += self.F[i, j] * self.D[loc[i], loc[j]]
        return fun_sum


    def delta_function(self, r, s) -> int:
        fun_sum = 0
        loc = np.where(self.adj_matrix == 1)[1]
        for k in range(self.n):
            if k != r and k != s:
                fun_sum += (self.F[k, r] + self.F[r, k]) * \
                           (self.D[loc[s], loc[k]] - self.D[loc[r], loc[k]]) + \
                           (self.F[k, s] + self.F[s, k]) * \
                           (self.D[loc[r], loc[k]] - self.D[loc[s], loc[k]])
        return fun_sum


    def run(self, method, dlb=True, eye=False, iters=100):
        method_name = copy.deepcopy(method)
        method = getattr(LocalSearch, method)
        self.adj_matrix = self.initial_solution(self.n, eye=eye)
        self.cost_fun = self.cost_function()
        except_fac = -1
        if 'stochastic_2_opt' in method_name:
            method(self, iters)
            return np.where(self.adj_matrix.T == 1)[1]
        else:
            while True:
                result = method(self, except_fac, dlb=dlb)
                if result is not None:
                    r = result['r']
                    s = result['s']
                    self.adj_matrix[[r, s]] = self.adj_matrix[[s, r]]
                    self.cost_fun += result['delta']
                    except_fac = result['r']
                else:
                    return np.where(self.adj_matrix.T == 1)[1]


    def first_improvement(self, except_fac, dlb=False):
        if dlb:
            bits = np.zeros(self.n)
        for r, row in enumerate(self.adj_matrix):
            if r == except_fac:
                continue
            for location in np.where(row == 0)[0]:
                s = np.where(self.adj_matrix[:, location] == 1)[0].item()
                if dlb and bits[s]:
                    continue
                delta = self.delta_function(s, r)
                if delta < 0:
                    return {'delta': delta, 'r': r, 's': s}
            bits[r] = 1
        return None


    def best_improvement(self, except_fac, dlb=False):
        min_delta = 0
        min_result = None
        if dlb:
            bits = np.zeros(self.n)
        for r, row in enumerate(self.adj_matrix):
            if r == except_fac:
                continue
            for location in np.where(row == 0)[0]:
                s = np.where(self.adj_matrix[:, location] == 1)[0].item()
                if dlb and bits[s]:
                    continue
                delta = self.delta_function(s, r)
                if delta < 0 and delta < min_delta:
                    min_delta = delta
                    min_result = {'delta': delta, 'r': r, 's': s}
            if min_result is not None:
                return min_result
            elif dlb:
                bits[r] = 0


    def stochastic_2_opt(self, iters):
        for iter in range(iters):
            a = np.random.randint(low=0, high=self.n-1)
            b = np.random.randint(low=a+1, high=self.n)
            swap_part = np.where(self.adj_matrix.T == 1)[1][a:b+1]
            prev_cost = self.cost_fun
            self.adj_matrix[swap_part] = self.adj_matrix[np.flip(swap_part)]
            self.cost_fun = self.cost_function()
            if prev_cost < self.cost_fun:
                self.adj_matrix[swap_part] = self.adj_matrix[np.flip(swap_part)]
                self.cost_fun = prev_cost
        return None



# n = 4
# D = np.array([[0, 22, 53, 53],
#               [22, 0, 40, 62],
#               [53, 40, 0, 55],
#               [53, 62, 55, 0]])
# F = np.array([[0, 3, 0, 2],
#               [3, 0, 0, 1],
#               [0, 0, 0, 4],
#               [2, 1, 4, 0]])
# test = LocalSearch(n, D, F)
# print('F is: \n', F)
# print('D is: \n', D)
# print(test.cost_fun)
# print('\n', test.run('stochastic_2_opt', itertations=100))
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
# test = LocalSearch(n, D, F)
# print('F is: \n', F)
# print('D is: \n', D)
# print(test.cost_fun)
# print('\n', test.run('stochastic_2_opt', itertations=100))
# print(test.cost_fun)