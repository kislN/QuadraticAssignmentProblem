import numpy as np
import copy
import matplotlib.pyplot as plt
# np.random.seed(42)

class LocalSearch:

    def __init__(self, n, D, F):
        self.n = n
        self.D = D
        self.F = F
        self.solution = None
        self.cost_fun = None
        self.plot_list = []

    def run(self, method, eye=False, **kwargs) -> np.array:
        method = getattr(LocalSearch, method)
        self.solution = self.initial_solution(eye=eye)
        self.cost_fun = self.cost_function()
        while True:
            result = method(self, **kwargs)
            if result is not None:
                r = result['r']
                s = result['s']
                self.solution[[r, s]] = self.solution[[s, r]]
                self.cost_fun += result['delta']
                self.plot_list.append(self.cost_fun)
                kwargs['except_fac'] = result['r']
            else:
                return np.argsort(self.solution)

    def initial_solution(self, eye=False) -> np.array:
        if eye:
            return np.arange(self.n, dtype=int)
        return np.random.permutation(np.arange(self.n))

    def cost_function(self) -> int:
        fun_sum = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    fun_sum += self.F[i, j] * self.D[self.solution[i], self.solution[j]]
        return fun_sum

    def delta_function(self, r, s) -> int:
        fun_sum = 0
        for k in range(self.n):
            if k != r and k != s:
                fun_sum += (self.F[k, r] + self.F[r, k]) * \
                           (self.D[self.solution[s], self.solution[k]] -
                            self.D[self.solution[r], self.solution[k]]) + \
                           (self.F[k, s] + self.F[s, k]) * \
                           (self.D[self.solution[r], self.solution[k]] -
                            self.D[self.solution[s], self.solution[k]])
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
                    delta = self.delta_function(r, s)
                    if delta < 0:
                        return {'delta': delta, 'r': r, 's': s}
            bits[r] = 1
        return None

    def best_improvement(self, except_fac=-1, dlb=True):
        min_delta = 0
        min_result = None
        bits = np.zeros(self.n)
        for r in range(self.n):
            if r == except_fac:
                continue
            for s in range(self.n):
                if r != s:
                    if dlb and bits[s]:
                        continue
                    delta = self.delta_function(r, s)
                    if delta < min_delta:
                        min_delta = delta
                        min_result = {'delta': delta, 'r': r, 's': s}
            if min_result is not None:
                return min_result
            else:
                bits[r] = 1
        return None

    def stochastic_2_opt(self, stochastic_iters=100):
        for iter in range(stochastic_iters):
            a = np.random.randint(low=0, high=self.n-1)
            b = np.random.randint(low=a+1, high=self.n)
            swap_part = np.arange(a, b+1)
            prev_cost = self.cost_fun
            prev_solution = self.solution
            self.plot_list.append(self.cost_fun)
            self.solution[swap_part] = self.solution[np.flip(swap_part)]
            self.cost_fun = self.cost_function()
            if prev_cost < self.cost_fun:
                self.solution = prev_solution
                self.cost_fun = prev_cost
        return None

    def plot(self, title='Algo'):
        plt.plot(list(range(len(self.plot_list))), self.plot_list)
        plt.grid()
        plt.title(title)
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()

