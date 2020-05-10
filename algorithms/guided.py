import numpy as np
import copy
from algorithms.local_search import LocalSearch

class Guided(LocalSearch):

    def unit_cost(self, adj) -> np.array:
        unit_costs = []
        loc = np.where(adj == 1)[1]
        for factory in range(self.n):
            fun_sum = 0
            for i in range(self.n):
                if factory != i:
                    fun_sum += self.F[factory, i] * self.D[loc[factory], loc[i]]
            unit_costs.append(fun_sum)
        return np.asarray(unit_costs)


    def fine(self, mu, adj, phi) -> int:
        fine_value = 0
        for fac in range(self.n):
            for loc in range(self.n):
                fine_value += adj[fac,loc] * phi[fac]
        return mu * fine_value


    def utility(self, unit_costs, phi) -> np.array:
        U = []
        for i in range(self.n):
            U.append(unit_costs[i] / (1 + phi[i]))
        return np.asarray(U)


    def run_local(self, method, dlb=True):
        method = getattr(Guided, method)
        except_fac = -1
        while True:
            result = method(self, except_fac, dlb=dlb)
            if result is not None:
                except_fac = result['r']
            else:
                return 0


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
                prev_cost = self.cost_fun
                prev_adj = self.adj_matrix
                self.adj_matrix[[r, s]] = self.adj_matrix[[s, r]]
                self.cost_fun = self.cost_function()
                if prev_cost > self.cost_fun:
                    return {'r': r}
                else:
                    self.adj_matrix = prev_adj
                    self.cost_fun = prev_cost
            if dlb:
                bits[r] = 1
        return None


    def run(self, method, dlb=True, epoches=10, mu=0.1):
        self.adj_matrix = self.initial_solution(self.n)
        self.cost_fun = self.cost_function()
        penalty = np.zeros(self.n, dtype=int)
        best_adj = self.adj_matrix
        best_cost = self.cost_fun
        # unit_costs = self.unit_cost(self.adj_matrix)

        for _ in range(epoches):
            prev_cost = self.cost_fun
            fine_value = self.fine(mu, self.adj_matrix, penalty)
            self.cost_fun += fine_value
            self.run_local(method)
            print(self.cost_fun)
            if self.cost_fun == prev_cost + fine_value:
                self.cost_fun = prev_cost
            unit_costs = self.unit_cost(self.adj_matrix)
            if best_cost > self.cost_fun:
                best_cost = self.cost_fun
                best_adj = copy.deepcopy(self.adj_matrix)

            U = self.utility(unit_costs, penalty)
            # loc = np.where(self.adj_matrix == 1)[1]
            max_fac = np.argmax(U)
            penalty[max_fac] += 1

        self.adj_matrix = best_adj
        self.cost_fun = best_cost
        return np.where(self.adj_matrix.T == 1)[1]




n = 5
D = np.array([[0, 50, 50, 94, 50],
              [50, 0, 22, 50, 36],
              [50, 22, 0, 44, 14],
              [94, 50, 44, 0, 50],
              [50, 36, 14, 50, 0]])
F = np.array([[0, 0, 2, 0, 3],
              [0, 0, 0, 3, 0],
              [2, 0, 0, 0, 0],
              [0, 3, 0, 0, 1],
              [3, 0, 0, 1, 0]])
test = Guided(n, D, F)
print('F is: \n', F)
print('D is: \n', D)
print('\n', test.run('first_improvement'))
print(test.cost_fun)