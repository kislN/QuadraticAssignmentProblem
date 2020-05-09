import numpy as np

np.random.seed(42)

"""
    r - the first factory
    s - the second factory 
    swap s and r    
    
    adj_matrix - the adjacency matrix of factories and locations
    xaxis - locations
    yaxis - factories
"""

class LocalSearch:
    def __init__(self, n, D, F):
        self.n = n
        self.D = D
        self.F = F
        self.adj_matrix = self.initial_solution(n, eye=True)
        self.cost_fun = self.cost_function()

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
        loc     = np.where(self.adj_matrix==1)[1]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    fun_sum += self.F[i,j] * self.D[loc[i], loc[j]]
        return fun_sum

    def delta_function(self, r, s):
        fun_sum = 0
        loc     = np.where(self.adj_matrix==1)[1]

        for k in range(self.n):
            if k != r and k != s:
                fun_sum += (self.F[k,r] + self.F[r, k]) * \
                           (self.D[loc[s],loc[k]] - self.D[loc[r],loc[k]]) + \
                           (self.F[k,s] + self.F[s, k]) * \
                           (self.D[loc[r],loc[k]] - self.D[loc[s],loc[k]])
        return fun_sum

    def first_improvement(self, d_l_bits=False):
        except_fac = -1
        while(1):
            if d_l_bits:
                bits = np.zeros(self.n)
            flag = 0
            for r, row in enumerate(self.adj_matrix):
                if r == except_fac:
                    continue

                for location in np.where(row==0)[0]:
                    s = np.where(self.adj_matrix[:,location]==1)[0].item()  # factory in the location
                    if d_l_bits and bits[s]:
                        continue

                    delta = self.delta_function(s, r)
                    if delta < 0:
                        self.cost_fun += delta
                        self.adj_matrix[[r, s]] = self.adj_matrix[[s, r]]
                        flag = 1
                        break

                if flag:
                    break
                elif d_l_bits:
                    bits[r] = 1

            if not flag:
                break

        return self.adj_matrix


n = 3
F = np.random.randint(size=(n,n), low=1, high=10)
np.fill_diagonal(F, 0)
D = np.random.randint(size=(n,n), low=1, high=5)
D = (D + D.T)
np.fill_diagonal(D, 0)

n = 4
D = np.array([[0, 22, 53, 53],[22, 0, 40, 62],[53, 40, 0, 55],[53, 62, 55, 0]])
F = np.array([[0, 3, 0,	2],[3, 0, 0, 1],[0,	0, 0, 4],[2,1,4,0]])
test = LocalSearch(n, D, F)
print('F is: \n', F)
print('D is: \n', D)
print(test.cost_fun)

print('\n', test.first_improvement(d_l_bits=True))
print(test.cost_fun)
