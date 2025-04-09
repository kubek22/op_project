import numpy as np


class BRBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.a = np.zeros(self.n_visible)
        self.b = np.zeros(self.n_hidden)
        self.W = np.zeros((self.n_visible, self.n_hidden))
        # self.Z = self.compute_Z()

    def energy(self, v, h):
        return - (v @ self.a + h @ self.b + v @ self.W @ h)

    @staticmethod
    def get_combinations(n):
        combinations = []
        v = np.zeros(n)
        def _get_combinations(v, idx):
            if idx == len(v):
                combinations.append(v)
                return
            v0 = v.copy()
            v0[idx] = 0
            _get_combinations(v0, idx + 1)
            v1 = v.copy()
            v1[idx] = 1
            _get_combinations(v1, idx + 1)

        _get_combinations(v, 0)
        return np.array(combinations)

    def compute_Z(self):
        v_comb = self.get_combinations(self.n_visible)
        h_comb = self.get_combinations(self.n_hidden)
        z = 0
        for v in v_comb:
            for h in h_comb:
                E = self.energy(v, h)
                z += np.exp(-E)
        return z

    # def probability(self, v, h):
    #     E = self.energy(v, h)
    #     return np.exp(-E) / self.Z

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def h_probability(self, v):
        return self.sigmoid(self.b  + v @ self.W)

    def v_probability(self, h):
        return self.sigmoid(self.a  + self.W @ h)

    def draw_hidden(self, v):
        h = np.zeros(self.n_hidden)
        p = self.h_probability(v)
        for i in range(self.n_hidden):
            h[i] = np.random.binomial(1, p[i])
        return h

    def draw_visible(self, h):
        v = np.zeros(self.n_visible)
        p = self.v_probability(h)
        for i in range(self.n_visible):
            v[i] = np.random.binomial(1, p[i])
        return v

    def gibbs_sampling(self, n, h):
        for i in range(n):
            v = self.draw_visible(h)
            h = self.draw_hidden(v)
        return v, h

    def fit(self, V, iterations, learning_rate, cd_n=1):
        for i in range(iterations):
            # gradient descent
            for v in V:
                h = self.draw_hidden(v)
                v_cd, h_cd = self.gibbs_sampling(cd_n, h)

                W_update = learning_rate * (np.array([v]).T @ np.array([h]) - np.array([v_cd]).T @ np.array([h_cd]))
                a_update = learning_rate * (v - v_cd)
                b_update = learning_rate * (h - h_cd)

                self.W += W_update
                self.b += b_update
                self.a += a_update
        # self.Z = self.compute_Z()

class GBRMB(BRBM):
    def __init__(self, n_visible, n_hidden):
        super().__init__(n_visible, n_hidden)
        self.mean = None
        self.std = None

    def v_probability(self, h, x):
        # probabilities vector for given x values
        N = 1 / np.sqrt(2 * np.pi)
        exp = np.exp(-0.5 * (x - self.b - self.W @ h) ** 2)
        return N * exp

    def draw_visible(self, h):
        mean = self.b - self.W @ h
        v = np.random.normal(loc=mean, scale=1, size=self.n_visible)
        return v

    def fit(self, V, iterations, learning_rate, cd_n=1):
        # input matrix needs to be normalized
        self.mean = np.mean(V, axis=0)
        V -= self.mean
        self.std = np.std(V, axis=0)
        V /= self.std
        super().fit(V, iterations, learning_rate, cd_n)
