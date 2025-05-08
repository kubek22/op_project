import torch

class RBM:
    def __init__(self, n_visible, n_hidden, device='cpu'):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = torch.device(device)

        self.a = torch.zeros(self.n_visible).to(self.device)
        self.b = torch.zeros(self.n_hidden).to(self.device)
        self.W = torch.zeros((self.n_visible, self.n_hidden)).to(self.device)

    def to(self, device):
        self.device = device
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.W = self.W.to(device)
        return self

    def h_probability(self, v):
        v = v.to(self.device)
        return torch.sigmoid(self.b + v @ self.W)

    def v_probability(self, h):
        h = h.to(self.device)
        return torch.sigmoid(self.a + (self.W @ h.T).T)

    def draw_hidden(self, v):
        v = v.to(self.device)
        p = self.h_probability(v)
        h = torch.bernoulli(p)
        return h

    def draw_visible(self, h):
        h = h.to(self.device)
        # v = torch.zeros(self.n_visible).to(self.device)
        p = self.v_probability(h) # keep tensor shape
        v = torch.bernoulli(p)
        return v

    def gibbs_sampling(self, n, h):
        h = h.to(self.device)
        for i in range(n):
            v = self.draw_visible(h)
            h = self.draw_hidden(v)
        return v, h

    # TODO exchange fit method
    def fit(self, V, iterations, learning_rate, cd_n=1, verbose=False):
        for i in range(iterations):
            if verbose:
                print(f"Iteration: {i+1} of {iterations}")
            # gradient descent
            for v in V:
                v = v.to(self.device)
                h = self.draw_hidden(v)
                v_cd, h_cd = self.gibbs_sampling(cd_n, h)

                W_update = learning_rate * torch.outer(v, h) - torch.outer(v_cd, h_cd)
                a_update = learning_rate * (v - v_cd)
                b_update = learning_rate * (h - h_cd)

                self.W += W_update
                self.b += b_update
                self.a += a_update

    def fit_batch(self, V, iterations, learning_rate, cd_n=1, batch_size=64, verbose=False):
        dataset = torch.utils.data.TensorDataset(V)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i in range(iterations):
            if verbose:
                print(f"Iteration: {i + 1} of {iterations}")
            for batch in dataloader:
                v = batch[0].to(self.device)
                h = self.draw_hidden(v)
                v_cd, h_cd = self.gibbs_sampling(cd_n, h)

                self.W += learning_rate * ((v.T @ h - v_cd.T @ h_cd) / v.size(0))
                self.a += learning_rate * torch.mean(v - v_cd, dim=0)
                self.b += learning_rate * torch.mean(h - h_cd, dim=0)
