import torch

class RBM:
    def __init__(self, n_visible, n_hidden, device='cpu'):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = torch.device(device)

        self.a = torch.zeros(self.n_visible).to(self.device).double()
        self.b = torch.zeros(self.n_hidden).to(self.device).double()
        self.W = torch.zeros((self.n_visible, self.n_hidden)).to(self.device).double()

    def state_dict(self):
        state_dict = {
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "device": self.device,
            "a": self.a,
            "b": self.b,
            "W": self.W
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.n_visible = state_dict["n_visible"]
        self.n_hidden = state_dict["n_hidden"]
        self.device = state_dict["device"]
        self.a = state_dict["a"]
        self.b = state_dict["b"]
        self.W = state_dict["W"]

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

    def fit(self, V, iterations, learning_rate, cd_n=1, batch_size=64, verbose=False):
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
