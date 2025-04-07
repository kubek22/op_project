import numpy as np
from rbm import RBM

def generate_bars_stripes(n=4):
    patterns = []
    for i in range(2 ** n):
        row_pattern = [(i >> j) & 1 for j in range(n)]
        horiz = np.tile(row_pattern, (n, 1))
        vert = np.tile(np.array(row_pattern).reshape(-1, 1), (1, n))
        patterns.append(horiz.flatten())
        patterns.append(vert.flatten())
    return np.unique(np.array(patterns), axis=0)

data = generate_bars_stripes(n=4)  # shape (32, 16)
print(data)

rbm = RBM(n_visible=16, n_hidden=8)
rbm.fit(data, 100, 0.001, 1)

reconstruction = [rbm.draw_visible(rbm.draw_hidden(v)) for v in data]
reconstruction = np.array(reconstruction)
print(reconstruction)

print(f"error: {np.linalg.norm(reconstruction - data)}")
print(rbm.W)
print(rbm.a)
print(rbm.b)
