import numpy as np


def hessian_fd(f, x, h=1, k=1):
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_ij = x.copy()
            x_ij[i] += h
            x_ij[j] += k
            x_ih_jk = x.copy()
            x_ih_jk[i] += h
            x_ih_jk[j] -= k
            x_imh_jk = x.copy()
            x_imh_jk[i] -= h
            x_imh_jk[j] += k
            x_imh_jm = x.copy()
            x_imh_jm[i] -= h
            x_imh_jm[j] -= k

            H[i, j] = (f(x_ij) - f(x_ih_jk) - f(x_imh_jk) + f(x_imh_jm)) / (4 * h * k)

    return H

