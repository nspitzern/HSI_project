import numpy as np


def lidstone(counts, eps=1e-4):
    d = counts.size
    N = np.sum(counts)

    return (counts + eps) / (N + eps * d)


if __name__ == '__main__':
    a = np.zeros(3)
    a[0] = 3
    a[1] = 3
    a[2] = 2

    res = [round(x, 8) for x in lidstone(a, eps=2)]
    print(res)

    assert res == [0.35714286, 0.35714286, 0.28571429]