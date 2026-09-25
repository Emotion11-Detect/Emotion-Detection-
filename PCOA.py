import time
import numpy as np
from math import tan, pi, exp
from scipy.optimize import minimize


# Utility functions
def levy_flight(beta=1.5, dim=1):
    sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    return u / (np.abs(v) ** (1.0 / beta))


def ensure_bounds(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


# Pine Cone Optimization Algorithm (PCOA)
def PCOA(trees, func, lb, ub, FESmax):
    N_tree, dim = trees.shape[0], trees.shape[1]
    N_cone = 5
    PN = 20
    P1 = 0.05
    P2 = 0.8
    CX = np.zeros((N_tree, N_cone, dim))
    CF = np.full((N_tree, N_cone), np.inf)
    archive = []

    # Generate cones
    FES = 0
    for i in range(N_tree):
        LbS = lb + i * (ub - lb) / N_tree
        UbS = lb + (i + 1) * (ub - lb) / N_tree
        for j in range(N_cone):
            r1, r2 = np.random.rand(dim), np.random.rand()
            CX[i, j] = LbS[i, :] + r1 * (r2 * UbS[i, :] - r2 * LbS[i, :])
            CF[i, j] = func(CX[i, j])
            FES += 1
            archive.append((CX[i, j].copy(), CF[i, j]))
        trees[i] = CX[i, np.argmin(CF[i])]

    # Global best
    all_x = CX.reshape(-1, dim)
    all_f = CF.flatten()
    best_idx = np.argmin(all_f)
    Xbest, Fbest = all_x[best_idx].copy(), all_f[best_idx]
    Convergence_curve = np.zeros((FESmax, 1))

    t = 0
    ct = time.time()
    while FES < FESmax:
        # Gravity dispersal
        for i in range(N_tree):
            for j in range(N_cone):
                control = 0 if np.random.rand() < 0.25 else 1
                TXi, CXji = trees[i], CX[i, j]
                Lbi_s = lb + i * (ub - lb) / N_tree
                Ubi_s = lb + (i + 1) * (ub - lb) / N_tree
                R1, R2 = np.random.rand(dim), np.random.rand(dim)
                if control == 0:
                    new = TXi + 0.5 * R1 * (R2 * (Ubi_s - Lbi_s - TXi))
                else:
                    if len(archive) > 0:
                        Tpop = archive[np.random.randint(len(archive))][0]
                    else:
                        Tpop = CXji
                    new = CXji + 0.5 * R1 * (R2 * (Ubi_s - Lbi_s - Tpop) - Xbest)
                new = ensure_bounds(new, lb, ub)
                fnew = func(new)
                FES += 1
                if fnew < CF[i, j]:
                    CX[i, j], CF[i, j] = new, fnew
                    archive.append((new.copy(), fnew))

        # Animal dispersal
        for i in range(N_tree):
            for j in range(N_cone):
                CXij = CX[i, j].copy()
                if (FES > P2 * FESmax and np.random.rand() < 0.9) or \
                        (FES < P1 * FESmax and np.random.rand() < 0.9):
                    Xinit = Xbest + np.random.rand(dim) * (CXij - Xbest)
                    res = minimize(lambda x: func(x), Xinit,
                                   bounds=list(zip(lb, ub)), method='SLSQP',
                                   options={'maxiter': 100, 'ftol': 1e-6})
                    Xanimal = res.x if res.success else Xinit
                else:
                    wd = exp(-20 * FES / max(1, FESmax))
                    if np.random.rand() < 0.5:
                        TX = np.mean(trees, axis=0)
                        Treex = trees[np.random.randint(N_tree)]
                        step = levy_flight(dim=dim)
                        Xanimal = CXij + (1 - wd) * TX + wd * step * (step * (lb + ub - TX) - Treex)
                    else:
                        step = levy_flight(dim=dim)
                        Xanimal = CXij + wd * step * (step * (lb + ub - CXij) - CXij)
                Xanimal = ensure_bounds(Xanimal, lb, ub)
                f_anim = func(Xanimal)
                FES += 1
                if f_anim < CF[i, j]:
                    CX[i, j], CF[i, j] = Xanimal, f_anim
                    archive.append((Xanimal.copy(), f_anim))

        # Update trees
        for i in range(N_tree):
            trees[i] = CX[i, np.argmin(CF[i])]

        # Update global best
        all_x, all_f = CX.reshape(-1, dim), CF.flatten()
        best_idx = np.argmin(all_f)
        if all_f[best_idx] < Fbest:
            Fbest, Xbest = all_f[best_idx], all_x[best_idx].copy()
        Convergence_curve[t] = Fbest
        t = t + 1
    Fbest = Convergence_curve[FESmax - 1][0]
    ct = time.time() - ct
    return Fbest, Convergence_curve, Xbest, ct
