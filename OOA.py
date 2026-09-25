import time

import numpy as np


def ensure_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Clamp vector x to box bounds [lb, ub]."""
    return np.minimum(np.maximum(x, lb), ub)


def safe_rand_normal_in_01(mu=0.5, sigma=0.1666) -> float:
    """
    Random number r ~ Normal(0,1) but restricted in [0,1] as per paper.
    Eq. (5) mentions r ∈ [0,1] with normal distribution.
    """
    v = np.random.normal(loc=mu, scale=sigma)
    return float(np.clip(v, 0.0, 1.0))


# Orangutan Optimization Algorithm (OOA)

def OOA(X, func, lb, ub, max_iter):
    n_pop, dim = X.shape[0], X.shape[1]
    maximize = False
    # For minimization (default), smaller is better.
    sign = -1.0 if maximize else 1.0

    # ----- Eq.(3): evaluate fitness -----
    F = np.apply_along_axis(func, 1, X)
    if maximize:
        F = -F  # flip sign if maximizing

    # Best solution so far
    best_idx = int(np.argmin(F))
    best_x = X[best_idx].copy()
    best_f = float('inf')
    Convergence_curve = np.zeros((max_iter, 1))
    ct = time.time()
    for t in range(max_iter):

        # ==================================================
        # Phase 1: Foraging skill (Exploration)
        # Eq.(4): build SFS_i = {solutions with better fitness}
        # Eq.(5): x_i,d^P1 = x_i,d + r * (SFS_d - I * x_i,d)
        # Eq.(6): accept if f(x^P1) < f(x_i)
        # ==================================================
        for i in range(n_pop):
            xi = X[i].copy()
            fi = F[i]

            # Select better food sources (Eq.4)
            better_idx = np.where(F < fi)[0]
            if better_idx.size == 0:
                continue  # no better source

            # Pick one SFS randomly
            k = np.random.choice(better_idx)
            sfs = X[k]

            r = safe_rand_normal_in_01()  # r ∈ [0,1], normal (Eq.5)
            I_val = np.random.choice([1, 2])  # I ∈ {1,2} (Eq.5)

            # Eq.(5)
            x_p1 = xi + r * (sfs - I_val * xi)
            x_p1 = ensure_bounds(x_p1, lb, ub)

            f_p1 = func(x_p1)
            f_p1_signed = -f_p1 if maximize else f_p1

            # Eq.(6) acceptance
            if f_p1_signed[i] <= fi:
                X[i] = x_p1[i]
                F[i] = f_p1_signed[i]
                if f_p1_signed[i] < best_f:
                    best_f = f_p1_signed[i]
                    best_x = x_p1.copy()

        # ==================================================
        # Phase 2: Nesting skill (Exploitation)
        # Eq.(7): x_i,j^P2 = x_i,j + (1 - 2*r_ij) * ((ub_j - lb_j)/t)
        # Eq.(8): accept if f(x^P2) < f(x_i)
        # ==================================================
        for i in range(n_pop):
            xi = X[i].copy()
            fi = F[i]

            r_vec = np.random.rand(dim)

            # Eq.(7)
            delta = (1.0 - 2.0 * r_vec) * ((ub - lb) / float(max(1, t)))
            x_p2 = xi + delta
            x_p2 = ensure_bounds(x_p2, lb, ub)

            f_p2 = func(x_p2)
            f_p2_signed = -f_p2 if maximize else f_p2

            # Eq.(8) acceptance
            if f_p2_signed[i] <= fi:
                X[i] = x_p2[i]
                F[i] = f_p2_signed[i]
                if f_p2_signed[i] < best_f:
                    best_f = f_p2_signed[i]
                    best_x = x_p2.copy()
        Convergence_curve[t] = best_f
    best_f = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct
    return best_f, Convergence_curve, best_x, ct
