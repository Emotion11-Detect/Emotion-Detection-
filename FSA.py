import time

import numpy as np


def levy_flight(beta=1.5, dim=1):
    """
    Levy flight using Mantegna's algorithm.
    Implements the heavy-tailed step distribution (used in Eq.3, Eq.5).
    """
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, size=dim)
    v = np.random.normal(0, 1, size=dim)
    return u / (np.abs(v) ** (1.0 / beta))


def ensure_bounds(vec, lb, ub):
    """Eq. (1) bounds: ensure candidate stays inside search space"""
    return np.clip(vec, lb, ub)


# Flamingo Search Algorithm (FSA)
def FSA(X, func, lb, ub, max_iter):
    n_pop, dim = X.shape[0], X.shape[1]
    fitness = np.apply_along_axis(func, 1, X)  # Eq. (2)

    # Global best (Eq.8)
    best_idx = np.argmin(fitness)
    gbest = X[best_idx].copy()
    gbest_score = fitness[best_idx]
    convergence = []
    ct = time.time()

    # Main Loop
    for t in range(max_iter):
        a = 2 * (1 - t / max_iter)  # decreasing factor for Eq. (7)

        for i in range(n_pop):
            xi = X[i].copy()
            r1, r2 = np.random.rand(), np.random.rand()
            # ---------------- FSA Phases ----------------
            if r1 < 0.25:
                # Spiral exploration (Eq.3)
                step = levy_flight(dim=dim)
                new = xi + r2 * step * (gbest - xi)
            elif r1 < 0.5:
                # Group following (Eq.4)
                peer = X[np.random.randint(n_pop)]
                new = xi + r2 * (gbest - peer)
            elif r1 < 0.75:
                # Searching for food (Eq.5)
                step = levy_flight(dim=dim)
                new = xi + step * (xi - gbest)
            else:
                # Attacking prey (Eq.6, Eq.7)
                C = 2 * r2
                A = 2 * a * r2 - a
                D = abs(C * gbest - xi)
                new = gbest - A * D
            # ---------------- Bounds & Evaluation ----------------
            new = ensure_bounds(new, lb, ub)
            fnew = func(new)
            # Greedy selection
            if fnew[i] < fitness[i]:
                X[i], fitness[i] = new[i], fnew[i]
        # ---------------- Global Best Update (Eq.8) ----------------
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < gbest_score:
            gbest, gbest_score = X[best_idx].copy(), fitness[best_idx]

        convergence.append(gbest_score)
    ct = time.time() - ct
    return gbest, convergence, gbest_score, ct
