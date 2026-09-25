import time

import numpy as np


#  Levy flight (eq. 9-10)
def levy_flight(beta, dim):
    scaling = np.random.normal(0, 1, dim)
    r1 = np.random.normal(0, 1, dim)
    r2 = np.random.normal(0, 1, dim)
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    step = scaling * r1 * sigma / (np.abs(r2) ** (1 / beta))
    return step


# Lotus Effect Optimization Algorithm (LEOA)
def LEA(X, objective, lb, ub, iterations):
    pop_size, dim = X.shape[0], X.shape[1]
    s = 1.0
    a = 1.0
    c = 1.0
    f = 1.0
    e = 1.0
    w = 0.9,
    beta = 1.5
    drop_beta_q = 0.9
    DeltaX = np.zeros_like(X)  # eq. (6) ΔX
    V = np.zeros_like(X)  # for drop moves (eq. 16)
    fitness = np.array([objective(x) for x in X])
    g_best = X[np.argmin(fitness)].copy()
    g_worst = X[np.argmax(fitness)].copy()
    bestsol = np.zeros((dim, 1))
    L = iterations  # for eq. (13)
    Convergence_curve = np.zeros((iterations, 1))
    ct = time.time()
    # --- Main loop ---
    for t in range(iterations):
        for i in range(pop_size):
            # neighbors: all except self (simplified full neighborhood)
            neigh_idx = [j for j in range(pop_size) if j != i]
            neigh = X[neigh_idx]
            vel_neigh = DeltaX[neigh_idx]

            # eq. (1) Separation
            St = s * (-np.sum(X[i] - neigh, axis=0))
            # eq. (2) Alignment
            At = a * np.mean(vel_neigh, axis=0)
            # eq. (3) Cohesion
            Ct = c * (np.mean(neigh, axis=0) - X[i])
            # eq. (4) Food attraction
            Ft = f * (g_best - X[i])
            # eq. (5) Enemy repulsion
            Et = e * (g_worst + X[i])

            # eq. (6) Step update
            DeltaX[i] = (St + At + Ct + Ft + Et) + w * DeltaX[i]

            # eq. (7-8) Position update / Levy flight if isolated
            if len(neigh) > 0:
                X[i] = X[i] + w * DeltaX[i]
            else:
                X[i] = X[i] + levy_flight(beta, dim) * X[i]

        # eq. (12-13) Exploitation move (local pollination)
        R = 2.0 * np.exp(- (4.0 * t / float(L)) ** 2)
        for i in range(pop_size):
            if np.random.rand() < 0.5:
                X[i] = X[i] + R * (X[i] - g_best)

        # eq. (14) Pit capacities
        fitness = np.array([objective(x) for x in X])
        fmax, fmin = np.max(fitness), np.min(fitness)
        denom = np.abs(fmin - fmax) if np.abs(fmin - fmax) > 1e-12 else 1e-12
        capacities = np.abs(fitness - fmax) / denom

        # eq. (15-17) Drop overflow / normal drop
        for i in range(pop_size):
            if np.random.rand() < 0.1:  # overflow chance
                higher = np.where(capacities > capacities[i])[0]
                if len(higher) > 0:
                    probs = capacities[higher] / np.sum(capacities[higher])
                    dest = np.random.choice(higher, p=probs)
                    V[i] = V[i] + np.random.rand(*X[i].shape) * (X[dest] - X[i])  # eq. (17)
                else:
                    V[i] = drop_beta_q * V[i]  # eq. (16)
            else:
                V[i] = drop_beta_q * V[i]
            X[i] = X[i] + V[i]

        # Boundary handling
        X = np.clip(X, -1.0, 1.0)

        # Update global best/worst
        fitness = np.array([objective(x) for x in X])
        best_idx, worst_idx = np.argmin(fitness), np.argmax(fitness)
        if fitness[best_idx] < objective(g_best):
            g_best = X[best_idx].copy()
        g_worst = X[worst_idx].copy()
        Convergence_curve[t] = g_best[0]
    g_best = Convergence_curve[iterations - 1][0]
    ct = time.time() - ct
    return g_best, Convergence_curve, bestsol, ct
