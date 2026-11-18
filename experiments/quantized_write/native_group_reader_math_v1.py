"""Shared-query write control through headwise normalization and output metric."""
import numpy as np
from native_normalized_reader_math_v1 import unit, oriented, normalized, project, objective_gradient


def group_objective(u, states, deltas, qnorm, target):
    value = 0.
    gradient = np.zeros_like(u)
    for s, d, t in zip(states, deltas, target):
        f, g = objective_gradient(u, s, d, qnorm, t)
        value += f
        gradient += g
    return value, gradient


def optimize(states, deltas, native, target, starts):
    qnorm = np.linalg.norm(native)
    n = unit(native)
    solutions = []
    for initial in starts:
        u = project(initial, n)
        for _ in range(80):
            f, g = group_objective(u, states, deltas, qnorm, target)
            g = unit(g-u*(u@g))
            options = [(f, u)]
            for step in (.2, .1, .05, .025, .0125, .00625, .003125):
                v = project(u+step*g, n)
                fv, _ = group_objective(v, states, deltas, qnorm, target)
                options.append((fv, v))
            u = max(options, key=lambda x: x[0])[1]
        solutions.append((group_objective(u, states, deltas, qnorm, target)[0], u))
    return max(solutions, key=lambda x: x[0])[1]*qnorm


def candidates(states, deltas, native, key, output_gram, gated_weights):
    states, deltas = np.asarray(states), np.asarray(deltas)
    heads = len(states)
    assert states.shape == deltas.shape == (heads, 128, 128)
    assert gated_weights.shape == (heads, 128) and output_gram.shape == (heads*128, heads*128)
    qnorm = np.linalg.norm(native)
    n = unit(native)
    combined = np.concatenate(deltas, axis=1)
    left, singular, _ = np.linalg.svd(combined, full_matrices=False)
    rank = int(np.sum(singular > max(combined.shape)*np.finfo(float).eps*singular[0])) if singular[0] else 0
    svd = oriented(left[:, 0], n) if rank else n.copy()
    covariance = sum(s@s.T for s in states)
    ridge = max(1e-3*np.trace(covariance)/128, 1e-12)
    regular = covariance + ridge*np.eye(128)
    white_key = oriented(np.linalg.solve(regular, key), n)
    eigenvalues, eigenvectors = np.linalg.eigh(regular)
    inverse_root = (eigenvectors/np.sqrt(eigenvalues))@eigenvectors.T
    wh_left = np.linalg.svd(inverse_root@combined, full_matrices=False)[0][:, 0]
    white_svd = oriented(inverse_root@wh_left, n)
    null = n-left[:, :rank]@(left[:, :rank].T@n)
    if np.linalg.norm(null) < 1e-12:
        null = np.eye(128)[np.argmin(np.sum(left[:, :rank]**2, axis=1))]
        null -= left[:, :rank]@(left[:, :rank].T@null)
    null = oriented(null, n)
    delta_norm = np.stack([normalized(native@(s+d))-normalized(native@s) for s, d in zip(states, deltas)])
    normalized_target = unit(delta_norm.reshape(-1)).reshape(heads, 128)
    gate = gated_weights.reshape(-1)
    gated_delta = gate*delta_norm.reshape(-1)
    residual_norm = np.sqrt(max(float(gated_delta@output_gram@gated_delta), 0.))
    output_target = (gate*(output_gram@gated_delta)/max(residual_norm, 1e-30)).reshape(heads, 128)
    starts = [n, svd, white_key, white_svd]
    queries = {'raw_svd': svd*qnorm, 'transported_key': oriented(key, n)*qnorm,
               'whitened_key': white_key*qnorm, 'whitened_svd': white_svd*qnorm,
               'complete_null': null*qnorm,
               'group_normalized': optimize(states, deltas, native, normalized_target, starts),
               'group_output': optimize(states, deltas, native, output_target, starts)}
    settings = {'ridge': ridge, 'minimum_cosine': .5, 'iterations': 80, 'starts': 4,
                'rank': rank, 'native_write_residual_norm': residual_norm}
    return queries, {'normalized': normalized_target, 'output': output_target}, settings
