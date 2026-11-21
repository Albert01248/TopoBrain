"""Fixed electrode-geometry controls; these graphs are not brain connectivity claims."""
from __future__ import annotations

import numpy as np


def _validate_graph(adjacency):
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError("adjacency must be nonempty square [C,C]")
    if not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
        raise ValueError("adjacency must be finite and binary")
    if not np.array_equal(a, a.T) or np.any(np.diag(a)):
        raise ValueError("adjacency must be symmetric with ZERO diagonal")
    return a.astype(np.float32, copy=True)


def _geometry(coords, k):
    channels = len(coords)
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    if channels < 2:
        return np.zeros((channels, channels), dtype=np.float32)
    # Stable sort fixes equal-distance ties using the input channel order.
    distance = np.sum((coords[:, None] - coords[None, :]) ** 2, axis=-1)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :min(int(k), channels - 1)]
    a = np.zeros((channels, channels), dtype=np.float32)
    a[np.arange(channels)[:, None], neighbors] = 1
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0)
    return a


def _degree_randomization(reference, rng):
    """Degree-preserving double-edge swaps, with explicit failure on rigid graphs."""
    a = reference.copy()
    edge_count = int(a.sum() // 2)
    if edge_count < 2:
        raise ValueError("random_degree cannot randomize this graph: fewer than two edges")
    accepted = 0
    target = max(10, edge_count * 10)
    last_changed = None
    for _ in range(max(1000, target * 100)):
        edges = np.argwhere(np.triu(a, 1))
        first, second = rng.choice(len(edges), size=2, replace=False)
        u, v = (int(x) for x in edges[first])
        x, y = (int(z) for z in edges[second])
        if len({u, v, x, y}) != 4:
            continue
        if rng.integers(2):
            x, y = y, x
        if a[u, y] or a[x, v]:
            continue
        a[u, v] = a[v, u] = a[x, y] = a[y, x] = 0
        a[u, y] = a[y, u] = a[x, v] = a[v, x] = 1
        accepted += 1
        if not np.array_equal(a, reference):
            last_changed = a.copy()
        if accepted >= target:
            break
    if last_changed is None:
        raise ValueError("random_degree could not produce a different degree-matched graph; topology may be rigid/dense. Choose a justified sparser graph or report this control as unavailable.")
    if not np.array_equal(last_changed.sum(axis=1), reference.sum(axis=1)):
        raise RuntimeError("internal error: edge swaps changed node degrees")
    return last_changed


def build_graph(coords, kind="geometry", k=4, seed=0):
    """Return binary symmetric ZERO-diagonal adjacency as numpy.float32.

    identity means no edges here: the model adds self loops for observed nodes.
    misaligned permutes A alone; input features must retain their original order.
    random_degree preserves each labeled node's unweighted degree, not geometry.
    A successful swap is not evidence of adequate random-graph mixing.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 1 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite nonempty [C,3]")
    channels = len(coords)
    if kind == "identity":
        return np.zeros((channels, channels), dtype=np.float32)
    if kind == "dense":
        return np.ones((channels, channels), dtype=np.float32) - np.eye(channels, dtype=np.float32)
    if kind not in {"geometry", "misaligned", "random_degree"}:
        raise ValueError("kind must be geometry, identity, misaligned, random_degree, or dense")
    reference = _geometry(coords, k)
    if kind == "geometry":
        return reference
    rng = np.random.default_rng(seed)
    if kind == "random_degree":
        return _degree_randomization(reference, rng)
    for _ in range(256):
        permutation = rng.permutation(channels)
        changed = reference[np.ix_(permutation, permutation)]
        if not np.array_equal(changed, reference):
            return changed
    raise ValueError("misaligned could not change this graph; all attempted permutations preserved its topology (for example a complete/empty graph)")


def graph_diagnostics(adjacency, reference=None):
    a = _validate_graph(adjacency)
    channels = len(a)
    remaining = set(range(channels))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = [start]
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(a[node]):
                neighbor = int(neighbor)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    result = {"channels": channels, "edge_count": int(a.sum() // 2),
              "degrees": a.sum(axis=1).astype(int).tolist(), "components": components,
              "component_count": len(components), "connected": len(components) == 1,
              "checks": {"binary": True, "symmetric": True, "zero_diagonal": True},
              "changed_edge_fraction": None}
    if reference is not None:
        ref = _validate_graph(reference)
        if ref.shape != a.shape:
            raise ValueError("reference shape differs from adjacency")
        changed = int(np.count_nonzero(np.triu(a != ref, 1)))
        total_edges = int((a.sum() + ref.sum()) // 2)
        result.update({"changed_edge_fraction": changed / max(total_edges, 1),
                       "changed_undirected_pairs": changed,
                       "changed_edge_fraction_definition": "symmetric-difference edge count / sum of edge counts; equal-size graphs give the fraction of replaced edges",
                       "same_labeled_node_degrees": bool(np.array_equal(a.sum(axis=1), ref.sum(axis=1))),
                       "same_degree_sequence": bool(np.array_equal(np.sort(a.sum(axis=1)), np.sort(ref.sum(axis=1)))),
                       "same_edge_count": bool(a.sum() == ref.sum()),
                       "topology_changed": not np.array_equal(a, ref)})
    return result
