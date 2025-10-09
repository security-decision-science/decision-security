def parents(edges, node):
    """Return set of parents of `node` given directed edges (u, v)."""
    return {u for (u, v) in edges if v == node}

def children(edges, node):
    """Return set of children of `node` given directed edges (u, v)."""
    return {v for (u, v) in edges if u == node}

def descendants(edges, node):
    """Return set of descendants of `node` in a directed acyclic graph."""
    g = {}
    for u, v in edges:
        g.setdefault(u, set()).add(v)
    out, stack = set(), [node]
    while stack:
        x = stack.pop()
        for ch in g.get(x, ()):
            if ch not in out:
                out.add(ch)
                stack.append(ch)
    out.discard(node)
    return out

def topological_sort(nodes, edges):
    """Topological ordering of DAG nodes."""
    from collections import deque, defaultdict
    indeg = defaultdict(int)
    G = defaultdict(set)
    for u, v in edges:
        if v not in G[u]:
            G[u].add(v)
            indeg[v] += 1
            indeg.setdefault(u, 0)
    q = deque([n for n in nodes if indeg.get(n, 0) == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in G.get(u, ()):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(set(nodes)):
        raise ValueError("Graph has a cycle; not a DAG.")
    return order

def backdoor_adjustment_set(edges, treatment, outcome, candidates):
    """
    Small heuristic: adjust for treatment's parents, excluding treatment's descendants.
    (Teaching/demo only; not a full d-separation solver.)
    """
    cand = set(candidates)
    adj = parents(edges, treatment)
    adj -= descendants(edges, treatment)
    if outcome in adj:
        adj.remove(outcome)
    return adj & cand