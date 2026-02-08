"""
Reference implementation for:

    "Constructor-Modulated Causal Substrates and Life-Like Motifs
     in Discrete Hypergraph Rewrite Systems"

This script implements the model described in the manuscript:
- simple undirected graphs as 2-uniform hypergraph substrates,
- motifs M1 (open triples), M2 (triangles), M3 (double triangles),
- local rewrite rules R1 (closure), R2 (duplication), R3 (decay),
- constructor-mediated bias, and the free-energy-like functional F = E - T_eff * S.

The code is provided as a reproducible reference for generating
simulation data and figures in the paper (constructor-biased vs unbiased runs).
"""

import random
from collections import defaultdict, deque

# ============================================================
# Graph substrate (simple undirected graph as 2-uniform hypergraph)
# ============================================================

class Graph:
    """
    Counterfactual causal substrate: simple undirected graph.

    Nodes: arbitrary hashable labels.
    Edges: unordered pairs {u,v} with u != v.
    """
    def __init__(self, nodes=None, edges=None):
        self.nodes = set(nodes) if nodes else set()
        # store edges as frozenset({u,v}) with u!=v
        self.edges = {frozenset(e) for e in (edges or []) if len(set(e)) == 2}

    def add_edge(self, u, v):
        if u != v:
            self.nodes.update([u, v])
            self.edges.add(frozenset((u, v)))

    def remove_edge(self, u, v):
        self.edges.discard(frozenset((u, v)))

    def has_edge(self, u, v):
        return frozenset((u, v)) in self.edges

    def neighbors(self, u):
        return {v for e in self.edges for v in e if u in e and v != u}

    def copy(self):
        return Graph(self.nodes.copy(), [tuple(e) for e in self.edges])


# ============================================================
# Motif detection: M1 (open triple), M2 (triangle), M3 (double triangle)
# ============================================================

def find_triangles(G):
    """
    M2: all triangles (3-cycles) in G, returned as sorted triples (a,b,c).
    """
    triangles = []
    nodes = sorted(G.nodes)
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = nodes[i], nodes[j], nodes[k]
                if G.has_edge(a,b) and G.has_edge(b,c) and G.has_edge(a,c):
                    triangles.append((a, b, c))
    return triangles


def find_open_triples(G):
    """
    M1: open triples (paths of length 2) without closing edge.

    Returns triples (a, b, c) where edges (a,b) and (b,c) exist but (a,c) does not.
    """
    open_triples = []
    nodes = list(G.nodes)
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if len({nodes[i], nodes[j], nodes[k]}) != 3:
                    continue
                a, b, c = nodes[i], nodes[j], nodes[k]
                if G.has_edge(a,b) and G.has_edge(b,c) and not G.has_edge(a,c):
                    open_triples.append((a, b, c))
    # deduplicate by node set
    seen = set()
    uniq = []
    for a, b, c in open_triples:
        key = tuple(sorted((a, b, c)))
        if key not in seen:
            seen.add(key)
            uniq.append((a, b, c))
    return uniq


def find_double_triangles(G):
    """
    M3: double triangles (two triangles sharing exactly one edge, 4 distinct nodes).

    Returns sorted 4-tuples of node labels.
    """
    triangles = find_triangles(G)
    edge_to_triangles = defaultdict(list)
    for t in triangles:
        a, b, c = t
        edges = [tuple(sorted((a,b))), tuple(sorted((b,c))), tuple(sorted((a,c)))]
        for e in edges:
            edge_to_triangles[e].append(t)

    double_tris = set()
    for edge, tris in edge_to_triangles.items():
        if len(tris) < 2:
            continue
        for i in range(len(tris)):
            for j in range(i+1, len(tris)):
                t1 = tris[i]
                t2 = tris[j]
                if len(set(t1 + t2)) == 4:
                    nodes = tuple(sorted(set(t1 + t2)))
                    double_tris.add(nodes)
    return list(double_tris)


# ============================================================
# Rules R1, R2, R3
# ============================================================

class RuleApplication:
    """
    Represents a single candidate rule application.
    """
    def __init__(self, rule_name, data):
        self.rule_name = rule_name  # 'R1', 'R2', 'R3'
        self.data = data            # pattern-specific info


def enumerate_R1(G, open_triples):
    """
    R1: triangle closure.

    For each open triple a-b-c, closing edge is (a,c).
    """
    apps = []
    for a, b, c in open_triples:
        apps.append(RuleApplication('R1', (a, c)))
    return apps


def enumerate_R2(G, triangles):
    """
    R2: triangle duplication into double triangle.

    For each triangle (i,j,k), pick neighbours of j or k as candidate l.
    Application will add edges to form triangle (j,k,l).
    """
    apps = []
    for (i, j, k) in triangles:
        tri_nodes = {i, j, k}
        for l in G.neighbors(j) | G.neighbors(k):
            if l in tri_nodes:
                continue
            apps.append(RuleApplication('R2', (j, k, l)))
    return apps


def enumerate_R3(G, triangles):
    """
    R3: triangle decay.

    For each triangle, define a candidate that will later remove one of its edges.
    """
    apps = []
    for (i, j, k) in triangles:
        apps.append(RuleApplication('R3', (i, j, k)))
    return apps


# ============================================================
# Constructors and bias (M3 motifs as constructors)
# ============================================================

class ConstructorMotif:
    """
    Model-level constructor: here, any double triangle (M3).
    """
    def __init__(self, nodes):
        self.nodes = set(nodes)


def detect_constructors(G):
    """
    Return list of ConstructorMotif objects corresponding to all M3s.
    """
    return [ConstructorMotif(nodes) for nodes in find_double_triangles(G)]


def constructor_bias(app, constructors, R=1, alpha=0.5, gamma=0.5):
    """
    Aggregate top-down bias from constructors for a given rule application.

    Neighbourhood: nodes within graph distance <= R from any constructor node.
    Growth rules (R1,R2) get positive bias; decay rule (R3) gets negative bias.
    """
    # For now, approximate neighbourhood by direct overlap with constructor nodes
    # (R=1); you can later generalise to true distance if desired.
    if app.rule_name == 'R1':
        touched = set(app.data)              # (a,c)
    elif app.rule_name == 'R2':
        touched = set(app.data)              # (j,k,l)
    elif app.rule_name == 'R3':
        touched = set(app.data)              # (i,j,k)
    else:
        touched = set()

    n = 0
    for ctor in constructors:
        if touched & ctor.nodes:
            n += 1

    if n == 0:
        return 0.0

    if app.rule_name in ('R1', 'R2'):
        return alpha * n
    elif app.rule_name == 'R3':
        return -gamma * n
    return 0.0


# ============================================================
# Free-energy-like functional F = E - T_eff * S
# ============================================================

def graph_strain(G, open_triples):
    """
    E(G): graph-strain term = number of open triples (M1) not in any triangle.
    """
    return len(open_triples)


def branching_measure_window(rule_history, window=20):
    """
    S(G): windowed branching measure over the last `window` steps.

    Here we use the number of distinct rule types applied in the last
    W steps as a bounded proxy for local branching / dynamical richness.
    """
    if not rule_history:
        return 0
    recent = rule_history[-window:]
    return len(set(recent))


def effective_temperature(num_ctors, base_T=1.0, beta=0.1):
    """
    T_eff(G): temperature-like parameter, boosted by constructor count.

    T_eff = base_T * (1 + beta * #constructors).
    """
    return base_T * (1.0 + beta * num_ctors)


def free_energy_like(G, open_triples, constructors, rule_history,
                     window=20, base_T=1.0, beta=0.1):
    """
    Free-energy-like functional on graphs:

    F(G) = E(G) - T_eff(G) * S(G),
    where all quantities are model-level analogues.
    """
    E = graph_strain(G, open_triples)
    S = branching_measure_window(rule_history, window=window)
    T_eff = effective_temperature(len(constructors), base_T=base_T, beta=beta)
    return E - T_eff * S


# ============================================================
# Simulation core (with persistence + repair metrics)
# ============================================================

REPAIR_WINDOW = 10  # steps ahead to look for repair after a decay

def find_motifs(G):
    """
    Detect motifs M1, M2, M3.
    """
    open_triples = find_open_triples(G)
    triangles = find_triangles(G)
    double_tris = find_double_triangles(G)
    return open_triples, triangles, double_tris


def enumerate_rule_applications(G, open_triples, triangles, constructors,
                                use_bias=True):
    """
    Enumerate candidates for R1, R2, R3 and compute weights with or without constructor bias.
    """
    apps = []
    apps += enumerate_R1(G, open_triples)
    apps += enumerate_R2(G, triangles)
    apps += enumerate_R3(G, triangles)

    if not apps:
        return [], []

    weights = []
    for app in apps:
        w = 1.0
        if use_bias:
            b = constructor_bias(app, constructors)
            w *= max(0.0, 1.0 + b)
        weights.append(w)

    return apps, weights


def apply_rule(G, app):
    """
    Apply one rule application to the graph.
    """
    if app.rule_name == 'R1':
        a, c = app.data
        G.add_edge(a, c)

    elif app.rule_name == 'R2':
        j, k, l = app.data
        G.add_edge(j, k)
        G.add_edge(j, l)
        G.add_edge(k, l)

    elif app.rule_name == 'R3':
        i, j, k = app.data
        edges = [(i, j), (j, k), (i, k)]
        u, v = random.choice(edges)
        G.remove_edge(u, v)


def run_simulation(G0, T_steps=200, use_bias=True, seed=0,
                   window_S=20, base_T=1.0, beta=0.1):
    """
    Run one simulation (one model universe) for T_steps.

    Records time series of:
    - F(G_tau)
    - counts of M1, M2, M3
    - simple persistence proxies
    - cumulative repair events
    - rule names applied
    """
    random.seed(seed)
    G = G0.copy()
    rule_history = []
    records = []

    # persistence counters: consecutive steps with CN > 0
    persistence = {'M1': 0, 'M2': 0, 'M3': 0}
    # track recent rules for other uses (if needed)
    recent_rules = deque(maxlen=REPAIR_WINDOW)
    # NEW: track recent decays for forward-looking repair proxy
    recent_decays = deque(maxlen=10)  # times when R3 was applied

    repair_events = 0

    for t in range(T_steps):
        open_triples, triangles, double_tris_nodes = find_motifs(G)
        constructors = [ConstructorMotif(nodes) for nodes in double_tris_nodes]

        apps, weights = enumerate_rule_applications(
            G, open_triples, triangles, constructors, use_bias=use_bias
        )
        if not apps:
            F = free_energy_like(G, open_triples, constructors, rule_history,
                                 window=window_S, base_T=base_T, beta=beta)
            counts = {
                'M1': len(open_triples),
                'M2': len(triangles),
                'M3': len(double_tris_nodes),
            }
            records.append({
                'time': t,
                'F': F,
                'num_M1': counts['M1'],
                'num_M2': counts['M2'],
                'num_M3': counts['M3'],
                'pers_M1': persistence['M1'],
                'pers_M2': persistence['M2'],
                'pers_M3': persistence['M3'],
                'repair_events': repair_events,
                'rule': None,
            })
            break

        total_w = sum(weights)
        r = random.random() * total_w
        acc = 0.0
        chosen = None
        for app, w in zip(apps, weights):
            acc += w
            if r <= acc:
                chosen = app
                break

        # record chosen rule name for legacy / other uses
        recent_rules.append(chosen.rule_name)

        apply_rule(G, chosen)
        rule_history.append(chosen.rule_name)

        # update motifs and F after move
        open_triples2, triangles2, double_tris_nodes2 = find_motifs(G)
        constructors2 = [ConstructorMotif(nodes) for nodes in double_tris_nodes2]
        F = free_energy_like(G, open_triples2, constructors2, rule_history,
                             window=window_S, base_T=base_T, beta=beta)

        counts = {
            'M1': len(open_triples2),
            'M2': len(triangles2),
            'M3': len(double_tris_nodes2),
        }

        # update persistence counters (simple consecutive presence)
        for m in ['M1', 'M2', 'M3']:
            if counts[m] > 0:
                persistence[m] += 1
            else:
                persistence[m] = 0

        # Forward-looking repair proxy — matches Methods.md:
        # decay at t → later (1–10 steps) R1/R2 with constructor present at that later time
        if chosen.rule_name == 'R3':
            recent_decays.append(t)

        repair_this_step = False
        if chosen.rule_name in ('R1', 'R2') and constructors2:
            for decay_time in recent_decays:
                steps_ago = t - decay_time
                if 1 <= steps_ago <= 10:
                    repair_this_step = True
                    break  # one is enough

        if repair_this_step:
            repair_events += 1

        records.append({
            'time': t,
            'F': F,
            'num_M1': counts['M1'],
            'num_M2': counts['M2'],
            'num_M3': counts['M3'],
            'pers_M1': persistence['M1'],
            'pers_M2': persistence['M2'],
            'pers_M3': persistence['M3'],
            'repair_events': repair_events,
            'rule': chosen.rule_name,
        })

    return records

# ============================================================
# Sanity checks (optional, lightweight)
# ============================================================
def _sanity_check_primitives():
    """
    Minimal smoke tests for core primitives.
    """
    # 1) Simple graph with exactly one triangle and no double triangles.
    G = Graph(nodes=[0, 1, 2], edges=[(0, 1), (1, 2), (0, 2)])
    M1, M2, M3 = find_motifs(G)
    assert len(M2) == 1          # exactly one M2
    assert len(M3) == 0          # no M3 should be detected here

    # 2) F(G) is defined and numeric on a trivial graph.
    rule_history = []
    F = free_energy_like(G, M1, [ConstructorMotif(nodes) for nodes in M3], rule_history)
    assert isinstance(F, (int, float))  # F should be a finite scalar


# ============================================================
# Main: run simulations, save CSVs, and plot
# ============================================================

if __name__ == "__main__":
    import csv
    import pandas as pd
    import matplotlib.pyplot as plt

    _sanity_check_primitives()  # fail fast if primitives are broken

    # Build initial sparse graph (random tree on 20 nodes)
    nodes = list(range(20))
    G0 = Graph(nodes, [])
    for i in range(1, len(nodes)):
        j = random.randrange(0, i)
        G0.add_edge(nodes[i], nodes[j])

    # Run two conditions: no constructor bias vs bias on
    records_no_bias = run_simulation(G0, T_steps=300, use_bias=False, seed=1)
    records_bias    = run_simulation(G0, T_steps=300, use_bias=True,  seed=1)

    # Write CSVs
    with open('simulation_no_bias.csv', 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'time','F','num_M1','num_M2','num_M3',
                'pers_M1','pers_M2','pers_M3',
                'repair_events','rule'
            ]
        )
        writer.writeheader()
        for r in records_no_bias:
            writer.writerow(r)

    with open('simulation_bias.csv', 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'time','F','num_M1','num_M2','num_M3',
                'pers_M1','pers_M2','pers_M3',
                'repair_events','rule'
            ]
        )
        writer.writeheader()
        for r in records_bias:
            writer.writerow(r)

    # Convert to DataFrames directly
    no_bias = pd.DataFrame(records_no_bias)
    bias    = pd.DataFrame(records_bias)

    # Plot F vs time (windowed S)
    plt.figure(figsize=(8,4))
    plt.plot(no_bias["time"], no_bias["F"], label="no constructors")
    plt.plot(bias["time"],    bias["F"],    label="constructors active")
    plt.xlabel("model time τ")
    plt.ylabel("F(G_τ) (free-energy-like, windowed S)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("F_time_windowed.png", dpi=150)
    plt.close()

    # Plot motif counts vs time
    fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)

    for df, title, axis in [
        (no_bias,"no constructors",ax[0]),
        (bias,"constructors active",ax[1])
    ]:
        axis.plot(df["time"], df["num_M1"], label="M1 (open triples)")
        axis.plot(df["time"], df["num_M2"], label="M2 (triangles)")
        axis.plot(df["time"], df["num_M3"], label="M3 (double triangles)")
        axis.set_title(title)
        axis.set_xlabel("time τ")

    ax[0].set_ylabel("# motifs")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("motif_counts_time.png", dpi=150)
    plt.close()

    # Plot cumulative repair events
    plt.figure(figsize=(8,4))
    plt.plot(no_bias["time"], no_bias["repair_events"], label="no constructors")
    plt.plot(bias["time"],    bias["repair_events"],    label="constructors active")
    plt.xlabel("model time τ")
    plt.ylabel("cumulative repair events")
    plt.legend()
    plt.tight_layout()
    plt.savefig("repair_events_time.png", dpi=150)
    plt.close()

    # Plot simple persistence proxy for triangles (M2)
    plt.figure(figsize=(8,4))
    plt.plot(no_bias["time"], no_bias["pers_M2"], label="M2 persistence (no constructors)")
    plt.plot(bias["time"],    bias["pers_M2"],    label="M2 persistence (constructors)")
    plt.xlabel("model time τ")
    plt.ylabel("consecutive presence of triangles (steps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("persistence_M2_time.png", dpi=150)
    plt.close()
