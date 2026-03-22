"""
simulation_final.py

Complete deterministic hypergraph rewrite system:
- N=3, N=4, N=5 experiments
- Assembly index computation
- Export for stochastic resilience experiments

Demonstrates phase transition at N=4 with superlinear scaling above threshold.
"""

import random
from collections import defaultdict, deque
from itertools import permutations, combinations
import json
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================

# Node pools
C_CORE_NODES = frozenset({0, 1, 2, 3})  # Constructor nodes {a,b,c,d}
SUB_NODES = frozenset({3, 4, 5, 6})      # Substrate nodes {d,e,f,r}
INTERFACE_NODE = 3                        # Shared node 'd'

# Constructor edges (fixed, never rewritten)
C_EDGES = frozenset({
    frozenset({0, 1, 2}),  # Executor 3-edge
    frozenset({0, 2}),      # Loop-closing 2-edge
    frozenset({2, 3})       # Interface 2-edge
})

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_active_substrate_nodes(config):
    """Return set of active substrate nodes in config."""
    active = set()
    for edge in config:
        active.update(edge & SUB_NODES)
    return active

def get_inactive_substrate_nodes(config):
    """Return set of inactive substrate nodes."""
    active = get_active_substrate_nodes(config)
    return SUB_NODES - active

def canonicalize(config):
    """
    Return canonical form of config (for isomorphism checking).
    Permute active substrate nodes only.
    """
    active = get_active_substrate_nodes(config)
    if not active:
        return "EMPTY"
    
    active_list = sorted(active)
    n = len(active_list)
    
    # Try all permutations of active nodes
    min_repr = None
    
    for perm in permutations(range(n)):
        mapping = {active_list[i]: active_list[perm[i]] for i in range(n)}
        
        remapped = frozenset(
            frozenset(mapping.get(node, node) for node in edge)
            for edge in config
        )
        
        repr_str = str(sorted(sorted(edge) for edge in remapped))
        
        if min_repr is None or repr_str < min_repr:
            min_repr = repr_str
    
    return min_repr

def c_neighborhood(config):
    """
    Compute constructor's neighborhood: substrate nodes within
    graph-distance-1 of interface node 3.
    """
    nbhd = {INTERFACE_NODE}
    
    for edge in config:
        if INTERFACE_NODE in edge:
            nbhd.update(edge & SUB_NODES)
    
    return frozenset(nbhd)

# ============================================================================
# RULE IMPLEMENTATIONS (Deterministic)
# ============================================================================

def apply_alpha(config):
    """Rule α: Activate inactive substrate node."""
    successors = []
    inactive = get_inactive_substrate_nodes(config)
    
    if not inactive:
        return successors
    
    for edge in config:
        if len(edge) == 2:  # 2-edge
            u, v = sorted(edge)
            for w in inactive:
                new_edge = frozenset({u, v, w})
                new_config = (config - {edge}) | {new_edge}
                successors.append(new_config)
    
    return successors

def apply_beta(config):
    """Rule β: Edge propagation."""
    successors = []
    
    hedges_3 = [e for e in config if len(e) == 3]
    edges_2 = [e for e in config if len(e) == 2]
    
    for hedge in hedges_3:
        for edge in edges_2:
            shared = hedge & edge
            if len(shared) == 1:  # Share exactly one node
                w = list(shared)[0]
                v = min(hedge - {w})
                x = list(edge - {w})[0]
                
                if x not in hedge:
                    new_hedge = frozenset({w, x, v})
                    new_config = config | {new_hedge}
                    successors.append(new_config)
    
    return successors

def apply_gamma(config, nbhd):
    """Rule γ: Merge two 3-edges sharing 2-subset (R_c)."""
    successors = []
    
    hedges_3 = [e for e in config if len(e) == 3]
    
    for h1, h2 in combinations(hedges_3, 2):
        shared = h1 & h2
        if len(shared) == 2:  # Share exactly 2 nodes
            # Check if both edges intersect neighborhood
            if not (h1 & nbhd and h2 & nbhd):
                continue
            
            merged = h1 | h2  # 4-edge
            new_config = (config - {h1, h2}) | {merged}
            successors.append(new_config)
    
    return successors

def apply_delta(config, nbhd):
    """Rule δ: Split 4-edge into two 3-edges (R_c)."""
    successors = []
    
    for edge in config:
        if len(edge) == 4:
            # Check if edge intersects neighborhood
            if not (edge & nbhd):
                continue
            
            nodes = sorted(edge)
            h1 = frozenset(nodes[:3])   # First 3 nodes
            h2 = frozenset(nodes[1:4])  # Last 3 nodes (overlap at middle 2)
            
            new_config = (config - {edge}) | {h1, h2}
            successors.append(new_config)
    
    return successors

def apply_epsilon(config, nbhd):
    """Rule ε: Loop closing (R_c)."""
    successors = []
    
    hedges_3 = [e for e in config if len(e) == 3]
    
    for h1, h2 in combinations(hedges_3, 2):
        shared = h1 & h2
        if len(shared) == 2:  # Share exactly 2 nodes
            # Check if both edges intersect neighborhood
            if not (h1 & nbhd and h2 & nbhd):
                continue
            
            u = list(h1 - h2)[0]
            x = list(h2 - h1)[0]
            closing_edge = frozenset({u, x})
            
            new_config = config | {closing_edge}
            successors.append(new_config)
    
    return successors

# ============================================================================
# BFS REACHABILITY ENUMERATION
# ============================================================================

def bfs_reachability(s0, rules, has_constructor=False, max_depth=None):
    """
    Exhaustive BFS enumeration of reachable configurations.
    
    Returns:
        reach: set of canonical config strings
        graph: dict mapping canonical -> list of canonical successors
        depths: dict mapping canonical -> BFS depth from s0
    """
    s0_canon = canonicalize(s0)
    
    reach = {s0_canon}
    graph = defaultdict(list)
    depths = {s0_canon: 0}
    
    frontier = deque([(s0, 0)])
    visited_configs = {s0_canon: s0}
    
    while frontier:
        config, depth = frontier.popleft()
        
        if max_depth is not None and depth >= max_depth:
            continue
        
        canon = canonicalize(config)
        nbhd = c_neighborhood(config) if has_constructor else frozenset()
        
        # Apply all rules
        successors = []
        for rule in rules:
            if rule in [apply_gamma, apply_delta, apply_epsilon]:
                successors.extend(rule(config, nbhd))
            else:
                successors.extend(rule(config))
        
        for succ in successors:
            succ_canon = canonicalize(succ)
            
            # Add edge to graph
            if succ_canon not in graph[canon]:
                graph[canon].append(succ_canon)
            
            # If new config, add to reachability
            if succ_canon not in reach:
                reach.add(succ_canon)
                depths[succ_canon] = depth + 1
                visited_configs[succ_canon] = succ
                frontier.append((succ, depth + 1))
    
    return reach, dict(graph), depths

# ============================================================================
# ASSEMBLY INDEX COMPUTATION
# ============================================================================

def compute_assembly_indices(reach, depths, max_depth_global):
    """
    Compute normalized assembly indices for all configs.
    
    Normalized AI = depth / max_depth_global
    """
    ai_dict = {}
    for canon in reach:
        depth = depths.get(canon, 0)
        ai_norm = depth / max_depth_global if max_depth_global > 0 else 0.0
        ai_dict[canon] = {
            'depth': depth,
            'ai_norm': ai_norm
        }
    return ai_dict

# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_experiment_n3():
    """
    N=3 experiment: Subcritical test.
    |SUB_NODES| = 3, seed = {{3,4}}
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: N=3 (Subcritical)")
    print("="*80)
    
    # Temporarily override SUB_NODES
    global SUB_NODES
    original_sub = SUB_NODES
    SUB_NODES = frozenset({3, 4, 5})
    
    s0 = frozenset({frozenset({3, 4})})
    
    # Run 1: R_free only
    print("\n[Run 1] R_free = {α, β} (no constructor)")
    R_free = [apply_alpha, apply_beta]
    reach_free, graph_free, depths_free = bfs_reachability(s0, R_free, has_constructor=False)
    print(f"  Reachable configs: {len(reach_free)}")
    print(f"  Max depth: {max(depths_free.values()) if depths_free else 0}")
    
    # Run 2: R_free + R_c (with constructor)
    print("\n[Run 2] R_free + R_c (with constructor)")
    R_c = [apply_gamma, apply_delta, apply_epsilon]
    R_all = R_free + R_c
    reach_gated, graph_gated, depths_gated = bfs_reachability(s0, R_all, has_constructor=True)
    print(f"  Reachable configs: {len(reach_gated)}")
    print(f"  Max depth: {max(depths_gated.values()) if depths_gated else 0}")
    
    # Compute Δ
    delta = reach_gated - reach_free
    print(f"\n|Δ| = {len(delta)}")
    print(f"Δ/Reach_gated = {len(delta)/len(reach_gated) if reach_gated else 0:.4f}")
    
    # Restore SUB_NODES
    SUB_NODES = original_sub
    
    return {
        'n': 3,
        'reach_free': len(reach_free),
        'reach_gated': len(reach_gated),
        'delta': len(delta),
        'delta_fraction': len(delta)/len(reach_gated) if reach_gated else 0
    }

def run_experiment_n4():
    """
    N=4 experiment: Critical threshold.
    |SUB_NODES| = 4, seed = {{3,4}, {3,5}}
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: N=4 (Critical Threshold)")
    print("="*80)
    
    s0 = frozenset({frozenset({3, 4}), frozenset({3, 5})})
    
    # Run 1: R_free only
    print("\n[Run 1] R_free = {α, β} (no constructor)")
    R_free = [apply_alpha, apply_beta]
    reach_free, graph_free, depths_free = bfs_reachability(s0, R_free, has_constructor=False)
    print(f"  Reachable configs: {len(reach_free)}")
    print(f"  Max depth: {max(depths_free.values()) if depths_free else 0}")
    
    # Run 2: R_free + R_c (with constructor)
    print("\n[Run 2] R_free + R_c (with constructor)")
    R_c = [apply_gamma, apply_delta, apply_epsilon]
    R_all = R_free + R_c
    reach_gated, graph_gated, depths_gated = bfs_reachability(s0, R_all, has_constructor=True)
    print(f"  Reachable configs: {len(reach_gated)}")
    print(f"  Max depth: {max(depths_gated.values()) if depths_gated else 0}")
    
    # Compute Δ
    delta = reach_gated - reach_free
    print(f"\n|Δ| = {len(delta)}")
    print(f"Δ/Reach_gated = {len(delta)/len(reach_gated) if reach_gated else 0:.4f}")
    
    # Assembly indices
    max_depth_global = max(max(depths_free.values()) if depths_free else 0,
                          max(depths_gated.values()) if depths_gated else 0)
    
    ai_free = compute_assembly_indices(reach_free, depths_free, max_depth_global)
    ai_gated = compute_assembly_indices(reach_gated, depths_gated, max_depth_global)
    ai_delta = {k: v for k, v in ai_gated.items() if k in delta}
    
    mean_ai_free = sum(v['ai_norm'] for v in ai_free.values()) / len(ai_free) if ai_free else 0
    mean_ai_delta = sum(v['ai_norm'] for v in ai_delta.values()) / len(ai_delta) if ai_delta else 0
    
    print(f"\nMean AI (R_free): {mean_ai_free:.3f}")
    print(f"Mean AI (Δ): {mean_ai_delta:.3f}")
    print(f"Asymmetry: {mean_ai_delta - mean_ai_free:+.3f}")
    
    return {
        'n': 4,
        'reach_free': len(reach_free),
        'reach_gated': len(reach_gated),
        'delta': len(delta),
        'delta_fraction': len(delta)/len(reach_gated) if reach_gated else 0,
        'mean_ai_free': mean_ai_free,
        'mean_ai_delta': mean_ai_delta,
        'ai_asymmetry': mean_ai_delta - mean_ai_free,
        'baseline_set': reach_free,
        'delta_set': delta,
        'reachability_graph': graph_gated,
        'depths': depths_gated
    }

def run_experiment_n5_sampling(n_trials=500, max_depth=50):
    """
    N=5 experiment: Supercritical (sampling-based).
    |SUB_NODES| = 5, seed = {{3,4}, {3,5}}
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: N=5 (Supercritical, Sampling)")
    print("="*80)
    
    # Temporarily override SUB_NODES
    global SUB_NODES
    original_sub = SUB_NODES
    SUB_NODES = frozenset({3, 4, 5, 6, 7})
    
    s0 = frozenset({frozenset({3, 4}), frozenset({3, 5})})
    
    print(f"\nRunning {n_trials} random walk trials (max depth {max_depth})...")
    
    # Sample reachability via random walks
    R_free = [apply_alpha, apply_beta]
    R_c = [apply_gamma, apply_delta, apply_epsilon]
    R_all = R_free + R_c
    
    sampled_free = set()
    sampled_gated = set()
    
    # Sample R_free
    for trial in range(n_trials):
        config = s0
        for step in range(max_depth):
            canon = canonicalize(config)
            sampled_free.add(canon)
            
            successors = []
            for rule in R_free:
                successors.extend(rule(config))
            
            if not successors:
                break
            
            config = random.choice(successors)
    
    # Sample R_gated
    for trial in range(n_trials):
        config = s0
        for step in range(max_depth):
            canon = canonicalize(config)
            sampled_gated.add(canon)
            
            nbhd = c_neighborhood(config)
            successors = []
            for rule in R_all:
                if rule in [apply_gamma, apply_delta, apply_epsilon]:
                    successors.extend(rule(config, nbhd))
                else:
                    successors.extend(rule(config))
            
            if not successors:
                break
            
            config = random.choice(successors)
    
    delta_sampled = sampled_gated - sampled_free
    
    print(f"\nSampled configs (R_free): {len(sampled_free)}")
    print(f"Sampled configs (R_gated): {len(sampled_gated)}")
    print(f"|Δ| (sampled): {len(delta_sampled)}")
    print(f"Δ/Reach_gated: {len(delta_sampled)/len(sampled_gated) if sampled_gated else 0:.4f}")
    
    # Restore SUB_NODES
    SUB_NODES = original_sub
    
    return {
        'n': 5,
        'reach_free': len(sampled_free),
        'reach_gated': len(sampled_gated),
        'delta': len(delta_sampled),
        'delta_fraction': len(delta_sampled)/len(sampled_gated) if sampled_gated else 0,
        'note': 'Sampling-based estimates'
    }

# ============================================================================
# EXPORT FOR STOCHASTIC
# ============================================================================

def export_for_stochastic(n4_results):
    """
    Export N=4 deterministic results for stochastic resilience experiments.
    """
    print("\n" + "="*80)
    print("EXPORTING FOR STOCHASTIC SIMULATION")
    print("="*80)
    
    baseline_set = list(n4_results['baseline_set'])
    delta_set = list(n4_results['delta_set'])
    graph = n4_results['reachability_graph']
    
    # Save to results/
    import os
    os.makedirs('results', exist_ok=True)
    
    with open('results/baseline_configs.json', 'w') as f:
        json.dump(baseline_set, f, indent=2)
    
    with open('results/delta_configs.json', 'w') as f:
        json.dump(delta_set, f, indent=2)
    
    with open('results/reachability_graph.pkl', 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"\n✓ Exported:")
    print(f"  Baseline: {len(baseline_set)} configs → results/baseline_configs.json")
    print(f"  Delta: {len(delta_set)} configs → results/delta_configs.json")
    print(f"  Graph: {len(graph)} nodes → results/reachability_graph.pkl")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("DETERMINISTIC HYPERGRAPH REWRITE SYSTEM")
    print("Assembly Theory + Constructor Theory Synthesis")
    print("="*80)
    
    # Run all experiments
    results_n3 = run_experiment_n3()
    results_n4 = run_experiment_n4()
    results_n5 = run_experiment_n5_sampling()
    
    # Summary table
    print("\n" + "="*80)
    print("SCALING SUMMARY")
    print("="*80)
    
    print(f"\n{'|SUB_NODES|':<15} {'|Reach_free|':<15} {'|Reach_gated|':<15} {'|Δ|':<10} {'Δ/Reach':<12} {'AI_asym':<10}")
    print("-" * 80)
    
    for r in [results_n3, results_n4, results_n5]:
        ai_asym = r.get('ai_asymmetry', 0.0)
        print(f"{r['n']:<15} {r['reach_free']:<15} {r['reach_gated']:<15} {r['delta']:<10} {r['delta_fraction']:<12.4f} {ai_asym:<10.3f}")
    
    # Export for stochastic
    export_for_stochastic(results_n4)
    
    print("\n" + "="*80)
    print("✓ DETERMINISTIC EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nNext step: Run stochastic resilience experiments")
    print("  python stochastic_resilience.py")

if __name__ == "__main__":
    main()
