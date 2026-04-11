# Reference Implementation

Reference implementation for the paper:

**"Constructor-Defined Possibility Boundaries in Hypergraph Rewrite Systems"**

This repository contains the deterministic simulation code used to compute reachability, constructor-gated dynamics, and Assembly Index in a minimal typed hypergraph rewrite system.

---

## Overview

The model implements:

- a **typed hypergraph architecture** separating constructor edges (C-edges) and substrate edges (S-edges)
- a finite substrate with bounded node set and bounded rank
- local rewrite rules
  - free rules: `R_free = {alpha, beta}`
  - constructor-gated rules: `R_c = {gamma, delta, epsilon}`
- a **constructor neighbourhood constraint** centered on the interface node
- **breadth-first search (BFS)** enumeration of reachable configurations
- **canonicalization under isomorphism** over active substrate nodes only
- **Assembly Index** computed as minimal derivation depth from the seed configuration

The central object computed is:

`Delta = Reach(S0, R_free ∪ R_c, C) \ Reach(S0, R_free)`

---

## Key Features

- exact BFS enumeration of configuration space
- canonicalization over substrate-node relabelings
- typed separation enforcing constructor retention
- locality-constrained rule activation
- deterministic reproduction of the reported N=4 results

---

## Repository Structure

```text
.
├── hypergraph_core.py                 # Shared deterministic core
├── simulation_final_refactored.py     # Deterministic BFS entry point
├── plot_static_figures_refactored.py  # Static figure generation
├── README.md
├── LICENSE
└── .gitignore
```

---

## Quick Start

```bash
python simulation_final_refactored.py
```

This will:

- enumerate reachable configurations under baseline and constructor-gated dynamics
- compute minimal depths / Assembly Index values
- report summary statistics consistent with the paper

To generate static figures:

```bash
python plot_static_figures_refactored.py
```

---

## Core Concepts

### Configuration

A configuration is a finite set of substrate hyperedges:

- nodes are drawn from `SUB_NODES`
- valid hyperedge arities are 2, 3, and 4
- constructor edges are fixed in the background and are not stored as part of the evolving substrate configuration

### Canonicalization

Configurations are identified up to relabeling of **active substrate nodes only**.

This ensures that:

- structural identity is preserved
- equivalent labelings are not double-counted during BFS
- Assembly Index is computed over structural equivalence classes rather than literal node labels

### Constructor Gating

Rules in `R_c` apply only when the matched substrate pattern intersects the constructor neighbourhood.

This enforces:

- locality of constructive influence
- a nontrivial possibility boundary
- strict separation between baseline and constructor-enabled reachability

### Assembly Index

Assembly Index is operationalized here as:

> the minimal number of valid rule applications required to derive a configuration from `S0`

It is computed exactly as BFS depth in the finite reachability graph.

---

## Deterministic N=4 Results

These are the fully enumerated results for the `|SUB_NODES| = 4` instance reported in the paper.

| Quantity | Value |
|---|---:|
| Reach_free | 3 |
| Reach_gated | 106 |
| Delta | 103 |
| Delta / Reach_gated | 0.9717 |
| Mean normalized AI — R_free | 0.0714 |
| Mean normalized AI — Delta | 0.6110 |
| Asymmetry | +0.540 |
| Repro-pattern configs in Delta | 92 |
| Type-system violations | 0 |
| Retention check | PASS |

Here:

- `Reach_free = Reach(S0, R_free)`
- `Reach_gated = Reach(S0, R_free ∪ R_c, C)`
- `Delta = Reach_gated \ Reach_free`

---

## Reproducibility

The implementation provides:

- exact reachability enumeration
- minimal-path Assembly Index values
- typed constructor/substrate separation
- locality-constrained constructor gating

The shared deterministic semantics are defined in `hypergraph_core.py`.

---

## Historical Note

Earlier exploratory versions of this project used stochastic graph-based motif dynamics and free-energy-like scoring. Those prototypes have been superseded by the current typed hypergraph rewrite formulation, which is the version aligned with the paper.

---

## License

MIT License
