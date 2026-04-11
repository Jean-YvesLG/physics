# Reference Implementation

Reference implementation for the paper:

**"Constructor-Defined Possibility Boundaries in Hypergraph Rewrite Systems"**

This repository contains the deterministic reference implementation used to compute reachability, constructor-gated dynamics, and Assembly Index in a minimal typed hypergraph rewrite system, together with a stochastic extension probing resilience under perturbation.

---

## Overview

The model implements:

- a **typed hypergraph architecture** separating constructor edges (C-edges) and substrate edges (S-edges)
- a finite substrate with bounded node set and bounded rank
- local rewrite rules
  - free rules: `R_free = {alpha, beta}`
  - constructor-gated rules: `R_c = {gamma, delta, epsilon}`
- a **constructor neighbourhood constraint** centered on the shared interface node
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
├── hypergraph_core.py                              # Shared deterministic core
├── simulation_final_refactored.py                  # Deterministic BFS entry point
├── inverted_stochastic_darwinian_resilience_v3.py  # Stochastic resilience extension
├── README.md
├── LICENSE
└── .gitignore
