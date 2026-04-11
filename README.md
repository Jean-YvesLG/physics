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
