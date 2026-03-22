# Hypergraph Rewrite Model: Constructor-Mediated Complexity

**Computational demonstration that constructor-mediated organization emerges generically in discrete causal substrates.**

## Summary

This model tests whether assembly-theoretic biosignatures—high assembly index, elevated copy number, constructor-gated complexity asymmetry—emerge from minimal causal ingredients: discrete state spaces, local rewrite rules, and constructor-mediated feedback.

**Key Results:**
- **Phase transition at N=4**: Constructor effect emerges abruptly (Δ = 103/106 configs, 97% exclusivity)
- **Superlinear scaling above threshold**: N=5 shows 74× growth (Δ ≈ 7,891, 99.94% exclusivity)
- **Resilience under noise**: Constructor presence enables population survival, error correction, and evolutionary selection
- **All four Marletto criteria verified**: Task specification, locality, retention, resilience

## Files

### Core Simulation Code
- **`simulation_final.py`**: Deterministic hypergraph rewrite system with N=3, N=4, N=5 experiments
- **`stochastic_resilience.py`**: Stochastic extension with probabilistic rules, decay, and copy number tracking
- **`plot_resilience_results.py`**: Visualization and analysis tools

### Reference Experiments
- **`experiment_1_n3.py`**: N=3 subcritical test (geometric constraint, Δ=0)
- **`experiment_2_n5.py`**: N=5 supercritical test (sampling-based, Δ≈7,891)

### Interactive Visualizations
- **`visualizations/hypergraph_viz.html`**: N=4 dynamics (3 phases)
- **`visualizations/typed_edge_viz.html`**: Architectural discovery (5 phases)
- **`visualizations/hypergraph_viz_scaling.html`**: Phase transition N=3→N=4→N=5 (3 phases)

## Quick Start

### Run Deterministic Experiments
```bash
python simulation_final.py
```

Outputs:
- `results/baseline_configs.json` (3 configs from Run 1)
- `results/delta_configs.json` (103 configs from Run 2 Δ)
- `results/reachability_graph.pkl` (for assembly index computation)

### Run Stochastic Resilience Experiments
```bash
python stochastic_resilience.py
```

Outputs:
- `results/resilience_exp2_no_C.json` (population without constructor)
- `results/resilience_exp2_with_C.json` (population with constructor)

### Generate Plots
```bash
python plot_resilience_results.py
```

Outputs:
- `results/resilience_results.png` (population survival + composition dynamics)

## Requirements
```bash
pip install networkx matplotlib numpy pandas
```

## Citation

If you use this code, please cite:
```
[Author]. (2026). Constructor-Defined Possibility Boundaries in Hypergraph 
Rewrite Systems: A Formal Precursor to Assembly-Theoretic Biosignatures. 
Artificial Life [in review].
```

## License

MIT License - see LICENSE file for details.

## Contact

GitHub: https://github.com/Jean-YvesLG/physics
