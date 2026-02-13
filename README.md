**Reference implementation** for the paper _"Constructor-Modulated Causal Substrates and Life-Like Motifs in Discrete Hypergraph Rewrite Systems"_.

This repo contains the complete simulation code from my Gist, now properly structured as a full GitHub repository. Models simple undirected graphs as hypergraph substrates with motif detection (M1 open triples, M2 triangles, M3 double triangles), rewrite rules (R1 closure, R2 duplication, R3 decay), constructor bias via M3s, and free-energy functional `F = E - T_eff * S`.

[

## Features

- Complete motif detection (M1, M2, M3)
    
- Stochastic rewrite rules with constructor bias
    
- Free-energy-like functional tracking
    
- Persistence + repair event metrics
    
- CSV outputs + matplotlib plots matching paper figures
    

## Quick Start

bash

`pip install matplotlib pandas python physics.py`

Generates:

- `simulation_no_bias.csv` / `simulation_bias.csv`
    
- Plots: `F_time_windowed.png`, `motif_counts_time.png`, etc.
    

## Structure

text

`physics/ ├── physics.py          # Main simulation + plotting (from Gist) ├── README.md ├── LICENSE            # MIT   └── .gitignore         # Python template`

## Usage

Key function: `run_simulation(G0, T_steps=300, use_bias=True, seed=1)`

Tracks per timestep:

- `F`: Free energy functional
    
- `num_M1/2/3`: Motif counts
    
- `pers_M*`: Consecutive motif presence
    
- `repair_events`: Cumulative repairs post-decay
    

## Paper Reproduction

Run `python physics.py` to generate exact figure data:

- Free energy vs time
    
- Motif count evolution
    
- Repair event dynamics
    
- Triangle persistence
    

## Customization

text

`T_steps=300      # Simulation length alpha=0.5, gamma=0.5  # Constructor bias strength window=20        # Branching measure window base_T=1.0, beta=0.1  # Effective temperature`

## Related Work

- [My-first-repo](https://github.com/Jean-YvesLG/My-first-repo) (general playground)
    
- Original Gist → imported here with full commit history
    

## License

MIT License - see [LICENSE](https://www.perplexity.ai/search/LICENSE)
