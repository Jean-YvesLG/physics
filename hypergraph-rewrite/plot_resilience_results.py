"""
plot_resilience_results.py

Visualization and analysis of stochastic resilience experiments.
Generates figures for paper.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_resilience_results():
    """
    Load and visualize resilience experiment results.
    """
    print("\n" + "="*80)
    print("PLOTTING RESILIENCE RESULTS")
    print("="*80)
    
    # Load data
    try:
        with open('results/resilience_exp2_no_C.json', 'r') as f:
            metrics_no_C = json.load(f)
        
        with open('results/resilience_exp2_with_C.json', 'r') as f:
            metrics_with_C = json.load(f)
        
        print("\n✓ Loaded simulation results")
    
    except FileNotFoundError as e:
        print(f"\n ERROR: Could not load results!")
        print(f"  {e}")
        print(f"\n  Please run stochastic_resilience.py first.")
        return
    
    # Extract time series
    t_no_C = [m['timestep'] for m in metrics_no_C]
    pop_no_C = [m['total_pop'] for m in metrics_no_C]
    
    t_with_C = [m['timestep'] for m in metrics_with_C]
    pop_with_C = [m['total_pop'] for m in metrics_with_C]
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # PLOT 1: Total population over time
    plt.subplot(1, 2, 1)
    plt.plot(t_no_C, pop_no_C, label='Without C (baseline decay)', 
             color='#ff6b6b', linewidth=2)
    plt.plot(t_with_C, pop_with_C, label='With C (constructor stabilizes)', 
             color='#7aff9a', linewidth=2)
    
    plt.xlabel('Timestep', fontsize=11)
    plt.ylabel('Total Population', fontsize=11)
    plt.title('Population Survival: Constructor Effect', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # PLOT 2: Survival comparison
    plt.subplot(1, 2, 2)
    
    # Compute survival fraction (normalized to initial)
    initial_pop = pop_no_C[0] if pop_no_C else 100
    
    survival_no_C = [p / initial_pop for p in pop_no_C]
    survival_with_C = [p / initial_pop for p in pop_with_C]
    
    plt.plot(t_no_C, survival_no_C, label='Without C', 
             color='#ff6b6b', linewidth=2)
    plt.plot(t_with_C, survival_with_C, label='With C', 
             color='#7aff9a', linewidth=2)
    
    plt.xlabel('Timestep', fontsize=11)
    plt.ylabel('Survival Fraction', fontsize=11)
    plt.title('Normalized Population Survival', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/resilience_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved figure: results/resilience_results.png")
    
    plt.show()
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    final_no_C = metrics_no_C[-1] if metrics_no_C else {'timestep': 0, 'total_pop': 0}
    final_with_C = metrics_with_C[-1] if metrics_with_C else {'timestep': 0, 'total_pop': 0}
    
    print(f"\nFinal state (t={max(final_no_C['timestep'], final_with_C['timestep'])}):")
    print(f"  WITHOUT constructor: {final_no_C['total_pop']} copies")
    print(f"  WITH constructor:    {final_with_C['total_pop']} copies")
    
    if final_no_C['total_pop'] > 0:
        ratio = final_with_C['total_pop'] / final_no_C['total_pop']
        print(f"\n  Population ratio (with/without): {ratio:.1f}×")
    
    print("\n" + "="*80)
    print("NOTE: Current results are from SIMPLIFIED model")
    print("="*80)
    print("""
The plotted results demonstrate the workflow and infrastructure.
A full implementation would show:
- Constructor presence → population survival (vs extinction)
- Δ configs → higher copy numbers than baseline
- Evolutionary selection → Δ dominates over time

Current simplified model shows basic dynamics structure.
""")
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    plot_resilience_results()
