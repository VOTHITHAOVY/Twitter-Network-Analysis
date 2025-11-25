import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

def compute_k_core_decomposition(G):
    """Compute k-core decomposition of the network"""
    print("Computing k-core decomposition...")
    
    if G.is_directed():
        # Với directed graphs, sử dụng undirected version
        G_undirected = G.to_undirected()
        core_numbers = nx.core_number(G_undirected)
    else:
        core_numbers = nx.core_number(G)
    
    return core_numbers

def analyze_k_core_structure(core_numbers, G):
    """Analyze k-core structure"""
    core_distribution = Counter(core_numbers.values())
    
    analysis = {
        'max_core': max(core_numbers.values()) if core_numbers else 0,
        'min_core': min(core_numbers.values()) if core_numbers else 0,
        'core_distribution': dict(core_distribution),
        'avg_core_number': np.mean(list(core_numbers.values())) if core_numbers else 0,
        'nodes_in_max_core': sum(1 for k in core_numbers.values() if k == max(core_numbers.values()))
    }
    
    return analysis

def extract_k_cores(G, core_numbers, k_values=None):
    """Extract specific k-cores from the network"""
    if k_values is None:
        k_values = [1, 2, 3, 5]  # Default k values to extract
    
    k_cores = {}
    for k in k_values:
        if k <= max(core_numbers.values()):
            # Lấy subgraph của k-core
            k_core_nodes = [node for node, core in core_numbers.items() if core >= k]
            k_cores[k] = G.subgraph(k_core_nodes)
        else:
            k_cores[k] = None
    
    return k_cores

def plot_k_core_analysis(core_numbers, analysis, save_path=None):
    """Plot k-core analysis results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Core number distribution
    cores = list(core_numbers.values())
    ax1.hist(cores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_xlabel('Core Number (k)')
    ax1.set_ylabel('Number of Nodes')
    ax1.set_title('Distribution of Core Numbers')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative core distribution
    core_dist = analysis['core_distribution']
    k_values = sorted(core_dist.keys())
    cumulative_nodes = [sum(core_dist[k] for k in k_values if k >= k_min) for k_min in k_values]
    
    ax2.plot(k_values, cumulative_nodes, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Minimum Core Number (k)')
    ax2.set_ylabel('Number of Nodes (≥ k)')
    ax2.set_title('Cumulative Core Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. K-core sizes
    k_sizes = {k: core_dist[k] for k in k_values}
    ax3.bar(k_values, [k_sizes[k] for k in k_values], alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Core Number (k)')
    ax3.set_ylabel('Number of Nodes')
    ax3.set_title('K-Core Sizes')
    ax3.grid(True, alpha=0.3)
    
    # 4. Text summary
    ax4.axis('off')
    summary_text = f"""
    K-Core Analysis Summary
    
    Maximum k-core: {analysis['max_core']}
    Minimum k-core: {analysis['min_core']}
    Average core number: {analysis['avg_core_number']:.2f}
    Nodes in max k-core: {analysis['nodes_in_max_core']}
    Total k-cores: {len(analysis['core_distribution'])}
    
    Core Number Distribution:
    """
    
    # Thêm distribution details
    for k, count in sorted(analysis['core_distribution'].items())[:10]:  # Hiển thị 10 core đầu
        summary_text += f"\nk={k}: {count} nodes"
    
    if len(analysis['core_distribution']) > 10:
        summary_text += f"\n... and {len(analysis['core_distribution']) - 10} more"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('K-Core Decomposition Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved k-core analysis: {save_path}")
    plt.close()

def analyze_k_core_properties(k_cores, G):
    """Analyze properties of different k-cores"""
    properties = {}
    
    for k, k_core in k_cores.items():
        if k_core is not None and k_core.number_of_nodes() > 0:
            properties[k] = {
                'nodes': k_core.number_of_nodes(),
                'edges': k_core.number_of_edges(),
                'density': nx.density(k_core),
                'avg_degree': sum(dict(k_core.degree()).values()) / k_core.number_of_nodes(),
                'clustering': nx.average_clustering(k_core)
            }
    
    return properties

def main(config=None):
    """Main function for k-core analysis"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("K-CORE DECOMPOSITION ANALYSIS")
    print("="*60)
    
    G = config.get_network()
    
    if G is None:
        print("❌ No network data found. Run data loading first.")
        return None
    
    print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Tính k-core decomposition
    core_numbers = compute_k_core_decomposition(G)
    
    # Phân tích cấu trúc
    k_core_analysis = analyze_k_core_structure(core_numbers, G)
    
    print("\n📊 K-CORE ANALYSIS RESULTS:")
    print(f"Maximum k-core: {k_core_analysis['max_core']}")
    print(f"Minimum k-core: {k_core_analysis['min_core']}")
    print(f"Average core number: {k_core_analysis['avg_core_number']:.2f}")
    print(f"Nodes in maximum k-core: {k_core_analysis['nodes_in_max_core']}")
    print(f"Number of distinct k-cores: {len(k_core_analysis['core_distribution'])}")
    
    # Trích xuất các k-core quan trọng
    important_ks = [1, 2, 3, 5, 10]
    important_ks = [k for k in important_ks if k <= k_core_analysis['max_core']]
    k_cores = extract_k_cores(G, core_numbers, important_ks)
    
    # Phân tích properties của các k-core
    k_core_properties = analyze_k_core_properties(k_cores, G)
    
    print("\n🔍 K-CORE PROPERTIES:")
    for k, props in k_core_properties.items():
        print(f"\nk-core {k}:")
        print(f"  Nodes: {props['nodes']}")
        print(f"  Edges: {props['edges']}")
        print(f"  Density: {props['density']:.6f}")
        print(f"  Avg Degree: {props['avg_degree']:.2f}")
        print(f"  Clustering: {props['clustering']:.4f}")
    
    # Tạo visualizations
    plot_k_core_analysis(
        core_numbers, k_core_analysis,
        save_path=config.RESULTS_DIR + "/charts/k_core_analysis.png"
    )
    
    # Phân tích k-core vs centrality (nếu có)
    if config.CENTRALITY:
        print("\n📈 Analyzing k-core vs centrality...")
        centrality = config.CENTRALITY['degree']  # Sử dụng degree centrality
        
        # Tính average centrality cho mỗi k-core
        k_core_centrality = {}
        for k in range(1, k_core_analysis['max_core'] + 1):
            k_core_nodes = [node for node, core in core_numbers.items() if core == k]
            if k_core_nodes:
                centralities = [centrality[node] for node in k_core_nodes if node in centrality]
                if centralities:
                    k_core_centrality[k] = np.mean(centralities)
        
        # Plot k-core vs centrality
        if k_core_centrality:
            plt.figure(figsize=(10, 6))
            plt.plot(list(k_core_centrality.keys()), list(k_core_centrality.values()), 'o-', linewidth=2)
            plt.xlabel('Core Number (k)')
            plt.ylabel('Average Degree Centrality')
            plt.title('K-Core Number vs Centrality')
            plt.grid(True, alpha=0.3)
            plt.savefig(config.RESULTS_DIR + "/charts/k_core_centrality.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved k-core vs centrality plot")
    
    results = {
        'core_numbers': core_numbers,
        'analysis': k_core_analysis,
        'k_cores': k_cores,
        'k_core_properties': k_core_properties
    }
    
    return results

if __name__ == "__main__":
    config = Config()
    
    # Load sample network nếu chưa có
    if config.get_network() is None:
        from data.load_data import load_sample_network
        G = load_sample_network()
        config.set_network(G)
    
    main(config)