import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bây giờ mới import Config
from utils.config import Config

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_basic_metrics(G):
    """Calculate comprehensive network metrics"""
    print("Calculating basic network metrics...")
    
    metrics = {}
    
    # Basic properties
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    metrics['directed'] = G.is_directed()
    
    # Degree statistics
    if G.is_directed():
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        metrics['avg_in_degree'] = np.mean(in_degrees)
        metrics['avg_out_degree'] = np.mean(out_degrees)
        metrics['max_in_degree'] = np.max(in_degrees)
        metrics['max_out_degree'] = np.max(out_degrees)
    else:
        degrees = [d for _, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['max_degree'] = np.max(degrees)
    
    # Connectivity
    if G.is_directed():
        metrics['weakly_connected'] = nx.is_weakly_connected(G)
        metrics['strongly_connected'] = nx.is_strongly_connected(G)
        
        # Weakly connected components
        wcc = list(nx.weakly_connected_components(G))
        metrics['num_weakly_components'] = len(wcc)
        largest_wcc = max(wcc, key=len)
        metrics['largest_wcc_size'] = len(largest_wcc)
        
        # Strongly connected components
        scc = list(nx.strongly_connected_components(G))
        metrics['num_strongly_components'] = len(scc)
        if scc:
            largest_scc = max(scc, key=len)
            metrics['largest_scc_size'] = len(largest_scc)
        else:
            metrics['largest_scc_size'] = 0
            
    else:
        metrics['connected'] = nx.is_connected(G)
        components = list(nx.connected_components(G))
        metrics['num_components'] = len(components)
        largest_cc = max(components, key=len)
        metrics['largest_cc_size'] = len(largest_cc)
    
    # Diameter and radius (for largest connected component)
    if G.is_directed():
        G_main = G.subgraph(largest_wcc)
    else:
        G_main = G.subgraph(largest_cc)
    
    try:
        if G_main.number_of_nodes() > 1:
            metrics['diameter'] = nx.diameter(G_main)
            metrics['radius'] = nx.radius(G_main)
        else:
            metrics['diameter'] = 0
            metrics['radius'] = 0
    except:
        metrics['diameter'] = float('inf')
        metrics['radius'] = float('inf')
    
    # Clustering coefficients
    metrics['avg_clustering'] = nx.average_clustering(G)
    
    # Average shortest path length
    try:
        if G_main.number_of_nodes() > 1:
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G_main)
        else:
            metrics['avg_shortest_path'] = 0
    except:
        metrics['avg_shortest_path'] = float('inf')
    
    # Reciprocity (for directed graphs)
    if G.is_directed():
        metrics['reciprocity'] = nx.reciprocity(G)
    
    # Assortativity
    try:
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        metrics['degree_assortativity'] = 0
    
    return metrics

def plot_network_properties(G, metrics, save_dir):
    """Create various plots for network properties"""
    
    # 1. Degree distribution
    plt.figure(figsize=(15, 10))
    
    if G.is_directed():
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        
        plt.subplot(2, 3, 1)
        plt.hist(in_degrees, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('In-Degree')
        plt.ylabel('Frequency')
        plt.title('In-Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.hist(out_degrees, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Out-Degree')
        plt.ylabel('Frequency')
        plt.title('Out-Degree Distribution')
        plt.grid(True, alpha=0.3)
        
    else:
        degrees = [d for _, d in G.degree()]
        plt.subplot(2, 3, 1)
        plt.hist(degrees, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)
    
    # 2. Clustering coefficient distribution
    clustering_coeffs = list(nx.clustering(G).values())
    plt.subplot(2, 3, 2)
    plt.hist(clustering_coeffs, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.title('Clustering Coefficient Distribution')
    plt.grid(True, alpha=0.3)
    
    # 3. Connected components size distribution
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    component_sizes = [len(comp) for comp in components]
    plt.subplot(2, 3, 3)
    plt.hist(component_sizes, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Component Size')
    plt.ylabel('Frequency')
    plt.title('Connected Components Size Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. Summary table
    plt.subplot(2, 3, 4)
    plt.axis('off')
    
    summary_text = "NETWORK SUMMARY\n\n"
    summary_text += f"Nodes: {metrics['nodes']}\n"
    summary_text += f"Edges: {metrics['edges']}\n"
    summary_text += f"Density: {metrics['density']:.6f}\n"
    summary_text += f"Directed: {metrics['directed']}\n"
    
    if G.is_directed():
        summary_text += f"Avg In-Degree: {metrics['avg_in_degree']:.2f}\n"
        summary_text += f"Avg Out-Degree: {metrics['avg_out_degree']:.2f}\n"
        summary_text += f"Reciprocity: {metrics['reciprocity']:.3f}\n"
    else:
        summary_text += f"Avg Degree: {metrics['avg_degree']:.2f}\n"
    
    summary_text += f"Avg Clustering: {metrics['avg_clustering']:.3f}\n"
    summary_text += f"Assortativity: {metrics['degree_assortativity']:.3f}\n"
    
    if 'avg_shortest_path' in metrics and np.isfinite(metrics['avg_shortest_path']):
        summary_text += f"Avg Path Length: {metrics['avg_shortest_path']:.2f}\n"
    
    plt.text(0.1, 0.9, summary_text, fontsize=12, fontfamily='monospace',
             verticalalignment='top', transform=plt.gca().transAxes)
    
    # 5. Network type visualization
    plt.subplot(2, 3, 5)
    if G.is_directed():
        # Plot small example of directed relationships
        if G.number_of_nodes() > 50:
            # Take a sample for visualization
            sample_nodes = list(G.nodes())[:50]
            G_sample = G.subgraph(sample_nodes)
        else:
            G_sample = G
            
        pos = nx.spring_layout(G_sample, seed=42)
        nx.draw(G_sample, pos, node_size=50, node_color='lightblue', 
                arrows=True, arrowsize=10, edge_color='gray', alpha=0.7)
        plt.title('Directed Network Sample')
    else:
        # Plot small example of undirected relationships
        if G.number_of_nodes() > 50:
            sample_nodes = list(G.nodes())[:50]
            G_sample = G.subgraph(sample_nodes)
        else:
            G_sample = G
            
        pos = nx.spring_layout(G_sample, seed=42)
        nx.draw(G_sample, pos, node_size=50, node_color='lightgreen', 
                edge_color='gray', alpha=0.7)
        plt.title('Undirected Network Sample')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "network_basic_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved basic analysis plots")

def export_metrics_to_csv(metrics, save_path):
    """Export metrics to CSV file"""
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
    print(f"✅ Metrics exported to: {save_path}")

def main(config=None):
    """Main function for basic analysis"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("BASIC NETWORK ANALYSIS")
    print("="*60)
    
    G = config.get_network()
    
    if G is None:
        print("❌ No network data found. Run data loading first.")
        return None
    
    print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"📊 Network type: {'Directed' if G.is_directed() else 'Undirected'}")
    
    # Calculate metrics
    metrics = calculate_basic_metrics(G)
    
    # Display results
    print("\n📈 NETWORK METRICS:")
    print("-" * 40)
    
    categories = {
        'Basic Properties': ['nodes', 'edges', 'density', 'directed'],
        'Degree Statistics': ['avg_degree', 'avg_in_degree', 'avg_out_degree', 'max_degree', 'max_in_degree', 'max_out_degree'],
        'Connectivity': ['connected', 'weakly_connected', 'strongly_connected', 'num_components', 'num_weakly_components', 'num_strongly_components'],
        'Path Analysis': ['diameter', 'radius', 'avg_shortest_path'],
        'Clustering & Assortativity': ['avg_clustering', 'degree_assortativity', 'reciprocity']
    }
    
    for category, fields in categories.items():
        print(f"\n{category}:")
        for field in fields:
            if field in metrics and metrics[field] is not None:
                value = metrics[field]
                if isinstance(value, float):
                    print(f"  {field}: {value:.4f}")
                else:
                    print(f"  {field}: {value}")
    
    # Create visualizations
    plots_dir = os.path.join(config.RESULTS_DIR, "charts")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_network_properties(G, metrics, plots_dir)
    
    # Export metrics
    data_dir = os.path.join(config.RESULTS_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    export_metrics_to_csv(metrics, os.path.join(data_dir, "basic_metrics.csv"))
    
    # Save metrics to config
    config.set_metrics(metrics)
    
    # Additional insights
    print("\n💡 NETWORK INSIGHTS:")
    if G.is_directed():
        if metrics['reciprocity'] > 0.3:
            print("  • High reciprocity: Mutual relationships are common")
        else:
            print("  • Low reciprocity: Relationships are mostly one-directional")
    
    if metrics['avg_clustering'] > 0.1:
        print("  • High clustering: Nodes tend to form tight clusters")
    else:
        print("  • Low clustering: Sparse local connections")
    
    if abs(metrics['degree_assortativity']) > 0.1:
        if metrics['degree_assortativity'] > 0:
            print("  • Assortative: High-degree nodes connect to other high-degree nodes")
        else:
            print("  • Disassortative: High-degree nodes connect to low-degree nodes")
    else:
        print("  • Neutral mixing: No strong degree correlation")
    
    return metrics

if __name__ == "__main__":
    config = Config()
    
    # Load sample network if not exists
    if config.get_network() is None:
        from data.load_data import load_sample_network
        G = load_sample_network()
        config.set_network(G)
    
    main(config)