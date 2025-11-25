import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_random_models(G, num_samples=5):
    """Generate random graph models for comparison"""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = m / (n * (n - 1))  # density
    
    random_models = {}
    
    # 1. Erdős-Rényi model
    print("Generating Erdős-Rényi random graphs...")
    er_graphs = []
    for i in range(num_samples):
        G_er = nx.erdos_renyi_graph(n, p, seed=42+i)
        if G.is_directed():
            G_er = G_er.to_directed()
        er_graphs.append(G_er)
    random_models['erdos_renyi'] = er_graphs
    
    # 2. Configuration model (preserving degree sequence)
    print("Generating configuration model graphs...")
    if G.is_directed():
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        config_graphs = []
        for i in range(num_samples):
            try:
                G_config = nx.directed_configuration_model(in_degrees, out_degrees, seed=42+i)
                G_config = nx.DiGraph(G_config)  # Remove parallel edges
                config_graphs.append(G_config)
            except:
                print(f"⚠️ Configuration model failed for sample {i}, using ER instead")
                config_graphs.append(nx.erdos_renyi_graph(n, p, seed=42+i))
        random_models['configuration'] = config_graphs
    else:
        degrees = [d for _, d in G.degree()]
        config_graphs = []
        for i in range(num_samples):
            try:
                G_config = nx.configuration_model(degrees, seed=42+i)
                G_config = nx.Graph(G_config)  # Remove parallel edges
                G_config.remove_edges_from(nx.selfloop_edges(G_config))
                config_graphs.append(G_config)
            except:
                print(f"⚠️ Configuration model failed for sample {i}, using ER instead")
                config_graphs.append(nx.erdos_renyi_graph(n, p, seed=42+i))
        random_models['configuration'] = config_graphs
    
    return random_models

def calculate_comparison_metrics(G, random_models):
    """Calculate comparison metrics between real and random networks"""
    metrics = {}
    
    # Real network metrics
    real_metrics = {
        'avg_clustering': nx.average_clustering(G),
        'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
        'assortativity': nx.degree_assortativity_coefficient(G),
        'density': nx.density(G)
    }
    metrics['real'] = real_metrics
    
    # Random models metrics
    for model_name, graphs in random_models.items():
        model_metrics = {
            'avg_clustering': [],
            'avg_shortest_path': [], 
            'assortativity': [],
            'density': []
        }
        
        for G_rand in graphs:
            try:
                model_metrics['avg_clustering'].append(nx.average_clustering(G_rand))
            except:
                model_metrics['avg_clustering'].append(0)
            
            try:
                if nx.is_connected(G_rand):
                    model_metrics['avg_shortest_path'].append(nx.average_shortest_path_length(G_rand))
                else:
                    model_metrics['avg_shortest_path'].append(float('inf'))
            except:
                model_metrics['avg_shortest_path'].append(float('inf'))
            
            try:
                model_metrics['assortativity'].append(nx.degree_assortativity_coefficient(G_rand))
            except:
                model_metrics['assortativity'].append(0)
            
            model_metrics['density'].append(nx.density(G_rand))
        
        # Calculate averages
        avg_metrics = {}
        for key, values in model_metrics.items():
            valid_values = [v for v in values if np.isfinite(v)]
            avg_metrics[key] = np.mean(valid_values) if valid_values else 0
        
        metrics[model_name] = avg_metrics
    
    return metrics

def plot_comparison(metrics, save_path=None):
    """Plot comparison between real and random networks"""
    models = list(metrics.keys())
    metric_names = ['avg_clustering', 'avg_shortest_path', 'assortativity', 'density']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_names):
        if i < len(axes):
            values = [metrics[model][metric] for model in models]
            
            # Tạo bar plot
            bars = axes[i].bar(models, values, alpha=0.7, color=['red'] + ['blue']*(len(models)-1))
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Highlight real network
            bars[0].set_color('red')
            bars[0].set_alpha(0.8)
    
    plt.tight_layout()
    plt.suptitle('Real Network vs Random Models Comparison', y=1.02, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved random comparison: {save_path}")
    plt.close()

def main(config=None):
    """Main function for random graph comparison"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("RANDOM GRAPH COMPARISON")
    print("="*60)
    
    G = config.get_network()
    
    if G is None:
        print("❌ No network data found. Run data loading first.")
        return None
    
    print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Tạo random models
    print("\n🔄 Generating random graph models...")
    random_models = generate_random_models(G, num_samples=3)
    
    # Tính metrics so sánh
    print("📊 Calculating comparison metrics...")
    comparison_metrics = calculate_comparison_metrics(G, random_models)
    
    # Hiển thị kết quả
    print("\n📈 COMPARISON RESULTS:")
    print("\nReal Network vs Random Models:")
    
    metrics_df = pd.DataFrame(comparison_metrics).T
    print(metrics_df.round(4))
    
    # Phân tích small-worldness
    real_clustering = comparison_metrics['real']['avg_clustering']
    real_path = comparison_metrics['real']['avg_shortest_path']
    er_clustering = comparison_metrics['erdos_renyi']['avg_clustering']
    er_path = comparison_metrics['erdos_renyi']['avg_shortest_path']
    
    if er_clustering > 0 and er_path > 0:
        small_world_ratio = (real_clustering / er_clustering) / (real_path / er_path)
        print(f"\n🔍 Small-world Ratio: {small_world_ratio:.4f}")
        if small_world_ratio > 1:
            print("   → Network exhibits small-world properties")
        else:
            print("   → Network does not exhibit strong small-world properties")
    
    # Tạo visualization
    plot_comparison(
        comparison_metrics,
        save_path=config.RESULTS_DIR + "/charts/random_comparison.png"
    )
    
    return comparison_metrics

if __name__ == "__main__":
    config = Config()
    
    # Load sample network nếu chưa có
    if config.get_network() is None:
        from data.load_data import load_sample_network
        G = load_sample_network()
        config.set_network(G)
    
    main(config)