import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats

def calculate_centrality_measures(G):
    """Calculate comprehensive centrality measures"""
    print("Calculating centrality measures...")
    
    centrality = {}
    n_nodes = G.number_of_nodes()
    
    # 1. Degree Centrality
    print("  📊 Degree centrality...")
    centrality['degree'] = nx.degree_centrality(G)
    
    if G.is_directed():
        centrality['in_degree'] = nx.in_degree_centrality(G)
        centrality['out_degree'] = nx.out_degree_centrality(G)
    
    # 2. Betweenness Centrality (with sampling for large networks)
    print("  🔗 Betweenness centrality...")
    if n_nodes > 1000:
        # Use sampling for large networks
        k = min(500, n_nodes)
        centrality['betweenness'] = nx.betweenness_centrality(G, k=k, seed=42)
        print(f"    (Sampled {k} nodes for approximation)")
    else:
        centrality['betweenness'] = nx.betweenness_centrality(G)
    
    # 3. Closeness Centrality
    print("  📏 Closeness centrality...")
    try:
        centrality['closeness'] = nx.closeness_centrality(G)
    except:
        # For disconnected graphs, use harmonic closeness
        print("    Using harmonic closeness for disconnected graph...")
        centrality['closeness'] = nx.harmonic_centrality(G)
    
    # 4. Eigenvector Centrality
    print("  🔢 Eigenvector centrality...")
    try:
        centrality['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("    Eigenvector failed, using Katz centrality...")
        centrality['eigenvector'] = nx.katz_centrality(G, max_iter=1000)
    
    # 5. PageRank
    print("  🌐 PageRank...")
    centrality['pagerank'] = nx.pagerank(G, alpha=0.85)
    
    # 6. Additional measures for directed networks
    if G.is_directed():
        print("  🔄 HITS algorithm...")
        try:
            hits_hubs, hits_authorities = nx.hits(G, max_iter=100)
            centrality['hubs'] = hits_hubs
            centrality['authorities'] = hits_authorities
        except:
            print("    HITS failed, skipping...")
    
    print("✅ All centrality measures calculated!")
    return centrality

def get_top_central_nodes(centrality_dict, G, top_n=10):
    """Get top nodes for each centrality measure"""
    top_nodes = {}
    
    for measure, values in centrality_dict.items():
        # Sort nodes by centrality value
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_nodes[measure] = sorted_nodes
    
    return top_nodes

def analyze_centrality_distributions(centrality_dict):
    """Analyze distributions of centrality measures"""
    distributions = {}
    
    for measure, values in centrality_dict.items():
        vals = list(values.values())
        distributions[measure] = {
            'mean': np.mean(vals),
            'median': np.median(vals),
            'std': np.std(vals),
            'max': np.max(vals),
            'min': np.min(vals),
            'skewness': stats.skew(vals) if len(vals) > 1 else 0
        }
    
    return distributions

def plot_centrality_analysis(centrality_dict, top_nodes, G, save_dir):
    """Create comprehensive centrality analysis plots"""
    
    # Create subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Centrality distributions
    measures = list(centrality_dict.keys())
    n_measures = len(measures)
    
    # Determine grid size
    n_cols = 4
    n_rows = (n_measures + n_cols - 1) // n_cols
    
    # Plot distributions
    for i, measure in enumerate(measures):
        plt.subplot(n_rows, n_cols, i + 1)
        values = list(centrality_dict[measure].values())
        
        plt.hist(values, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel(f'{measure.replace("_", " ").title()} Value')
        plt.ylabel('Frequency')
        plt.title(f'{measure.replace("_", " ").title()} Distribution')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "centrality_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top nodes comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Select 4 main measures for detailed analysis
    main_measures = ['degree', 'betweenness', 'closeness', 'pagerank']
    available_measures = [m for m in main_measures if m in top_nodes]
    
    for i, measure in enumerate(available_measures):
        if i < len(axes):
            nodes, scores = zip(*top_nodes[measure][:8])  # Top 8 nodes
            axes[i].bar(range(len(nodes)), scores, alpha=0.7, color=plt.cm.Set3(i))
            axes[i].set_xticks(range(len(nodes)))
            axes[i].set_xticklabels([f'Node {n}' for n in nodes], rotation=45)
            axes[i].set_ylabel(f'{measure.title()} Score')
            axes[i].set_title(f'Top {measure.title()} Nodes')
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(available_measures), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "centrality_top_nodes.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Centrality comparison scatter matrix
    if len(centrality_dict) >= 2:
        # Create DataFrame for scatter matrix
        centrality_df = pd.DataFrame(centrality_dict)
        
        # Select top 4 measures for scatter plot
        if len(centrality_df.columns) > 4:
            # Use most important measures
            important_measures = ['degree', 'betweenness', 'closeness', 'pagerank']
            available_important = [m for m in important_measures if m in centrality_df.columns]
            plot_df = centrality_df[available_important]
        else:
            plot_df = centrality_df
        
        if len(plot_df.columns) >= 2:
            try:
                sns.pairplot(plot_df, diag_kind='hist', corner=True)
                plt.suptitle('Centrality Measures Correlation Matrix', y=1.02)
                plt.savefig(os.path.join(save_dir, "centrality_correlation.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except:
                print("⚠️  Could not create correlation matrix")
    
    print("✅ Saved centrality analysis plots")

def identify_key_players(centrality_dict, G, top_n=5):
    """Identify key players based on multiple centrality measures"""
    # Normalize centrality scores
    normalized_scores = {}
    
    for measure, values in centrality_dict.items():
        vals = np.array(list(values.values()))
        if np.std(vals) > 0:
            normalized = (vals - np.mean(vals)) / np.std(vals)
        else:
            normalized = np.zeros_like(vals)
        
        # Create dictionary of normalized scores
        nodes = list(values.keys())
        normalized_scores[measure] = dict(zip(nodes, normalized))
    
    # Calculate composite score (average of normalized scores)
    composite_scores = {}
    all_nodes = list(centrality_dict['degree'].keys())
    
    for node in all_nodes:
        scores = []
        for measure in normalized_scores:
            if node in normalized_scores[measure]:
                scores.append(normalized_scores[measure][node])
        
        if scores:
            composite_scores[node] = np.mean(scores)
    
    # Get top key players
    key_players = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return key_players, composite_scores

def export_centrality_results(centrality_dict, top_nodes, key_players, save_path):
    """Export centrality results to CSV"""
    # Create comprehensive DataFrame
    all_nodes = list(centrality_dict['degree'].keys())
    
    data = []
    for node in all_nodes:
        row = {'node': node}
        for measure, values in centrality_dict.items():
            if node in values:
                row[measure] = values[node]
            else:
                row[measure] = 0
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Export main results
    df.to_csv(os.path.join(save_path, "centrality_all_nodes.csv"), index=False)
    
    # Export top nodes summary
    top_nodes_data = []
    for measure, nodes_scores in top_nodes.items():
        for rank, (node, score) in enumerate(nodes_scores, 1):
            top_nodes_data.append({
                'measure': measure,
                'rank': rank,
                'node': node,
                'score': score
            })
    
    top_df = pd.DataFrame(top_nodes_data)
    top_df.to_csv(os.path.join(save_path, "centrality_top_nodes.csv"), index=False)
    
    # Export key players
    key_players_df = pd.DataFrame(key_players, columns=['node', 'composite_score'])
    key_players_df.to_csv(os.path.join(save_path, "centrality_key_players.csv"), index=False)
    
    print("✅ Centrality results exported to CSV")

def main(config=None):
    """Main function for centrality analysis"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("CENTRALITY ANALYSIS")
    print("="*60)
    
    G = config.get_network()
    
    if G is None:
        print("❌ No network data found. Run data loading first.")
        return None
    
    print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Calculate centrality measures
    centrality_dict = calculate_centrality_measures(G)
    
    # Get top nodes
    top_nodes = get_top_central_nodes(centrality_dict, G, top_n=8)
    
    # Identify key players
    key_players, composite_scores = identify_key_players(centrality_dict, G, top_n=10)
    
    # Display results
    print("\n🏆 TOP CENTRAL NODES:")
    print("-" * 50)
    
    for measure, nodes_scores in top_nodes.items():
        if measure in ['degree', 'betweenness', 'closeness', 'pagerank']:
            print(f"\n{measure.upper()} CENTRALITY:")
            for i, (node, score) in enumerate(nodes_scores[:5], 1):
                print(f"  {i}. Node {node}: {score:.4f}")
    
    print("\n🎯 KEY PLAYERS (Composite Score):")
    print("-" * 40)
    for i, (node, score) in enumerate(key_players, 1):
        print(f"  {i}. Node {node}: {score:.3f}")
    
    # Analyze distributions
    from scipy import stats
    distributions = analyze_centrality_distributions(centrality_dict)
    
    print("\n📈 CENTRALITY DISTRIBUTIONS:")
    print("-" * 40)
    for measure, stats_dict in distributions.items():
        if measure in ['degree', 'betweenness', 'closeness', 'pagerank']:
            print(f"\n{measure.upper()}:")
            print(f"  Mean: {stats_dict['mean']:.4f}")
            print(f"  Std:  {stats_dict['std']:.4f}")
            print(f"  Max:  {stats_dict['max']:.4f}")
    
    # Create visualizations
    plots_dir = os.path.join(config.RESULTS_DIR, "charts")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_centrality_analysis(centrality_dict, top_nodes, G, plots_dir)
    
    # Export results
    data_dir = os.path.join(config.RESULTS_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    export_centrality_results(centrality_dict, top_nodes, key_players, data_dir)
    
    # Save to config
    config.set_centrality(centrality_dict)
    
    # Network insights
    print("\n💡 NETWORK INSIGHTS FROM CENTRALITY:")
    
    # Degree vs Betweenness analysis
    degree_top = [node for node, _ in top_nodes['degree'][:5]]
    betweenness_top = [node for node, _ in top_nodes['betweenness'][:5]]
    
    common_nodes = set(degree_top) & set(betweenness_top)
    if len(common_nodes) >= 3:
        print("  • Strong hubs: Same nodes are important in both degree and betweenness")
    else:
        print("  • Specialized roles: Different nodes important for connectivity vs information flow")
    
    # PageRank analysis
    pagerank_scores = list(centrality_dict['pagerank'].values())
    if np.max(pagerank_scores) > 0.01:
        print("  • Influential nodes: Some nodes have very high PageRank (influence)")
    
    # Closeness analysis
    closeness_scores = list(centrality_dict['closeness'].values())
    if np.max(closeness_scores) > 0.5:
        print("  • Efficient communicators: Some nodes can reach others quickly")
    
    results = {
        'centrality_dict': centrality_dict,
        'top_nodes': top_nodes,
        'key_players': key_players,
        'distributions': distributions
    }
    
    return results

if __name__ == "__main__":
    config = Config()
    
    # Load sample network if not exists
    if config.get_network() is None:
        from data.load_data import load_sample_network
        G = load_sample_network()
        config.set_network(G)
    
    main(config)