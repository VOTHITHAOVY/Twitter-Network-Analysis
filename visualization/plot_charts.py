import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_network_summary(G, metrics, save_path):
    """Vẽ biểu đồ tổng quan mạng"""
    print("Creating network summary plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Degree distribution
    if G.is_directed():
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        
        axes[0,0].hist(in_degrees, bins=50, alpha=0.7, label='In-degree', color='blue')
        axes[0,0].hist(out_degrees, bins=50, alpha=0.7, label='Out-degree', color='red')
        axes[0,0].set_xlabel('Degree')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('In/Out Degree Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    else:
        degrees = [d for _, d in G.degree()]
        axes[0,0].hist(degrees, bins=50, alpha=0.7, color='green')
        axes[0,0].set_xlabel('Degree')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Degree Distribution')
        axes[0,0].grid(True, alpha=0.3)
    
    # 2. Centrality comparison (nếu có)
    if 'centrality' in metrics:
        centrality_data = pd.DataFrame(metrics['centrality'])
        if len(centrality_data.columns) > 0:
            # Lấy 4 centrality measures đầu tiên
            measures = centrality_data.columns[:4]
            centrality_data[measures].boxplot(ax=axes[0,1])
            axes[0,1].set_title('Centrality Measures Distribution')
            axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Community sizes (nếu có)
    if 'communities' in metrics:
        community_sizes = metrics['communities'].get('community_sizes', {})
        if community_sizes:
            sizes = list(community_sizes.values())
            axes[1,0].hist(sizes, bins=20, alpha=0.7, color='purple')
            axes[1,0].set_xlabel('Community Size')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Community Size Distribution')
            axes[1,0].grid(True, alpha=0.3)
    
    # 4. Network metrics summary
    axes[1,1].axis('off')
    summary_text = "NETWORK SUMMARY\n\n"
    basic_metrics = ['nodes', 'edges', 'density', 'avg_clustering']
    for metric in basic_metrics:
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, float):
                summary_text += f"{metric}: {value:.4f}\n"
            else:
                summary_text += f"{metric}: {value}\n"
    
    axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=12, fontfamily='monospace', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Summary plot saved: {save_path}")

def plot_correlation_analysis(centrality_dict, save_path):
    """Vẽ biểu đồ phân tích tương quan"""
    if len(centrality_dict) < 2:
        print("⚠️ Not enough centrality measures for correlation analysis")
        return
    
    print("Creating correlation analysis plots...")
    
    centrality_df = pd.DataFrame(centrality_dict)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap correlation
    corr_matrix = centrality_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0], square=True, fmt='.2f')
    axes[0].set_title('Centrality Measures Correlation')
    
    # Scatter plot của 2 measures đầu tiên
    if len(centrality_df.columns) >= 2:
        col1, col2 = centrality_df.columns[:2]
        axes[1].scatter(centrality_df[col1], centrality_df[col2], alpha=0.5, s=10)
        axes[1].set_xlabel(col1)
        axes[1].set_ylabel(col2)
        axes[1].set_title(f'{col1} vs {col2}')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Correlation plot saved: {save_path}")

def main(config=None):
    """Main function for visualization"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("DATA VISUALIZATION")
    print("="*60)
    
    try:
        G = config.get_network()
        print(f"✅ Network loaded: {G.number_of_nodes()} nodes")
    except:
        print("❌ No network data available")
        return None
    
    # Thu thập metrics từ config
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G)
    }
    
    # Thêm centrality nếu có
    if hasattr(config, 'CENTRALITY') and config.CENTRALITY:
        metrics['centrality'] = config.CENTRALITY
    
    # Thêm communities nếu có
    if hasattr(config, 'COMMUNITIES') and config.COMMUNITIES:
        metrics['communities'] = config.COMMUNITIES.get('analysis', {})
    
    # Tạo các biểu đồ
    plots_dir = os.path.join(config.RESULTS_DIR, "charts")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Biểu đồ tổng quan
    plot_network_summary(G, metrics, os.path.join(plots_dir, "network_summary.png"))
    
    # Biểu đồ tương quan (nếu có centrality data)
    if 'centrality' in metrics:
        plot_correlation_analysis(metrics['centrality'], os.path.join(plots_dir, "correlation_analysis.png"))
    
    print("✅ All visualizations completed!")
    return True

if __name__ == "__main__":
    main()