import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_community_algorithms(community_results):
    """Compare different community detection algorithms"""
    comparison_data = []
    
    for algo, results in community_results.items():
        analysis = results['analysis']
        comparison_data.append({
            'Algorithm': algo.replace('_', ' ').title(),
            'Number of Communities': analysis['num_communities'],
            'Modularity': analysis['modularity'],
            'Largest Community': analysis['largest_community'],
            'Smallest Community': analysis['smallest_community'],
            'Average Size': analysis['avg_community_size']
        })
    
    return pd.DataFrame(comparison_data)

def plot_algorithm_comparison(comparison_df, save_path=None):
    """Plot comparison of community detection algorithms"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Modularity comparison
    axes[0,0].bar(comparison_df['Algorithm'], comparison_df['Modularity'], 
                  color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Modularity Comparison')
    axes[0,0].set_ylabel('Modularity')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Number of communities
    axes[0,1].bar(comparison_df['Algorithm'], comparison_df['Number of Communities'],
                  color='lightcoral', alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Number of Communities')
    axes[0,1].set_ylabel('Count')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Community size range
    algorithms = comparison_df['Algorithm']
    largest = comparison_df['Largest Community']
    smallest = comparison_df['Smallest Community']
    
    for i, algo in enumerate(algorithms):
        axes[1,0].plot([i, i], [smallest[i], largest[i]], 'o-', color='blue', linewidth=3)
    axes[1,0].set_xticks(range(len(algorithms)))
    axes[1,0].set_xticklabels(algorithms, rotation=45)
    axes[1,0].set_title('Community Size Range')
    axes[1,0].set_ylabel('Number of Nodes')
    
    # 4. Average community size
    axes[1,1].bar(comparison_df['Algorithm'], comparison_df['Average Size'],
                  color='orange', alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Average Community Size')
    axes[1,1].set_ylabel('Nodes')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Community Detection Algorithms Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved algorithm comparison: {save_path}")
    plt.close()

def analyze_community_quality(community_results, G):
    """Analyze quality metrics for communities"""
    quality_metrics = {}
    
    for algo, results in community_results.items():
        partition = results['partition']
        
        # Tính internal density và external connectivity
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # Chuyển thành list of sets cho nx
        community_sets = [set(nodes) for nodes in communities.values()]
        
        # Tính các metrics
        coverage = nx.algorithms.community.quality.coverage(G, community_sets)
        performance = nx.algorithms.community.quality.performance(G, community_sets)
        
        quality_metrics[algo] = {
            'coverage': coverage,
            'performance': performance,
            'modularity': results['modularity']
        }
    
    return quality_metrics

def main(config=None):
    """Main function for community algorithm comparison"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("COMMUNITY ALGORITHM COMPARISON")
    print("="*60)
    
    # Lấy kết quả community detection
    community_results = config.COMMUNITIES
    
    if not community_results:
        print("❌ No community results found. Run community detection first.")
        
        # Thử chạy community detection
        from analysis.community import main as run_community
        community_results = run_community(config)
        
        if not community_results:
            return None
    
    print("✅ Community results loaded for comparison")
    
    # So sánh algorithms
    comparison_df = compare_community_algorithms(community_results)
    
    print("\n📊 ALGORITHM COMPARISON:")
    print(comparison_df.round(4))
    
    # Phân tích chất lượng
    G = config.get_network()
    quality_metrics = analyze_community_quality(community_results, G)
    
    print("\n🎯 QUALITY METRICS:")
    quality_df = pd.DataFrame(quality_metrics).T
    print(quality_df.round(4))
    
    # Tìm algorithm tốt nhất
    best_modularity = comparison_df.loc[comparison_df['Modularity'].idxmax()]
    print(f"\n🏆 BEST ALGORITHM:")
    print(f"  Algorithm: {best_modularity['Algorithm']}")
    print(f"  Modularity: {best_modularity['Modularity']:.4f}")
    print(f"  Communities: {best_modularity['Number of Communities']}")
    
    # Tạo visualizations
    plot_algorithm_comparison(
        comparison_df,
        save_path=config.RESULTS_DIR + "/charts/community_algorithm_comparison.png"
    )
    
    # Lưu kết quả so sánh
    results = {
        'comparison_df': comparison_df,
        'quality_metrics': quality_metrics,
        'best_algorithm': best_modularity['Algorithm']
    }
    
    return results

if __name__ == "__main__":
    config = Config()
    
    # Chạy community detection trước nếu chưa có
    if not config.COMMUNITIES:
        from analysis.community import main as run_community
        run_community(config)
    
    main(config)