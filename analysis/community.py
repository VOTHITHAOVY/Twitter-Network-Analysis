import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import community as community_louvain
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def detect_communities_louvain(G):
    """Detect communities using Louvain algorithm"""
    print("Detecting communities with Louvain algorithm...")
    
    try:
        # Chuyển sang undirected
        G_undirected = G.to_undirected()
        
        # Sử dụng networkx thay vì community_louvain
        communities = list(nx.algorithms.community.greedy_modularity_communities(G_undirected))
        
        # Chuyển thành partition dictionary
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        
        # Tính modularity
        modularity = nx.algorithms.community.quality.modularity(G_undirected, communities)
        
        print(f"✅ Louvain algorithm completed: {len(communities)} communities found")
        return partition, modularity
        
    except Exception as e:
        print(f"❌ Community detection failed: {e}")
        # Fallback: mỗi node là 1 community
        partition = {node: node for node in G.nodes()}
        modularity = 0.0
        return partition, modularity
    """Detect communities using Louvain algorithm"""
    print("Detecting communities with Louvain algorithm...")
    
    try:
        # Chuyển sang undirected cho Louvain (vì thuật toán chỉ hỗ trợ undirected)
        G_undirected = G.to_undirected()
        
        # Phát hiện communities sử dụng Louvain algorithm
        partition = community_louvain.best_partition(G_undirected)
        
        # Tính modularity trên đồ thị undirected
        modularity = community_louvain.modularity(partition, G_undirected)
        
        print(f"✅ Louvain algorithm completed: {len(set(partition.values()))} communities found")
        return partition, modularity
        
    except Exception as e:
        print(f"❌ Louvain algorithm failed: {e}")
        # Fallback: gán mỗi node vào community riêng
        partition = {node: node for node in G.nodes()}
        modularity = 0.0
        return partition, modularity
def detect_communities_girvan_newman(G, num_communities=None):
    """Detect communities using Girvan-Newman algorithm"""
    print("Detecting communities with Girvan-Newman algorithm...")
    
    if num_communities is None:
        num_communities = max(2, G.number_of_nodes() // 100)
    
    # Girvan-Newman trả về communities theo hierarchy
    comp = nx.algorithms.community.girvan_newman(G)
    
    # Lấy partition đầu tiên đạt số communities mong muốn
    for communities in comp:
        if len(communities) >= num_communities:
            # Chuyển thành dictionary partition
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            break
    else:
        # Fallback nếu không đạt được số communities mong muốn
        communities = next(comp)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
    
    # Tính modularity
    modularity = nx.algorithms.community.quality.modularity(G, communities)
    
    return partition, modularity, communities

def detect_communities_label_propagation(G):
    """Detect communities using Label Propagation"""
    print("Detecting communities with Label Propagation...")
    
    communities = list(nx.algorithms.community.label_propagation_communities(G))
    
    # Chuyển thành dictionary partition
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    
    # Tính modularity
    modularity = nx.algorithms.community.quality.modularity(G, communities)
    
    return partition, modularity, communities

def analyze_community_structure(partition, G):
    """Analyze community structure"""
    # Thống kê kích thước communities
    community_sizes = Counter(partition.values())
    
    # Tính modularity bằng networkx
    G_undirected = G.to_undirected()
    communities_dict = {}
    for node, comm_id in partition.items():
        if comm_id not in communities_dict:
            communities_dict[comm_id] = set()
        communities_dict[comm_id].add(node)
    communities = list(communities_dict.values())
    
    analysis = {
        'num_communities': len(community_sizes),
        'community_sizes': dict(community_sizes),
        'largest_community': max(community_sizes.values()),
        'smallest_community': min(community_sizes.values()),
        'avg_community_size': np.mean(list(community_sizes.values())),
        'modularity': nx.algorithms.community.quality.modularity(G_undirected, communities)
    }
    
    return analysis
    """Analyze community structure"""
    # Thống kê kích thước communities
    community_sizes = Counter(partition.values())
    
    # CHUYỂN SANG UNDIRECTED ĐỂ TÍNH MODULARITY
    G_undirected = G.to_undirected()
    
    analysis = {
        'num_communities': len(community_sizes),
        'community_sizes': dict(community_sizes),
        'largest_community': max(community_sizes.values()),
        'smallest_community': min(community_sizes.values()),
        'avg_community_size': np.mean(list(community_sizes.values())),
        'modularity': community_louvain.modularity(partition, G_undirected)  # SỬA THÀNH G_undirected
    }
    
    return analysis
def plot_community_analysis(partition, G, algorithm_name, save_path=None):
    """Plot community analysis results"""
    analysis = analyze_community_structure(partition, G)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Community size distribution
    sizes = list(analysis['community_sizes'].values())
    ax1.hist(sizes, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.set_xlabel('Community Size')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Community Size Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Community size bar chart
    communities = list(analysis['community_sizes'].keys())
    sizes = list(analysis['community_sizes'].values())
    ax2.bar(range(len(communities)), sizes, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Community ID')
    ax2.set_ylabel('Size')
    ax2.set_title('Community Sizes')
    ax2.grid(True, alpha=0.3)
    
    # 3. Pie chart of largest communities (top 10)
    if len(sizes) > 10:
        top_sizes = sorted(sizes, reverse=True)[:10]
        labels = [f'Comm {i}' for i in range(len(top_sizes))]
        ax3.pie(top_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Top 10 Largest Communities')
    else:
        labels = [f'Comm {i}' for i in range(len(sizes))]
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Community Distribution')
    
    # 4. Text summary
    ax4.axis('off')
    summary_text = f"""
    Community Analysis Summary
    ({algorithm_name})
    
    Total Communities: {analysis['num_communities']}
    Modularity: {analysis['modularity']:.4f}
    Largest Community: {analysis['largest_community']} nodes
    Smallest Community: {analysis['smallest_community']} nodes
    Average Size: {analysis['avg_community_size']:.1f} nodes
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Community Structure - {algorithm_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved community analysis: {save_path}")
    plt.close()

def main(config=None):
    """Main function for community detection"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("COMMUNITY DETECTION")
    print("="*60)
    
    G = config.get_network()
    
    if G is None:
        print("❌ No network data found. Run data loading first.")
        return None
    
    print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    results = {}
    
    # 1. Louvain algorithm
    print("\n🔍 Running Louvain algorithm...")
    louvain_partition, louvain_modularity = detect_communities_louvain(G)
    louvain_analysis = analyze_community_structure(louvain_partition, G)
    results['louvain'] = {
        'partition': louvain_partition,
        'modularity': louvain_modularity,
        'analysis': louvain_analysis
    }
    
    # 2. Label Propagation (nhanh hơn cho large networks)
    print("🔍 Running Label Propagation...")
    try:
        lp_partition, lp_modularity, lp_communities = detect_communities_label_propagation(G)
        lp_analysis = analyze_community_structure(lp_partition, G)
        results['label_propagation'] = {
            'partition': lp_partition,
            'modularity': lp_modularity,
            'communities': lp_communities,
            'analysis': lp_analysis
        }
    except Exception as e:
        print(f"⚠️ Label Propagation failed: {e}")
    
    # 3. Girvan-Newman (chỉ cho small networks)
    if G.number_of_nodes() < 1000:
        print("🔍 Running Girvan-Newman...")
        try:
            gn_partition, gn_modularity, gn_communities = detect_communities_girvan_newman(G)
            gn_analysis = analyze_community_structure(gn_partition, G)
            results['girvan_newman'] = {
                'partition': gn_partition,
                'modularity': gn_modularity,
                'communities': gn_communities,
                'analysis': gn_analysis
            }
        except Exception as e:
            print(f"⚠️ Girvan-Newman failed: {e}")
    
    # Hiển thị kết quả
    print("\n📊 COMMUNITY DETECTION RESULTS:")
    for algo, result in results.items():
        analysis = result['analysis']
        print(f"\n{algo.upper():<20}:")
        print(f"  Communities: {analysis['num_communities']}")
        print(f"  Modularity:  {analysis['modularity']:.4f}")
        print(f"  Size Range:  {analysis['smallest_community']} - {analysis['largest_community']} nodes")
        print(f"  Avg Size:    {analysis['avg_community_size']:.1f} nodes")
    
    # Tạo visualizations
    for algo, result in results.items():
        plot_community_analysis(
            result['partition'], G, algo,
            save_path=config.RESULTS_DIR + f"/charts/community_{algo}.png"
        )
    
    # Lưu kết quả tốt nhất
    best_algo = max(results.keys(), key=lambda x: results[x]['modularity'])
    config.set_communities(results[best_algo])
    
    print(f"\n🏆 Best algorithm: {best_algo} (modularity: {results[best_algo]['modularity']:.4f})")
    
    return results

if __name__ == "__main__":
    config = Config()
    
    # Load sample network nếu chưa có
    if config.get_network() is None:
        from data.load_data import load_sample_network
        G = load_sample_network()
        config.set_network(G)
    
    main(config)