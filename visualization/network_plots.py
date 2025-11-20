# visualization/network_plots.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from community import community_louvain
from collections import Counter
import pandas as pd

def create_all_visualizations(G):
    """
    Tạo tất cả các biểu đồ visualization cho báo cáo - ĐÃ FIX HIỂN THỊ
    """
    print("\n🎨 BẮT ĐẦU TẠO VISUALIZATIONS...")
    
    # 1. Degree distribution
    create_degree_distribution_plot(G)
    
    # 2. Centrality comparison
    create_centrality_comparison_plot(G)
    
    # 3. Community visualization
    create_community_visualization(G)
    
    # 4. K-core visualization
    create_kcore_visualization(G)
    
    # 5. Network layout đơn giản
    create_network_layout(G)
    
    print("✅ ĐÃ TẠO TẤT CẢ VISUALIZATIONS")

def create_degree_distribution_plot(G):
    """Tạo biểu đồ phân phối bậc và power-law fitting - ĐÃ FIX"""
    print("• Đang tạo biểu đồ phân phối bậc...")
    
    # Tính degrees
    degrees = [d for n, d in G.degree()]
    degree_counts = Counter(degrees)
    
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ 1: Histogram thông thường
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Số nodes')
    plt.title('PHÂN PHỐI BẬC - HISTOGRAM')
    plt.grid(True, alpha=0.3)
    
    # Thêm thống kê
    plt.text(0.7, 0.9, f'Degree TB: {np.mean(degrees):.1f}\n'
                       f'Degree max: {max(degrees)}\n'
                       f'Degree min: {min(degrees)}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # Biểu đồ 2: Log-log plot cho power-law
    plt.subplot(1, 2, 2)
    degree_vals = list(degree_counts.keys())
    frequencies = list(degree_counts.values())
    
    # Lọc bỏ giá trị 0
    valid_indices = [i for i, freq in enumerate(frequencies) if freq > 0 and degree_vals[i] > 0]
    degree_vals = [degree_vals[i] for i in valid_indices]
    frequencies = [frequencies[i] for i in valid_indices]
    
    plt.loglog(degree_vals, frequencies, 'bo', alpha=0.6, markersize=4)
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Tần suất (log scale)')
    plt.title('PHÂN PHỐI BẬC - LOG-LOG PLOT\n(Power Law Distribution)')
    plt.grid(True, alpha=0.3)
    
    # Thêm đường fit power-law (đơn giản)
    if len(degree_vals) > 1:
        try:
            # Linear regression trên log scale
            log_deg = np.log(degree_vals)
            log_freq = np.log(frequencies)
            slope, intercept = np.polyfit(log_deg, log_freq, 1)
            
            # Vẽ đường fit
            x_fit = np.linspace(min(degree_vals), max(degree_vals), 100)
            y_fit = np.exp(intercept) * x_fit**slope
            plt.loglog(x_fit, y_fit, 'r-', alpha=0.8, linewidth=2, 
                      label=f'Power-law fit: α = {-slope:.2f}')
            plt.legend()
            
            # Hiển thị hệ số power-law
            plt.text(0.05, 0.15, f'Power-law exponent: α = {-slope:.2f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.7))
        except:
            print("   - Không thể fit power-law")
    
    plt.tight_layout()
    plt.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()  # FIX: Đóng figure
    print("   💾 Đã lưu: degree_distribution.png")

def create_centrality_comparison_plot(G):
    """Tạo biểu đồ so sánh các centrality measures - ĐÃ FIX"""
    print("• Đang tạo biểu đồ so sánh centrality...")
    
    # Tính centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=100)
    closeness_centrality = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    
    # Lấy top 10 nodes cho mỗi measure
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    
    plt.figure(figsize=(15, 10))
    
    # Biểu đồ 1: So sánh top nodes across measures
    plt.subplot(2, 2, 1)
    top_nodes = list(set([node for node, _ in top_degree + top_betweenness + top_closeness + top_pagerank]))[:8]
    
    x_pos = np.arange(len(top_nodes))
    width = 0.2
    
    degree_scores = [degree_centrality.get(node, 0) for node in top_nodes]
    betweenness_scores = [betweenness_centrality.get(node, 0) for node in top_nodes]
    closeness_scores = [closeness_centrality.get(node, 0) for node in top_nodes]
    pagerank_scores = [pagerank.get(node, 0) for node in top_nodes]
    
    plt.bar(x_pos - 1.5*width, degree_scores, width, label='Degree', alpha=0.7, color='blue')
    plt.bar(x_pos - 0.5*width, betweenness_scores, width, label='Betweenness', alpha=0.7, color='green')
    plt.bar(x_pos + 0.5*width, closeness_scores, width, label='Closeness', alpha=0.7, color='orange')
    plt.bar(x_pos + 1.5*width, pagerank_scores, width, label='PageRank', alpha=0.7, color='red')
    
    plt.xlabel('Nodes')
    plt.ylabel('Centrality Score')
    plt.title('SO SÁNH CENTRALITY CỦA TOP NODES')
    plt.xticks(x_pos, [f'Node {n}' for n in top_nodes], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 2: Scatter plot Degree vs PageRank
    plt.subplot(2, 2, 2)
    nodes_sample = list(G.nodes())[:100]  # Lấy mẫu 100 nodes
    degree_sample = [degree_centrality[node] for node in nodes_sample]
    pagerank_sample = [pagerank[node] for node in nodes_sample]
    
    plt.scatter(degree_sample, pagerank_sample, alpha=0.6, color='purple')
    plt.xlabel('Degree Centrality')
    plt.ylabel('PageRank')
    plt.title('TƯƠNG QUAN: DEGREE vs PAGERANK')
    plt.grid(True, alpha=0.3)
    
    # Tính và hiển thị correlation
    correlation = np.corrcoef(degree_sample, pagerank_sample)[0,1]
    plt.text(0.05, 0.9, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.7))
    
    # Biểu đồ 3: Phân bố centrality values
    plt.subplot(2, 2, 3)
    centrality_data = [
        list(degree_centrality.values()),
        list(betweenness_centrality.values()),
        list(closeness_centrality.values()), 
        list(pagerank.values())
    ]
    labels = ['Degree', 'Betweenness', 'Closeness', 'PageRank']
    colors = ['blue', 'green', 'orange', 'red']
    
    box_plot = plt.boxplot(centrality_data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Centrality Value')
    plt.title('PHÂN BỐ GIÁ TRỊ CENTRALITY')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 4: Top influencers across measures
    plt.subplot(2, 2, 4)
    all_top_nodes = [node for node, _ in top_degree[:5]] + [node for node, _ in top_betweenness[:5]] + \
                   [node for node, _ in top_closeness[:5]] + [node for node, _ in top_pagerank[:5]]
    
    node_counts = Counter(all_top_nodes)
    super_influencers = [(node, count) for node, count in node_counts.items() if count >= 2]
    
    if super_influencers:
        nodes = [f'Node {node}' for node, count in super_influencers]
        counts = [count for node, count in super_influencers]
        
        plt.bar(nodes, counts, color='purple', alpha=0.7)
        plt.xlabel('Nodes')
        plt.ylabel('Số lần xuất hiện trong top lists')
        plt.title('SUPER INFLUENCERS\n(Xuất hiện trong nhiều top lists)')
        plt.xticks(rotation=45)
        
        # Thêm số lên biểu đồ
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'Không có super influencers', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('SUPER INFLUENCERS')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('centrality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # FIX: Đóng figure
    print("   💾 Đã lưu: centrality_comparison.png")

def create_community_visualization(G):
    """Tạo visualization cho community detection - ĐÃ FIX"""
    print("• Đang tạo biểu đồ communities...")
    
    # Chuyển sang đồ thị vô hướng cho community detection
    G_undirected = G.to_undirected()
    
    # Phát hiện communities bằng Louvain
    partition = community_louvain.best_partition(G_undirected)
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)
    
    modularity = community_louvain.modularity(partition, G_undirected)
    
    plt.figure(figsize=(15, 5))
    
    # Biểu đồ 1: Kích thước communities
    plt.subplot(1, 3, 1)
    comm_sizes = [len(nodes) for nodes in communities.values()]
    comm_ids = list(communities.keys())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(comm_sizes)))
    bars = plt.bar(comm_ids, comm_sizes, color=colors, alpha=0.7)
    plt.xlabel('Community ID')
    plt.ylabel('Số nodes')
    plt.title(f'KÍCH THƯỚC COMMUNITIES\n(Modularity: {modularity:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Thêm số lên biểu đồ
    for i, size in enumerate(comm_sizes):
        plt.text(comm_ids[i], size + 0.5, str(size), ha='center', va='bottom')
    
    # Biểu đồ 2: Phân bố degree trong communities
    plt.subplot(1, 3, 2)
    comm_degree_data = []
    comm_labels = []
    
    for comm_id, nodes in communities.items():
        degrees = [G_undirected.degree(node) for node in nodes]
        comm_degree_data.append(degrees)
        comm_labels.append(f'Comm {comm_id}')
    
    box_plot = plt.boxplot(comm_degree_data, labels=comm_labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(comm_degree_data)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Communities')
    plt.ylabel('Degree')
    plt.title('PHÂN BỐ DEGREE TRONG COMMUNITIES')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 3: Visualize network với communities
    plt.subplot(1, 3, 3)
    try:
        # Layout cho visualization
        pos = nx.spring_layout(G_undirected, seed=42, k=3/np.sqrt(G_undirected.number_of_nodes()))
        
        # Màu sắc cho communities
        node_colors = [partition[node] for node in G_undirected.nodes()]
        
        # Vẽ network
        nx.draw_networkx_nodes(G_undirected, pos, 
                              node_color=node_colors,
                              node_size=50,
                              cmap=plt.cm.Set3,
                              alpha=0.8)
        nx.draw_networkx_edges(G_undirected, pos, 
                              alpha=0.2, 
                              edge_color='gray',
                              width=0.5)
        
        plt.title('NETWORK VỚI COMMUNITIES')
        plt.axis('off')
        
    except Exception as e:
        plt.text(0.5, 0.5, f'Không thể vẽ network:\n{e}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('NETWORK VISUALIZATION')
    
    plt.tight_layout()
    plt.savefig('community_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # FIX: Đóng figure
    print("   💾 Đã lưu: community_analysis.png")
    
    # In thông tin communities
    print(f"   📊 Thông tin communities:")
    print(f"      - Số communities: {len(communities)}")
    print(f"      - Modularity: {modularity:.3f}")
    for comm_id, nodes in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"      - Community {comm_id}: {len(nodes)} nodes")

def create_kcore_visualization(G):
    """Tạo visualization cho K-core decomposition - ĐÃ FIX"""
    print("• Đang tạo biểu đồ K-core...")
    
    G_undirected = G.to_undirected()
    
    # Tính K-core decomposition
    core_numbers = nx.core_number(G_undirected)
    
    # Thống kê số nodes cho mỗi k-core
    kcore_stats = {}
    for k in range(1, max(core_numbers.values()) + 1):
        k_core = nx.k_core(G_undirected, k)
        kcore_stats[k] = k_core.number_of_nodes()
    
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ 1: K-core sizes
    plt.subplot(1, 2, 1)
    k_values = list(kcore_stats.keys())
    core_sizes = list(kcore_stats.values())
    
    plt.bar(k_values, core_sizes, color='lightcoral', alpha=0.7, edgecolor='darkred')
    plt.xlabel('K value')
    plt.ylabel('Số nodes trong K-core')
    plt.title('K-CORE DECOMPOSITION\n(Kích thước các lõi)')
    plt.grid(True, alpha=0.3)
    
    # Thêm số lên biểu đồ
    for i, size in enumerate(core_sizes):
        plt.text(k_values[i], size + 0.5, str(size), ha='center', va='bottom')
    
    # Biểu đồ 2: Phân bố core numbers
    plt.subplot(1, 2, 2)
    core_values = list(core_numbers.values())
    plt.hist(core_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    plt.xlabel('Core Number')
    plt.ylabel('Số nodes')
    plt.title('PHÂN BỐ CORE NUMBERS')
    plt.grid(True, alpha=0.3)
    
    # Thêm thống kê
    plt.text(0.7, 0.9, f'Core number TB: {np.mean(core_values):.1f}\n'
                       f'Core number max: {max(core_values)}\n'
                       f'Số nodes trong {max(core_values)}-core: {kcore_stats[max(core_values)]}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('kcore_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # FIX: Đóng figure
    print("   💾 Đã lưu: kcore_analysis.png")
    print(f"   📊 K-core max: {max(core_numbers.values())}")
    print(f"   📊 Số nodes trong {max(core_numbers.values())}-core: {kcore_stats[max(core_numbers.values())]}")

def create_network_layout(G):
    """Tạo network layout đơn giản - ĐÃ FIX"""
    print("• Đang tạo network layout...")
    
    G_undirected = G.to_undirected()
    
    plt.figure(figsize=(10, 8))
    
    try:
        # Tính layout
        pos = nx.spring_layout(G_undirected, seed=42, k=3/np.sqrt(G_undirected.number_of_nodes()))
        
        # Tính degree để làm kích thước node
        degrees = [G_undirected.degree(node) for node in G_undirected.nodes()]
        node_sizes = [d * 10 + 10 for d in degrees]  # Scale kích thước
        
        # Vẽ network
        nx.draw_networkx_nodes(G_undirected, pos,
                              node_size=node_sizes,
                              node_color='lightblue',
                              alpha=0.7,
                              edgecolors='black',
                              linewidths=0.5)
        
        nx.draw_networkx_edges(G_undirected, pos,
                              alpha=0.3,
                              edge_color='gray',
                              width=0.5)
        
        # Vẽ labels cho các nodes có degree cao
        high_degree_nodes = [node for node, degree in G_undirected.degree() if degree > np.percentile(degrees, 80)]
        labels = {node: f'{node}' for node in high_degree_nodes}
        nx.draw_networkx_labels(G_undirected, pos, labels, font_size=8, font_weight='bold')
        
        plt.title('NETWORK LAYOUT\n(Kích thước node theo degree)')
        plt.axis('off')
        
    except Exception as e:
        plt.text(0.5, 0.5, f'Không thể vẽ network layout:\n{e}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('NETWORK LAYOUT')
    
    plt.tight_layout()
    plt.savefig('network_layout.png', dpi=300, bbox_inches='tight')
    plt.close()  # FIX: Đóng figure
    print("   💾 Đã lưu: network_layout.png")

if __name__ == "__main__":
    # Test visualization
    print("🧪 TEST VISUALIZATION...")
    
    # Tạo đồ thị mẫu
    G = nx.erdos_renyi_graph(100, 0.1, seed=42)
    
    # Test các hàm visualization
    create_degree_distribution_plot(G)
    create_centrality_comparison_plot(G)
    create_community_visualization(G)
    create_kcore_visualization(G)
    create_network_layout(G)