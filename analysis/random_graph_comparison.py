# analysis/random_graph_comparison.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def random_graph_comparison(G):
    """So sánh mạng thực với random graphs - YÊU CẦU CHƯƠNG 3 - ĐÃ FIX HIỂN THỊ"""
    print("\n🔁 SO SÁNH VỚI RANDOM GRAPH MODELS")
    
    # Tạo đồ thị vô hướng để so sánh (vì ER graph thường vô hướng)
    G_undirected = G.to_undirected()
    
    # 1. Tạo Erdős–Rényi random graph
    n_nodes = G_undirected.number_of_nodes()
    n_edges = G_undirected.number_of_edges()
    p = (2 * n_edges) / (n_nodes * (n_nodes - 1))  # Xác suất kết nối
    
    print(f"• Tạo ER graph với {n_nodes} nodes, p={p:.4f}")
    er_graph = nx.erdos_renyi_graph(n_nodes, p, seed=42)
    
    # 2. Tính các metrics so sánh
    print("• Tính clustering coefficient...")
    real_clustering = nx.average_clustering(G_undirected)
    random_clustering = nx.average_clustering(er_graph)
    
    print("• Tính đường kính và bán kính...")
    # Lấy giant component để tính đường kính
    giant_real = max(nx.connected_components(G_undirected), key=len)
    G_giant_real = G_undirected.subgraph(giant_real)
    
    giant_random = max(nx.connected_components(er_graph), key=len) 
    G_giant_random = er_graph.subgraph(giant_random)
    
    real_diameter = nx.diameter(G_giant_real)
    random_diameter = nx.diameter(G_giant_random)
    
    real_radius = nx.radius(G_giant_real)
    random_radius = nx.radius(G_giant_random)
    
    # 3. Tính average shortest path length
    print("• Tính average path length...")
    real_avg_path = nx.average_shortest_path_length(G_giant_real)
    random_avg_path = nx.average_shortest_path_length(G_giant_random)
    
    # 4. In kết quả so sánh
    print(f"\n📊 KẾT QUẢ SO SÁNH:")
    print(f"• Clustering Coefficient:")
    print(f"  - Mạng thực: {real_clustering:.4f}")
    print(f"  - Random ER: {random_clustering:.4f}")
    print(f"  - Tỷ lệ: {real_clustering/random_clustering:.1f}x")
    
    print(f"• Đường kính (Giant Component):")
    print(f"  - Mạng thực: {real_diameter}")
    print(f"  - Random ER: {random_diameter}")
    
    print(f"• Bán kính (Giant Component):")
    print(f"  - Mạng thực: {real_radius}") 
    print(f"  - Random ER: {random_radius}")
    
    print(f"• Average Path Length:")
    print(f"  - Mạng thực: {real_avg_path:.2f}")
    print(f"  - Random ER: {random_avg_path:.2f}")
    
    # 5. Tạo bảng so sánh
    comparison_data = {
        'Metric': ['Clustering', 'Diameter', 'Radius', 'Avg Path Length'],
        'Real Network': [real_clustering, real_diameter, real_radius, real_avg_path],
        'Random ER': [random_clustering, random_diameter, random_radius, random_avg_path],
        'Ratio': [
            real_clustering/random_clustering,
            real_diameter/random_diameter, 
            real_radius/random_radius,
            real_avg_path/random_avg_path
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(f"\n📋 BẢNG SO SÁNH CHI TIẾT:")
    print(df_comparison.round(3))
    
    # 6. Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 5))
    
    # Biểu đồ 1: Degree distribution
    plt.subplot(1, 3, 1)
    real_degrees = [d for n, d in G_undirected.degree()]
    random_degrees = [d for n, d in er_graph.degree()]
    
    plt.hist(real_degrees, bins=30, alpha=0.7, label='Mạng thực', color='blue', density=True)
    plt.hist(random_degrees, bins=30, alpha=0.7, label='Random ER', color='red', density=True)
    plt.xlabel('Degree')
    plt.ylabel('Density')
    plt.title('PHÂN PHỐI BẬC')
    plt.legend()
    plt.yscale('log')
    
    # Biểu đồ 2: So sánh metrics
    plt.subplot(1, 3, 2)
    metrics = ['Clustering', 'Avg Path\nLength']
    real_values = [real_clustering, real_avg_path]
    random_values = [random_clustering, random_avg_path]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, real_values, width, label='Mạng thực', alpha=0.7, color='blue')
    plt.bar(x + width/2, random_values, width, label='Random ER', alpha=0.7, color='red')
    plt.xlabel('Metrics')
    plt.ylabel('Giá trị')
    plt.title('SO SÁNH METRICS')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Biểu đồ 3: Tỷ lệ so sánh
    plt.subplot(1, 3, 3)
    ratios = [real_clustering/random_clustering, real_avg_path/random_avg_path]
    ratio_labels = ['Clustering\nRatio', 'Path Length\nRatio']
    
    colors = ['green' if ratio > 1 else 'orange' for ratio in ratios]
    plt.bar(ratio_labels, ratios, color=colors, alpha=0.7)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    plt.ylabel('Tỷ lệ (Thực/Random)')
    plt.title('TỶ LỆ SO SÁNH')
    
    # Thêm giá trị lên biểu đồ
    for i, ratio in enumerate(ratios):
        plt.text(i, ratio + 0.1, f'{ratio:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('random_graph_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # FIX: Đóng figure
    print("💾 Đã lưu: random_graph_comparison.png")
    
    return df_comparison