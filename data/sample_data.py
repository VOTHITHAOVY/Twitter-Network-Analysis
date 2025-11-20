# data/sample_data.py
import networkx as nx
import numpy as np
import pandas as pd

def create_power_law_network(n_nodes=200, avg_degree=20, seed=42):
    """
    Tạo mạng với phân phối bậc power-law
    Mô phỏng cấu trúc mạng xã hội thực tế
    """
    print(f"🎯 TẠO MẠNG POWER-LAW: {n_nodes} nodes, degree TB: {avg_degree}")
    np.random.seed(seed)
    
    G = nx.DiGraph()
    nodes = range(1, n_nodes + 1)
    G.add_nodes_from(nodes)
    
    # Tạo danh sách degrees theo phân phối power-law
    degrees = []
    for i in range(n_nodes):
        # Phân phối zipf (power-law)
        degree = np.random.zipf(1.6)
        degree = min(degree, n_nodes//2)  # Giới hạn max degree
        degree = max(degree, 1)  # Đảm bảo ít nhất 1 connection
        degrees.append(degree)
    
    # Điều chỉnh để đạt degree trung bình mong muốn
    current_avg = np.mean(degrees)
    scaling_factor = avg_degree / current_avg
    degrees = [int(d * scaling_factor) for d in degrees]
    degrees = [max(d, 1) for d in degrees]  # Đảm bảo ít nhất 1 connection
    
    print(f"• Degree trung bình thực tế: {np.mean(degrees):.2f}")
    
    # Thêm edges dựa trên degrees
    edges_count = 0
    for i, source in enumerate(nodes):
        num_edges = degrees[i]
        
        # Tạo danh sách targets có trọng số (preferential attachment)
        targets = []
        weights = []
        
        for target in nodes:
            if target != source:
                # Ưu tiên kết nối đến nodes có degree cao (preferential attachment)
                weight = G.degree(target) + 1  # +1 để tránh chia 0
                targets.append(target)
                weights.append(weight)
        
        if targets and weights:
            # Chọn targets với xác suất tỷ lệ với weight
            weights = np.array(weights) / sum(weights)
            selected_targets = np.random.choice(
                targets, 
                size=min(num_edges, len(targets)), 
                replace=False, 
                p=weights
            )
            
            for target in selected_targets:
                G.add_edge(source, target)
                edges_count += 1
    
    print(f"✅ Đã tạo: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Kiểm tra phân phối power-law
    actual_degrees = [d for n, d in G.degree()]
    print(f"• Degree thực tế: TB={np.mean(actual_degrees):.2f}, Max={max(actual_degrees)}, Min={min(actual_degrees)}")
    
    return G

def create_network_with_communities(n_nodes=200, n_communities=4, intra_prob=0.3, inter_prob=0.02):
    """
    Tạo mạng với cấu trúc communities rõ ràng
    """
    print(f"🏘️ TẠO MẠNG VỚI {n_communities} COMMUNITIES...")
    
    G = nx.DiGraph()
    nodes = range(1, n_nodes + 1)
    G.add_nodes_from(nodes)
    
    # Phân chia nodes vào communities
    community_size = n_nodes // n_communities
    community_assignments = {}
    
    for i, node in enumerate(nodes):
        comm_id = i // community_size
        if comm_id >= n_communities:
            comm_id = n_communities - 1
        community_assignments[node] = comm_id
    
    # Thêm edges với xác suất phụ thuộc vào community
    edges_count = 0
    
    for i in nodes:
        comm_i = community_assignments[i]
        
        # Số edges cho node i (variation)
        num_edges = np.random.poisson(15) + 5  # 5-25 edges mỗi node
        
        for _ in range(num_edges):
            if np.random.random() < intra_prob:  # Edge trong community
                same_comm_nodes = [n for n in nodes if community_assignments[n] == comm_i and n != i]
                if same_comm_nodes:
                    target = np.random.choice(same_comm_nodes)
                    G.add_edge(i, target)
                    edges_count += 1
            else:  # Edge giữa các communities
                other_comm_nodes = [n for n in nodes if community_assignments[n] != comm_i]
                if other_comm_nodes and np.random.random() < inter_prob:
                    target = np.random.choice(other_comm_nodes)
                    G.add_edge(i, target)
                    edges_count += 1
    
    print(f"✅ Đã tạo: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"• Số communities: {len(set(community_assignments.values()))}")
    
    # Tính modularity để kiểm tra chất lượng communities
    try:
        from community import community_louvain
        partition = community_louvain.best_partition(G.to_undirected())
        modularity = community_louvain.modularity(partition, G.to_undirected())
        print(f"• Modularity: {modularity:.3f}")
    except:
        print("• Không thể tính modularity")
    
    return G, community_assignments

def export_network_stats(G, filename='network_stats.csv'):
    """
    Xuất thống kê mạng ra file CSV
    """
    print(f"💾 ĐANG XUẤT THỐNG KÊ: {filename}")
    
    stats_data = []
    
    # Tính các centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=100)
    closeness_centrality = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    
    for node in G.nodes():
        stats_data.append({
            'node_id': node,
            'degree': G.degree(node),
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node),
            'degree_centrality': degree_centrality[node],
            'betweenness_centrality': betweenness_centrality[node],
            'closeness_centrality': closeness_centrality[node],
            'pagerank': pagerank[node]
        })
    
    df = pd.DataFrame(stats_data)
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"✅ Đã xuất thống kê {len(df)} nodes")
    
    # Thống kê tổng quan
    print(f"\n📊 THỐNG KÊ TỔNG QUAN:")
    print(f"   - Số nodes: {G.number_of_nodes()}")
    print(f"   - Số edges: {G.number_of_edges()}")
    print(f"   - Đồ thị có hướng: {G.is_directed()}")
    print(f"   - Số thành phần liên thông: {nx.number_weakly_connected_components(G)}")
    
    return df

if __name__ == "__main__":
    # Test các hàm
    print("🧪 TEST TẠO DATASET MẪU...")
    
    # Tạo mạng power-law
    G1 = create_power_law_network(100, 15)
    
    print("\n" + "="*50)
    
    # Tạo mạng với communities
    G2, comm_assign = create_network_with_communities(150, 4)
    
    print("\n" + "="*50)
    
    # Xuất thống kê
    export_network_stats(G2, 'sample_network_stats.csv')