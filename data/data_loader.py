# data/data_loader.py
import networkx as nx
import pandas as pd
import numpy as np
import os

def load_network_data():
    """
    Load Higgs Twitter dataset hoặc tạo dataset mẫu
    Returns: NetworkX directed graph
    """
    print("📥 ĐANG LOAD DỮ LIỆU MẠNG...")
    
    # Ưu tiên load dataset thật nếu có
    if os.path.exists('higgs-retweet_network.edgelist.gz'):
        print("• Phát hiện dataset thật, đang load...")
        try:
            G = nx.read_edgelist(
                "higgs-retweet_network.edgelist.gz", 
                create_using=nx.DiGraph(),
                nodetype=int,
                data=False
            )
            print(f"✅ Load dataset thật thành công: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Lấy mẫu 200 nodes để phân tích nhanh
            if G.number_of_nodes() > 200:
                print("• Lấy mẫu 200 nodes để phân tích...")
                nodes_sample = list(G.nodes())[:200]
                G = G.subgraph(nodes_sample)
                
            return G
            
        except Exception as e:
            print(f"❌ Lỗi load dataset thật: {e}")
            print("• Chuyển sang dataset mẫu...")
    
    # Nếu không có dataset thật, tạo dataset mẫu
    print("• Tạo dataset mẫu với cấu trúc mạng xã hội thực tế...")
    return create_sample_network()

def create_sample_network():
    """
    Tạo dataset mẫu mô phỏng mạng Twitter Higgs
    với cấu trúc power-law và communities rõ ràng
    """
    G = nx.DiGraph()
    np.random.seed(42)  # Để kết quả có thể tái lập
    
    # Tạo 200 nodes
    nodes = range(1, 201)
    G.add_nodes_from(nodes)
    
    print("• Đang tạo edges với phân phối power-law...")
    
    # Tạo 3 communities rõ ràng
    community_assignments = {}
    community_sizes = [80, 70, 50]  # 3 communities
    
    start = 0
    for comm_id, size in enumerate(community_sizes):
        for i in range(start, start + size):
            community_assignments[i+1] = comm_id
        start += size
    
    # Thêm edges với cấu trúc community
    edges_count = 0
    
    for i in nodes:
        comm_i = community_assignments[i]
        
        # Số edges cho node i theo phân phối power-law
        base_edges = np.random.zipf(1.8)
        num_edges = min(base_edges, 50)  # Giới hạn max edges
        
        for _ in range(num_edges):
            # Chọn target với xác suất phụ thuộc vào community
            if np.random.random() < 0.7:  # 70% trong cùng community
                # Chọn node trong cùng community
                same_comm_nodes = [n for n in nodes if community_assignments[n] == comm_i and n != i]
                if same_comm_nodes:
                    target = np.random.choice(same_comm_nodes)
                    G.add_edge(i, target)
                    edges_count += 1
            else:  # 30% khác community
                other_comm_nodes = [n for n in nodes if community_assignments[n] != comm_i]
                if other_comm_nodes:
                    target = np.random.choice(other_comm_nodes)
                    G.add_edge(i, target)
                    edges_count += 1
    
    print(f"✅ Đã tạo dataset mẫu: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"• Số communities: {len(set(community_assignments.values()))}")
    
    # Tính một số thống kê cơ bản
    degrees = [d for n, d in G.degree()]
    print(f"• Degree trung bình: {np.mean(degrees):.2f}")
    print(f"• Degree lớn nhất: {max(degrees)}")
    
    return G

def load_real_dataset_info():
    """
    Hiển thị thông tin về dataset thật (nếu có)
    """
    dataset_info = {
        'name': 'Higgs Twitter Dataset',
        'source': 'Stanford SNAP',
        'nodes': 456626,  # ĐÃ SỬA - XÓA DẤU PHẨY
        'edges': 14855842,  # ĐÃ SỬA - XÓA DẤU PHẨY
        'description': 'Retweet network about Higgs boson discovery',
        'url': 'https://snap.stanford.edu/data/higgs-twitter.html'
    }
    
    print("\n📋 THÔNG TIN DATASET THẬT:")
    for key, value in dataset_info.items():
        print(f"   • {key}: {value}")
    
    return dataset_info

if __name__ == "__main__":
    # Test load data
    G = load_network_data()
    print(f"\n🎯 KẾT QUẢ TEST:")
    print(f"   - Số nodes: {G.number_of_nodes()}")
    print(f"   - Số edges: {G.number_of_edges()}")
    print(f"   - Có hướng: {G.is_directed()}")