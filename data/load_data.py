import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import pandas as pd
import gzip
import os

def load_higgs_network(network_type='retweet'):
    """
    Load Higgs Twitter Network - FIXED VERSION
    """
    config = Config()
    
    # THỬ CẢ 2 ĐỊNH DẠNG: .edgelist VÀ .edgelist.gz
    filename1 = f"higgs-{network_type}_network.edgelist"      # Không nén
    filename2 = f"higgs-{network_type}_network.edgelist.gz"   # Nén
    
    file_path1 = os.path.join(config.DATA_DIR, filename1)
    file_path2 = os.path.join(config.DATA_DIR, filename2)
    
    # Kiểm tra file nào tồn tại
    if os.path.exists(file_path1):
        file_path = file_path1
        compressed = False
        print(f"✅ Found dataset: {filename1}")
    elif os.path.exists(file_path2):
        file_path = file_path2
        compressed = True
        print(f"✅ Found dataset: {filename2}")
    else:
        print(f"❌ Dataset not found: {filename1} or {filename2}")
        return load_sample_network()
    
    print(f"Loading {network_type} network from: {file_path}")
    
    try:
        # Đọc file
        if compressed:
            with gzip.open(file_path, 'rt') as f:
                G = nx.read_edgelist(f, create_using=nx.DiGraph(), nodetype=int, data=False)
        else:
            G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int, data=False)
        
        print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("🔄 Using sample network instead...")
        return load_sample_network()
    """
    Load Higgs Twitter Network - FIXED VERSION
    """
    config = Config()
    
    filename = f"higgs-{network_type}_network.edgelist.gz"
    file_path = os.path.join(config.DATA_DIR, filename)
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(file_path):
        print(f"❌ Dataset not found: {filename}")
        return load_sample_network()
    
    print(f"Loading {network_type} network from: {file_path}")
    
    try:
        # Đọc file với data=False để bỏ qua edge data
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                G = nx.read_edgelist(f, create_using=nx.DiGraph(), nodetype=int, data=False)
        else:
            G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int, data=False)
        
        print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("🔄 Using sample network instead...")
        return load_sample_network()

def load_sample_network():
    """Tạo network mẫu cho testing"""
    print("Generating sample social network...")
    
    # Tạo network với cấu trúc small-world
    G = nx.connected_watts_strogatz_graph(n=300, k=10, p=0.1, seed=42)
    G = nx.DiGraph(G)  # Chuyển thành có hướng
    
    # Thêm một số node có degree cao (influencers)
    for i in range(10):
        influencer = 300 + i
        G.add_node(influencer)
        # Mỗi influencer được theo dõi bởi nhiều node
        for j in range(30):
            follower = (influencer * j) % 300
            G.add_edge(follower, influencer)
    
    print(f"Sample network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def get_network_statistics(G):
    """Calculate basic network statistics"""
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_directed': G.is_directed(),
        'is_connected': nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)
    }
    return stats

def main(config=None):
    """Main function for data loading module"""
    if config is None:
        config = Config()
    
    print("\n" + "="*50)
    print("DATA LOADING MODULE")
    print("="*50)
    
    # Load network
    G = load_higgs_network('retweet')
    
    # Lưu network vào config
    config.set_network(G)
    
    # Tính và hiển thị thống kê
    stats = get_network_statistics(G)
    print("\n📊 NETWORK STATISTICS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return G

if __name__ == "__main__":
    main()