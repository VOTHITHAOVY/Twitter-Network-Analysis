import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import pandas as pd
import numpy as np

def load_and_validate_network(file_path):
    """Tải và kiểm tra tính hợp lệ của mạng"""
    print(f"Loading network from: {file_path}")
    
    try:
        G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int, data=False)
        print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"❌ Error loading network: {e}")
        return None

def clean_network(G):
    """Làm sạch mạng - loại bỏ node cô lập và self-loops"""
    print("Cleaning network...")
    
    original_nodes = G.number_of_nodes()
    original_edges = G.number_of_edges()
    
    # Loại bỏ self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        G.remove_edges_from(self_loops)
        print(f"Removed {len(self_loops)} self-loops")
    
    # Loại bỏ node cô lập (degree = 0)
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    if isolated_nodes:
        G.remove_nodes_from(isolated_nodes)
        print(f"Removed {len(isolated_nodes)} isolated nodes")
    
    # Lấy thành phần liên thông lớn nhất
    if not nx.is_weakly_connected(G):
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Using largest connected component: {G.number_of_nodes()} nodes")
    
    print(f"✅ Network cleaned: {original_nodes}→{G.number_of_nodes()} nodes, {original_edges}→{G.number_of_edges()} edges")
    return G

def analyze_network_quality(G):
    """Phân tích chất lượng mạng"""
    print("Analyzing network quality...")
    
    quality_metrics = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_weakly_connected(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'self_loops': len(list(nx.selfloop_edges(G))),
        'isolated_nodes': sum(1 for node in G.nodes() if G.degree(node) == 0)
    }
    
    print("📊 Network Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"  {key}: {value}")
    
    return quality_metrics

def main():
    """Main function for data preprocessing"""
    config = Config()
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Tải mạng
    file_path = os.path.join(config.DATA_DIR, "higgs-retweet_network.edgelist")
    G = load_and_validate_network(file_path)
    
    if not G:
        print("❌ Cannot proceed without valid network")
        return None
    
    # Phân tích chất lượng ban đầu
    initial_quality = analyze_network_quality(G)
    
    # Làm sạch mạng
    G_cleaned = clean_network(G)
    
    # Phân tích chất lượng sau khi làm sạch
    final_quality = analyze_network_quality(G_cleaned)
    
    # Lưu mạng đã làm sạch
    cleaned_path = os.path.join(config.DATA_DIR, "network_cleaned.edgelist")
    nx.write_edgelist(G_cleaned, cleaned_path, data=False)
    print(f"✅ Cleaned network saved: {cleaned_path}")
    
    # So sánh kết quả
    print("\n📈 PREPROCESSING RESULTS:")
    print(f"  Nodes removed: {initial_quality['num_nodes'] - final_quality['num_nodes']}")
    print(f"  Edges removed: {initial_quality['num_edges'] - final_quality['num_edges']}")
    print(f"  Density change: {initial_quality['density']:.6f} → {final_quality['density']:.6f}")
    
    return G_cleaned

if __name__ == "__main__":
    main()