import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_network_basic(G, save_path, max_nodes=100):
    """Vẽ mạng cơ bản (cho mạng nhỏ)"""
    print(f"Drawing basic network (max {max_nodes} nodes)...")
    
    # Lấy subgraph nếu mạng quá lớn
    if G.number_of_nodes() > max_nodes:
        nodes_sample = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes_sample)
        print(f"Using sample of {len(nodes_sample)} nodes for visualization")
    
    plt.figure(figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(G.number_of_nodes()))
    
    # Node size based on degree
    degrees = dict(G.degree())
    node_sizes = [50 + degrees[node] * 10 for node in G.nodes()]
    
    # Node color based on in-degree (cho directed)
    if G.is_directed():
        in_degrees = dict(G.in_degree())
        node_colors = [in_degrees[node] for node in G.nodes()]
    else:
        node_colors = [degrees[node] for node in G.nodes()]
    
    # Vẽ mạng
    nodes = nx.draw_networkx_nodes(G, pos, 
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 cmap=plt.cm.viridis,
                                 alpha=0.8)
    
    edges = nx.draw_networkx_edges(G, pos, 
                                 alpha=0.5,
                                 arrows=G.is_directed(),
                                 arrowsize=10,
                                 edge_color='gray')
    
    # Thêm colorbar
    plt.colorbar(nodes, label='Node Degree')
    plt.title(f'Network Visualization\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Network visualization saved: {save_path}")

def draw_community_network(G, communities, save_path, max_nodes=100):
    """Vẽ mạng với màu sắc theo community"""
    if not communities or 'partition' not in communities:
        print("❌ No community data available")
        return
    
    print(f"Drawing community network (max {max_nodes} nodes)...")
    
    partition = communities['partition']
    
    # Lấy subgraph nếu mạng quá lớn
    if G.number_of_nodes() > max_nodes:
        nodes_sample = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes_sample)
        partition = {node: comm for node, comm in partition.items() if node in nodes_sample}
        print(f"Using sample of {len(nodes_sample)} nodes for visualization")
    
    plt.figure(figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Màu sắc theo community
    unique_communities = set(partition.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
    color_map = {comm: colors[i] for i, comm in enumerate(unique_communities)}
    node_colors = [color_map[partition[node]] for node in G.nodes()]
    
    # Node size based on degree
    degrees = dict(G.degree())
    node_sizes = [30 + degrees[node] * 8 for node in G.nodes()]
    
    # Vẽ mạng
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.8)
    
    nx.draw_networkx_edges(G, pos, 
                          alpha=0.3,
                          arrows=False,
                          edge_color='gray')
    
    plt.title(f'Network with Communities\n({len(unique_communities)} communities, Modularity: {communities.get("modularity", 0):.3f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Community network visualization saved: {save_path}")

def main(config=None):
    """Main function for network drawing"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("NETWORK VISUALIZATION")
    print("="*60)
    
    try:
        G = config.get_network()
        print(f"✅ Network loaded: {G.number_of_nodes()} nodes")
    except:
        print("❌ No network data available")
        return None
    
    # Tạo thư mục lưu
    plots_dir = os.path.join(config.RESULTS_DIR, "charts")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Vẽ mạng cơ bản
    draw_network_basic(G, os.path.join(plots_dir, "network_visualization.png"))
    
    # Vẽ mạng với communities (nếu có)
    if hasattr(config, 'COMMUNITIES') and config.COMMUNITIES:
        draw_community_network(G, config.COMMUNITIES, 
                             os.path.join(plots_dir, "community_network.png"))
    
    print("✅ All network visualizations completed!")
    return True

if __name__ == "__main__":
    main()