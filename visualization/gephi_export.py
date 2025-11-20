# visualization/gephi_export.py
import networkx as nx
import pandas as pd
from community import community_louvain

def export_to_gephi(G):
    """Xuất file cho Gephi visualization - YÊU CẦU CHƯƠNG 2"""
    print("\n📤 XUẤT FILE CHO GEPHI...")
    
    # Chuyển sang đồ thị vô hướng cho community detection
    G_undirected = G.to_undirected()
    
    # 1. Tính các thuộc tính cho visualization
    print("• Tính centrality measures...")
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=100)
    pagerank = nx.pagerank(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    print("• Phát hiện communities...")
    partition = community_louvain.best_partition(G_undirected)
    
    print("• Tính k-core...")
    core_numbers = nx.core_number(G_undirected)
    
    # 2. Tạo node dataframe
    print("• Tạo node data...")
    nodes_data = []
    for node in G.nodes():
        nodes_data.append({
            'Id': node,
            'Label': f'User_{node}',
            'Degree': G.degree(node),
            'InDegree': G.in_degree(node),
            'OutDegree': G.out_degree(node),
            'DegreeCentrality': degree_centrality[node],
            'BetweennessCentrality': betweenness_centrality[node],
            'ClosenessCentrality': closeness_centrality[node],
            'PageRank': pagerank[node],
            'Community': partition[node],
            'KCore': core_numbers[node],
            'Size': degree_centrality[node] * 50 + 5  # Kích thước node
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # 3. Tạo edge dataframe  
    print("• Tạo edge data...")
    edges_data = []
    for edge in G.edges():
        edges_data.append({
            'Source': edge[0],
            'Target': edge[1],
            'Type': 'Directed',
            'Weight': 1
        })
    
    edges_df = pd.DataFrame(edges_data)
    
    # 4. Xuất file
    nodes_df.to_csv('gephi_nodes.csv', index=False, encoding='utf-8')
    edges_df.to_csv('gephi_edges.csv', index=False, encoding='utf-8')
    
    print("✅ Đã xuất file:")
    print("   - gephi_nodes.csv (chứa nodes và thuộc tính)")
    print("   - gephi_edges.csv (chứa edges)")
    
    # 5. Thống kê file
    print(f"\n📊 THỐNG KÊ FILE:")
    print(f"   - Số nodes: {len(nodes_df)}")
    print(f"   - Số edges: {len(edges_df)}")
    print(f"   - Số communities: {nodes_df['Community'].nunique()}")
    print(f"   - K-core max: {nodes_df['KCore'].max()}")
    
    print("\n🎨 HƯỚNG DẪN SỬ DỤNG GEPHI:")
    print("1. Mở Gephi → New Project")
    print("2. Data Laboratory → Import Spreadsheet")
    print("3. Chọn gephi_nodes.csv → Import as: Nodes table")
    print("4. Chọn gephi_edges.csv → Import as: Edges table") 
    print("5. Overview → Layout: Force Atlas 2")
    print("6. Appearance → Nodes → Color: Partition → Community")
    print("7. Appearance → Nodes → Size: Ranking → DegreeCentrality")
    print("8. Run Layout và điều chỉnh parameters")
    
    return nodes_df, edges_df