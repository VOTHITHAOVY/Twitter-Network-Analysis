import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import pandas as pd

def export_gephi_files(G, centrality_dict=None, communities=None, output_dir="results/gephi"):
    """Xuất file cho Gephi"""
    print("Exporting files for Gephi...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export network (GEXF format - recommended for Gephi)
    gexf_path = os.path.join(output_dir, "network.gexf")
    nx.write_gexf(G, gexf_path)
    print(f"✅ Network exported: {gexf_path}")
    
    # 2. Export node attributes
    nodes_data = []
    
    for node in G.nodes():
        node_info = {'id': node}
        
        # Thêm degree information
        node_info['degree'] = G.degree(node)
        if G.is_directed():
            node_info['in_degree'] = G.in_degree(node)
            node_info['out_degree'] = G.out_degree(node)
        
        # Thêm centrality measures
        if centrality_dict:
            for measure, values in centrality_dict.items():
                if node in values:
                    node_info[measure] = values[node]
        
        # Thêm community information
        if communities and 'partition' in communities:
            node_info['community'] = communities['partition'].get(node, -1)
        
        nodes_data.append(node_info)
    
    # Lưu node attributes
    nodes_df = pd.DataFrame(nodes_data)
    nodes_csv_path = os.path.join(output_dir, "node_attributes.csv")
    nodes_df.to_csv(nodes_csv_path, index=False)
    print(f"✅ Node attributes exported: {nodes_csv_path}")
    
    # 3. Export edge list
    edges_csv_path = os.path.join(output_dir, "edge_list.csv")
    edge_data = []
    
    for edge in G.edges():
        edge_data.append({
            'source': edge[0],
            'target': edge[1],
            'weight': 1  # Có thể thay đổi nếu có weights
        })
    
    edges_df = pd.DataFrame(edge_data)
    edges_df.to_csv(edges_csv_path, index=False)
    print(f"✅ Edge list exported: {edges_csv_path}")
    
    # 4. Tạo file hướng dẫn
    readme_content = """
Gephi Import Guide:
1. Open Gephi
2. File -> Open -> Choose 'network.gexf'
3. The network will be loaded with basic structure
4. To import node attributes:
   - Go to Data Laboratory
   - Click 'Import Spreadsheet'
   - Choose 'node_attributes.csv'
   - Select 'Append to existing workspace'
   - Match columns appropriately

Exported Files:
- network.gexf: Main network file
- node_attributes.csv: Node properties (degree, centrality, communities)
- edge_list.csv: Edge list for reference
"""
    
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✅ Readme file created: {readme_path}")
    
    return {
        'gexf_file': gexf_path,
        'node_attributes': nodes_csv_path,
        'edge_list': edges_csv_path
    }

def main(config=None):
    """Main function for Gephi export"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("GEPHI EXPORT")
    print("="*60)
    
    try:
        G = config.get_network()
        print(f"✅ Network loaded: {G.number_of_nodes()} nodes")
    except:
        print("❌ No network data available")
        return None
    
    # Thu thập centrality và communities từ config
    centrality_dict = getattr(config, 'CENTRALITY', None)
    communities = getattr(config, 'COMMUNITIES', None)
    
    # Xuất file cho Gephi
    output_dir = os.path.join(config.RESULTS_DIR, "gephi")
    exported_files = export_gephi_files(G, centrality_dict, communities, output_dir)
    
    print(f"\n✅ All files exported to: {output_dir}")
    print("🎨 You can now open these files in Gephi for advanced visualization!")
    
    return exported_files

if __name__ == "__main__":
    main()