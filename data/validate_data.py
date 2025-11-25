import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import pandas as pd
import numpy as np

def validate_network_structure(G):
    """Kiểm tra cấu trúc mạng"""
    print("Validating network structure...")
    
    validation_results = {
        'is_directed': G.is_directed(),
        'has_self_loops': any(nx.selfloop_edges(G)),
        'has_isolated_nodes': any(G.degree(node) == 0 for node in G.nodes()),
        'is_weakly_connected': nx.is_weakly_connected(G),
        'is_strongly_connected': nx.is_strongly_connected(G) if G.is_directed() else True,
        'is_multigraph': G.is_multigraph(),
        'has_negative_weights': False  # Assuming no weights for now
    }
    
    # Kiểm tra node IDs
    node_ids = list(G.nodes())
    validation_results['all_nodes_integer'] = all(isinstance(node, (int, np.integer)) for node in node_ids)
    validation_results['node_id_range'] = (min(node_ids), max(node_ids)) if node_ids else (0, 0)
    
    # Kiểm tra edge weights (nếu có)
    if nx.is_weighted(G):
        edge_weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
        validation_results['has_negative_weights'] = any(weight < 0 for weight in edge_weights)
        validation_results['weight_range'] = (min(edge_weights), max(edge_weights))
    
    return validation_results

def check_data_quality(G):
    """Kiểm tra chất lượng dữ liệu"""
    print("Checking data quality...")
    
    quality_metrics = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'network_density': nx.density(G),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'degree_assortativity': nx.degree_assortativity_coefficient(G),
        'reciprocity': nx.reciprocity(G) if G.is_directed() else 1.0
    }
    
    # Phân tích degree distribution
    if G.is_directed():
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        quality_metrics['avg_in_degree'] = np.mean(in_degrees)
        quality_metrics['avg_out_degree'] = np.mean(out_degrees)
        quality_metrics['max_in_degree'] = np.max(in_degrees)
        quality_metrics['max_out_degree'] = np.max(out_degrees)
    else:
        degrees = [d for _, d in G.degree()]
        quality_metrics['avg_degree'] = np.mean(degrees)
        quality_metrics['max_degree'] = np.max(degrees)
    
    return quality_metrics

def generate_validation_report(validation_results, quality_metrics, G):
    """Tạo báo cáo kiểm tra"""
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)
    
    print("\n📋 STRUCTURE VALIDATION:")
    for key, value in validation_results.items():
        status = "✅ PASS" if value in [True, False] and value != True else "❌ FAIL" if value in [True, False] and value == True else f"ℹ️  {value}"
        print(f"  {key}: {status}")
    
    print("\n📊 QUALITY METRICS:")
    for key, value in quality_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Đánh giá tổng quan
    print("\n🎯 OVERALL ASSESSMENT:")
    
    issues = []
    if validation_results['has_self_loops']:
        issues.append("Self-loops present")
    if validation_results['has_isolated_nodes']:
        issues.append("Isolated nodes present")
    if not validation_results['all_nodes_integer']:
        issues.append("Non-integer node IDs")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No major issues found")
    
    # Đề xuất
    print("\n💡 RECOMMENDATIONS:")
    if quality_metrics['network_density'] < 0.001:
        print("  - Network is very sparse")
    if quality_metrics['average_degree'] < 2:
        print("  - Low average degree, consider filtering")
    if not validation_results['is_weakly_connected']:
        print("  - Network is disconnected, consider using largest component")

def main():
    """Main function for data validation"""
    config = Config()
    
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    # Tải mạng
    file_path = os.path.join(config.DATA_DIR, "higgs-retweet_network.edgelist")
    
    try:
        G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int, data=False)
        print(f"✅ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"❌ Error loading network: {e}")
        return None
    
    # Kiểm tra cấu trúc
    validation_results = validate_network_structure(G)
    
    # Kiểm tra chất lượng
    quality_metrics = check_data_quality(G)
    
    # Tạo báo cáo
    generate_validation_report(validation_results, quality_metrics, G)
    
    # Lưu kết quả
    results = {
        'validation': validation_results,
        'quality': quality_metrics
    }
    
    return results

if __name__ == "__main__":
    main()