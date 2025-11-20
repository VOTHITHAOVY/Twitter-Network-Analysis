# analysis/centrality_analysis.py
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter

def centrality_analysis(G):
    """
    Phân tích centrality measures - Chương 3
    """
    print("\n🎯 BẮT ĐẦU PHÂN TÍCH CENTRALITY")
    print("=" * 50)
    
    # 1. TÍNH CÁC CENTRALITY MEASURES
    print("📈 1. TÍNH TOÁN CENTRALITY MEASURES...")
    
    centrality_results = {}
    
    # Degree Centrality
    print("   • Degree Centrality...")
    centrality_results['degree'] = nx.degree_centrality(G)
    
    # Betweenness Centrality (dùng sampling cho mạng lớn)
    print("   • Betweenness Centrality...")
    k = min(100, G.number_of_nodes())  # Sample size
    centrality_results['betweenness'] = nx.betweenness_centrality(G, k=k, seed=42)
    
    # Closeness Centrality
    print("   • Closeness Centrality...")
    centrality_results['closeness'] = nx.closeness_centrality(G)
    
    # PageRank
    print("   • PageRank...")
    centrality_results['pagerank'] = nx.pagerank(G, alpha=0.85)
    
    # Eigenvector Centrality (nếu có thể)
    try:
        print("   • Eigenvector Centrality...")
        centrality_results['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        print("   • Eigenvector Centrality: Không thể tính (có thể do mạng không liên thông)")
    
    print("✅ Đã tính xong tất cả centrality measures")
    
    # 2. PHÂN TÍCH TOP NODES
    print("\n🏆 2. TOP INFLUENCERS:")
    analyze_top_nodes(centrality_results, G)
    
    # 3. PHÂN TÍCH TƯƠNG QUAN
    print("\n📊 3. PHÂN TÍCH TƯƠNG QUAN:")
    analyze_correlations(centrality_results)
    
    # 4. PHÂN TÍCH THEO NHÓM
    print("\n📋 4. PHÂN TÍCH THEO NHÓM:")
    analyze_node_categories(centrality_results, G)
    
    # 5. XUẤT KẾT QUẢ
    print("\n💾 5. XUẤT KẾT QUẢ:")
    export_centrality_results(centrality_results, G)
    
    print("\n✅ HOÀN THÀNH PHÂN TÍCH CENTRALITY")

def analyze_top_nodes(centrality_results, G, top_n=10):
    """Phân tích các nodes quan trọng nhất"""
    
    print(f"\n   📊 TOP {top_n} NODES THEO TỪNG ĐỘ ĐO:")
    
    top_nodes_by_measure = {}
    
    for measure_name, centrality_dict in centrality_results.items():
        if centrality_dict:  # Chỉ xử lý nếu có kết quả
            sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_nodes_by_measure[measure_name] = [node for node, _ in sorted_nodes]
            
            print(f"\n   🔹 {measure_name.upper()}:")
            for i, (node, score) in enumerate(sorted_nodes, 1):
                print(f"      {i:2d}. Node {node}: {score:.6f}")
    
    # Tìm super influencers (xuất hiện trong nhiều top lists)
    print(f"\n   🎯 SUPER INFLUENCERS:")
    all_top_nodes = []
    for nodes in top_nodes_by_measure.values():
        all_top_nodes.extend(nodes)
    
    node_counts = Counter(all_top_nodes)
    super_influencers = [(node, count) for node, count in node_counts.items() if count >= 2]
    
    if super_influencers:
        super_influencers.sort(key=lambda x: x[1], reverse=True)
        for node, count in super_influencers:
            print(f"      • Node {node}: xuất hiện trong {count} top lists")
            
            # Hiển thị scores của node này trong các measures
            scores_info = []
            for measure_name in centrality_results.keys():
                if measure_name in centrality_results and node in centrality_results[measure_name]:
                    score = centrality_results[measure_name][node]
                    scores_info.append(f"{measure_name}: {score:.4f}")
            print(f"        {', '.join(scores_info)}")
    else:
        print("      🤔 Không có node nào xuất hiện trong nhiều top lists")
    
    return top_nodes_by_measure, super_influencers

def analyze_correlations(centrality_results):
    """Phân tích tương quan giữa các centrality measures"""
    
    # Tạo DataFrame cho tất cả nodes
    data = {}
    valid_measures = [name for name, results in centrality_results.items() if results]
    
    for measure_name in valid_measures:
        data[measure_name] = list(centrality_results[measure_name].values())
    
    # Chuyển thành DataFrame
    min_length = min(len(values) for values in data.values())
    for key in data:
        data[key] = data[key][:min_length]
    
    df = pd.DataFrame(data)
    
    # Tính ma trận tương quan
    correlation_matrix = df.corr()
    
    print("\n   🔗 MA TRẬN TƯƠNG QUAN:")
    print("   " + " " * 12 + "".join([f"{col:12}" for col in correlation_matrix.columns]))
    for i, row_name in enumerate(correlation_matrix.index):
        row_str = f"   {row_name:12}"
        for j, col_name in enumerate(correlation_matrix.columns):
            row_str += f"{correlation_matrix.iloc[i, j]:12.3f}"
        print(row_str)
    
    # Phân tích cặp tương quan quan trọng
    print("\n   📈 PHÂN TÍCH TƯƠNG QUAN QUAN TRỌNG:")
    for i in range(len(valid_measures)):
        for j in range(i + 1, len(valid_measures)):
            measure1 = valid_measures[i]
            measure2 = valid_measures[j]
            corr = correlation_matrix.loc[measure1, measure2]
            
            if abs(corr) > 0.7:
                strength = "RẤT CAO" if abs(corr) > 0.8 else "CAO"
                direction = "dương" if corr > 0 else "âm"
                print(f"      • {measure1} vs {measure2}: {corr:.3f} ({strength}, {direction})")
    
    return correlation_matrix

def analyze_node_categories(centrality_results, G):
    """Phân tích nodes theo các nhóm centrality"""
    
    print("\n   🎪 PHÂN LOẠI NODES THEO VAI TRÒ:")
    
    # Lấy degree centrality làm cơ sở
    if 'degree' in centrality_results:
        degree_centrality = centrality_results['degree']
        
        # Phân loại theo degree
        degree_values = list(degree_centrality.values())
        thresholds = {
            'Rất thấp': np.percentile(degree_values, 25),
            'Trung bình': np.percentile(degree_values, 50),
            'Cao': np.percentile(degree_values, 75),
            'Rất cao': np.percentile(degree_values, 90)
        }
        
        print("   🔹 PHÂN LOẠI THEO DEGREE CENTRALITY:")
        for category, threshold in thresholds.items():
            count = len([v for v in degree_values if v >= threshold])
            print(f"      • {category}: {count} nodes (≥ {threshold:.4f})")
    
    # Phân tích nodes có betweenness cao nhưng degree thấp (cầu nối ẩn)
    if 'degree' in centrality_results and 'betweenness' in centrality_results:
        degree_threshold = np.percentile(list(degree_centrality.values()), 50)  # Median
        betweenness_threshold = np.percentile(list(centrality_results['betweenness'].values()), 75)  # Top 25%
        
        hidden_bridges = []
        for node in G.nodes():
            if (centrality_results['degree'][node] < degree_threshold and 
                centrality_results['betweenness'][node] > betweenness_threshold):
                hidden_bridges.append(node)
        
        print(f"\n   🌉 CẦU NỐI ẨN ({len(hidden_bridges)} nodes):")
        if hidden_bridges:
            for node in hidden_bridges[:5]:  # Hiển thị top 5
                print(f"      • Node {node}: degree={centrality_results['degree'][node]:.4f}, betweenness={centrality_results['betweenness'][node]:.4f}")
            if len(hidden_bridges) > 5:
                print(f"      • ... và {len(hidden_bridges) - 5} nodes khác")
        else:
            print("      🤔 Không tìm thấy cầu nối ẩn")

def export_centrality_results(centrality_results, G):
    """Xuất kết quả centrality ra file"""
    
    # Tạo DataFrame với tất cả centrality scores
    centrality_data = []
    for node in G.nodes():
        node_data = {'node_id': node}
        
        # Thêm degree thực tế
        node_data['degree'] = G.degree(node)
        if G.is_directed():
            node_data['in_degree'] = G.in_degree(node)
            node_data['out_degree'] = G.out_degree(node)
        
        # Thêm centrality measures
        for measure_name, centrality_dict in centrality_results.items():
            if centrality_dict and node in centrality_dict:
                node_data[measure_name] = centrality_dict[node]
            else:
                node_data[measure_name] = None
        
        centrality_data.append(node_data)
    
    df_centrality = pd.DataFrame(centrality_data)
    df_centrality.to_csv('centrality_results.csv', index=False, encoding='utf-8')
    print("   💾 Đã lưu: centrality_results.csv")
    
    # Tạo file summary cho top nodes
    top_nodes_summary = []
    for measure_name, centrality_dict in centrality_results.items():
        if centrality_dict:
            sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for rank, (node, score) in enumerate(sorted_nodes, 1):
                top_nodes_summary.append({
                    'measure': measure_name,
                    'rank': rank,
                    'node_id': node,
                    'score': score,
                    'degree': G.degree(node)
                })
    
    df_top = pd.DataFrame(top_nodes_summary)
    df_top.to_csv('top_centrality_nodes.csv', index=False, encoding='utf-8')
    print("   💾 Đã lưu: top_centrality_nodes.csv")
    
    # Thống kê
    print(f"\n   📊 THỐNG KÊ CENTRALITY:")
    print(f"      - Số nodes được phân tích: {len(df_centrality)}")
    print(f"      - Số centrality measures: {len(centrality_results)}")
    print(f"      - File đã xuất: centrality_results.csv, top_centrality_nodes.csv")

if __name__ == "__main__":
    # Test function
    print("🧪 TEST CENTRALITY ANALYSIS...")
    G = nx.erdos_renyi_graph(50, 0.2, seed=42)
    centrality_analysis(G)