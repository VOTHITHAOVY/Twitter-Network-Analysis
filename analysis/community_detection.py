# analysis/community_detection.py
import networkx as nx
import pandas as pd
import numpy as np
from community import community_louvain
from networkx.algorithms.community import greedy_modularity_communities, label_propagation_communities
import time
from collections import Counter

def community_analysis(G):
    """
    Phân tích community detection - Chương 4
    """
    print("\n👥 BẮT ĐẦU PHÂN TÍCH COMMUNITY DETECTION")
    print("=" * 50)
    
    # Chuyển sang đồ thị vô hướng cho community detection
    G_undirected = G.to_undirected()
    
    # 1. SO SÁNH CÁC THUẬT TOÁN
    print("🔍 1. SO SÁNH THUẬT TOÁN COMMUNITY DETECTION:")
    algorithms_results = compare_community_algorithms(G_undirected)
    
    # 2. PHÂN TÍCH CHI TIẾT COMMUNITIES
    print("\n📊 2. PHÂN TÍCH CHI TIẾT COMMUNITIES:")
    best_algorithm = select_best_algorithm(algorithms_results)
    detailed_analysis = analyze_communities_detail(G_undirected, algorithms_results[best_algorithm])
    
    # 3. K-CORE DECOMPOSITION
    print("\n🎯 3. K-CORE DECOMPOSITION:")
    kcore_analysis = perform_kcore_analysis(G_undirected)
    
    # 4. KẾT HỢP COMMUNITIES VÀ CENTRALITY
    print("\n🔗 4. KẾT HỢP COMMUNITIES VÀ CENTRALITY:")
    analyze_communities_centrality(G, algorithms_results[best_algorithm])
    
    # 5. XUẤT KẾT QUẢ
    print("\n💾 5. XUẤT KẾT QUẢ:")
    export_community_results(algorithms_results, detailed_analysis, kcore_analysis, G)
    
    print("\n✅ HOÀN THÀNH PHÂN TÍCH COMMUNITY DETECTION")

def compare_community_algorithms(G):
    """So sánh các thuật toán community detection"""
    
    algorithms = {
        'Louvain': detect_louvain_communities,
        'Greedy Modularity': detect_greedy_modularity_communities,
        'Label Propagation': detect_label_propagation_communities
    }
    
    results = {}
    
    for algo_name, algo_func in algorithms.items():
        print(f"\n   🧮 {algo_name}...")
        start_time = time.time()
        
        try:
            communities, modularity, additional_info = algo_func(G)
            execution_time = time.time() - start_time
            
            results[algo_name] = {
                'communities': communities,
                'modularity': modularity,
                'execution_time': execution_time,
                'n_communities': len(communities),
                'additional_info': additional_info
            }
            
            print(f"      ✅ Thành công:")
            print(f"        - Số communities: {len(communities)}")
            print(f"        - Modularity: {modularity:.4f}")
            print(f"        - Thời gian: {execution_time:.2f}s")
            
            if additional_info:
                for key, value in additional_info.items():
                    print(f"        - {key}: {value}")
                    
        except Exception as e:
            print(f"      ❌ Thất bại: {e}")
            results[algo_name] = None
    
    return results

def detect_louvain_communities(G):
    """Phát hiện communities bằng Louvain algorithm"""
    partition = community_louvain.best_partition(G)
    
    # Chuyển partition thành danh sách communities
    communities_dict = {}
    for node, comm_id in partition.items():
        if comm_id not in communities_dict:
            communities_dict[comm_id] = []
        communities_dict[comm_id].append(node)
    
    communities = list(communities_dict.values())
    modularity = community_louvain.modularity(partition, G)
    
    # Thông tin thêm
    additional_info = {
        'Community sizes': [len(comm) for comm in communities]
    }
    
    return communities, modularity, additional_info

def detect_greedy_modularity_communities(G):
    """Phát hiện communities bằng Greedy Modularity"""
    communities = list(greedy_modularity_communities(G))
    modularity = nx.algorithms.community.quality.modularity(G, communities)
    
    additional_info = {
        'Community sizes': [len(comm) for comm in communities]
    }
    
    return communities, modularity, additional_info

def detect_label_propagation_communities(G):
    """Phát hiện communities bằng Label Propagation"""
    communities = list(label_propagation_communities(G))
    
    # Tính modularity cho label propagation
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    modularity = community_louvain.modularity(partition, G)
    
    additional_info = {
        'Community sizes': [len(comm) for comm in communities]
    }
    
    return communities, modularity, additional_info

def select_best_algorithm(algorithms_results):
    """Chọn thuật toán tốt nhất dựa trên modularity và thời gian"""
    
    valid_results = {name: result for name, result in algorithms_results.items() if result is not None}
    
    if not valid_results:
        print("   ❌ Không có thuật toán nào chạy thành công")
        return None
    
    # Đánh giá dựa trên modularity (quan trọng hơn) và thời gian
    scores = {}
    for algo_name, result in valid_results.items():
        modularity_score = result['modularity']
        time_penalty = result['execution_time'] / 10  # Penalty cho thời gian dài
        total_score = modularity_score - time_penalty
        scores[algo_name] = total_score
    
    best_algorithm = max(scores.items(), key=lambda x: x[1])[0]
    best_result = valid_results[best_algorithm]
    
    print(f"\n   🏆 THUẬT TOÁN TỐT NHẤT: {best_algorithm}")
    print(f"      - Modularity: {best_result['modularity']:.4f}")
    print(f"      - Thời gian: {best_result['execution_time']:.2f}s")
    print(f"      - Số communities: {best_result['n_communities']}")
    
    return best_algorithm

def analyze_communities_detail(G, algorithm_result):
    """Phân tích chi tiết communities"""
    
    communities = algorithm_result['communities']
    modularity = algorithm_result['modularity']
    
    print(f"\n   📈 PHÂN TÍCH CHI TIẾT COMMUNITIES:")
    print(f"      - Tổng số communities: {len(communities)}")
    print(f"      - Modularity: {modularity:.4f}")
    
    # Phân tích kích thước communities
    comm_sizes = [len(comm) for comm in communities]
    print(f"      - Kích thước communities:")
    print(f"        • Lớn nhất: {max(comm_sizes)} nodes")
    print(f"        • Nhỏ nhất: {min(comm_sizes)} nodes") 
    print(f"        • Trung bình: {np.mean(comm_sizes):.1f} nodes")
    print(f"        • Độ lệch chuẩn: {np.std(comm_sizes):.1f}")
    
    # Phân loại communities theo kích thước
    size_categories = {
        'Rất nhỏ (1-5 nodes)': len([s for s in comm_sizes if 1 <= s <= 5]),
        'Nhỏ (6-15 nodes)': len([s for s in comm_sizes if 6 <= s <= 15]),
        'Trung bình (16-30 nodes)': len([s for s in comm_sizes if 16 <= s <= 30]),
        'Lớn (31-50 nodes)': len([s for s in comm_sizes if 31 <= s <= 50]),
        'Rất lớn (>50 nodes)': len([s for s in comm_sizes if s > 50])
    }
    
    print(f"      - Phân bố kích thước:")
    for category, count in size_categories.items():
        if count > 0:
            percentage = (count / len(communities)) * 100
            print(f"        • {category}: {count} communities ({percentage:.1f}%)")
    
    # Tính internal density cho mỗi community
    print(f"\n      🎯 CHẤT LƯỢNG COMMUNITIES:")
    internal_densities = []
    
    for i, comm in enumerate(communities):
        if len(comm) > 1:
            subgraph = G.subgraph(comm)
            density = nx.density(subgraph)
            internal_densities.append(density)
        else:
            internal_densities.append(0)
    
    print(f"        • Internal density trung bình: {np.mean(internal_densities):.4f}")
    print(f"        • Internal density lớn nhất: {max(internal_densities):.4f}")
    
    return {
        'n_communities': len(communities),
        'modularity': modularity,
        'community_sizes': comm_sizes,
        'internal_densities': internal_densities,
        'size_categories': size_categories
    }

def perform_kcore_analysis(G):
    """Thực hiện K-core decomposition"""
    
    print(f"\n   🔍 K-CORE DECOMPOSITION:")
    
    core_numbers = nx.core_number(G)
    max_k = max(core_numbers.values())
    
    print(f"      - Core number lớn nhất: {max_k}")
    print(f"      - Core number trung bình: {np.mean(list(core_numbers.values())):.2f}")
    
    # Thống kê số nodes cho mỗi k-core
    kcore_stats = {}
    for k in range(1, max_k + 1):
        k_core = nx.k_core(G, k)
        kcore_stats[k] = {
            'n_nodes': k_core.number_of_nodes(),
            'density': nx.density(k_core) if k_core.number_of_nodes() > 1 else 0
        }
    
    print(f"      - Số nodes trong các K-core:")
    for k in sorted(kcore_stats.keys()):
        stats = kcore_stats[k]
        percentage = (stats['n_nodes'] / G.number_of_nodes()) * 100
        print(f"        • {k}-core: {stats['n_nodes']} nodes ({percentage:.1f}%), density: {stats['density']:.4f}")
    
    return {
        'core_numbers': core_numbers,
        'max_k': max_k,
        'kcore_stats': kcore_stats
    }

def analyze_communities_centrality(G, algorithm_result):
    """Phân tích kết hợp communities và centrality"""
    
    communities = algorithm_result['communities']
    
    # Tính degree centrality cho toàn mạng
    degree_centrality = nx.degree_centrality(G)
    
    print(f"\n   🌟 INFLUENCERS TRONG COMMUNITIES:")
    
    for i, comm in enumerate(communities):
        if len(comm) >= 5:  # Chỉ xét communities có ít nhất 5 nodes
            # Tìm node có degree cao nhất trong community
            comm_degrees = [(node, degree_centrality[node]) for node in comm]
            top_node, top_degree = max(comm_degrees, key=lambda x: x[1])
            
            print(f"      • Community {i} ({len(comm)} nodes):")
            print(f"        - Influencer: Node {top_node} (degree centrality: {top_degree:.4f})")
            print(f"        - Top 3 nodes: {[node for node, _ in sorted(comm_degrees, key=lambda x: x[1], reverse=True)[:3]]}")

def export_community_results(algorithms_results, detailed_analysis, kcore_analysis, G):
    """Xuất kết quả community analysis"""
    
    # Xuất kết quả so sánh thuật toán
    algo_data = []
    for algo_name, result in algorithms_results.items():
        if result:
            algo_data.append({
                'algorithm': algo_name,
                'n_communities': result['n_communities'],
                'modularity': result['modularity'],
                'execution_time': result['execution_time'],
                'status': 'Success'
            })
        else:
            algo_data.append({
                'algorithm': algo_name,
                'n_communities': 0,
                'modularity': 0,
                'execution_time': 0,
                'status': 'Failed'
            })
    
    df_algorithms = pd.DataFrame(algo_data)
    df_algorithms.to_csv('community_algorithms_comparison.csv', index=False, encoding='utf-8')
    print("   💾 Đã lưu: community_algorithms_comparison.csv")
    
    # Xuất kết quả K-core
    kcore_data = []
    for k, stats in kcore_analysis['kcore_stats'].items():
        kcore_data.append({
            'k_value': k,
            'n_nodes': stats['n_nodes'],
            'density': stats['density'],
            'percentage': (stats['n_nodes'] / G.number_of_nodes()) * 100
        })
    
    df_kcore = pd.DataFrame(kcore_data)
    df_kcore.to_csv('kcore_analysis.csv', index=False, encoding='utf-8')
    print("   💾 Đã lưu: kcore_analysis.csv")
    
    # Xuất community assignments (cho thuật toán tốt nhất)
    best_algorithm = select_best_algorithm(algorithms_results)
    if best_algorithm:
        communities = algorithms_results[best_algorithm]['communities']
        community_assignments = []
        
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                community_assignments.append({
                    'node_id': node,
                    'community_id': comm_id,
                    'community_size': len(comm_nodes)
                })
        
        df_assignments = pd.DataFrame(community_assignments)
        df_assignments.to_csv('community_assignments.csv', index=False, encoding='utf-8')
        print("   💾 Đã lưu: community_assignments.csv")
    
    print(f"\n   📊 THỐNG KÊ COMMUNITY ANALYSIS:")
    print(f"      - Số thuật toán so sánh: {len(algorithms_results)}")
    print(f"      - Số communities (best): {detailed_analysis['n_communities']}")
    print(f"      - Modularity (best): {detailed_analysis['modularity']:.4f}")
    print(f"      - K-core max: {kcore_analysis['max_k']}")

if __name__ == "__main__":
    # Test function
    print("🧪 TEST COMMUNITY DETECTION...")
    G = nx.erdos_renyi_graph(100, 0.1, seed=42)
    community_analysis(G)