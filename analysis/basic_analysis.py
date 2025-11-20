# analysis/basic_analysis.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

def basic_network_analysis(G):
    """
    Phân tích tổng quan mạng - Chương 2
    """
    print("\n📊 BẮT ĐẦU PHÂN TÍCH TỔNG QUAN MẠNG")
    print("=" * 50)
    
    # 1. THỐNG KÊ CƠ BẢN
    print("🔢 1. THỐNG KÊ CƠ BẢN:")
    print(f"   • Số nodes: {G.number_of_nodes()}")
    print(f"   • Số edges: {G.number_of_edges()}")
    print(f"   • Đồ thị có hướng: {G.is_directed()}")
    print(f"   • Mật độ mạng: {nx.density(G):.6f}")
    
    # 2. PHÂN TÍCH LIÊN THÔNG
    print("\n🔗 2. PHÂN TÍCH LIÊN THÔNG:")
    if G.is_directed():
        n_components = nx.number_weakly_connected_components(G)
        print(f"   • Số thành phần liên thông yếu: {n_components}")
        
        # Lấy giant component
        giant_component = max(nx.weakly_connected_components(G), key=len)
        G_giant = G.subgraph(giant_component)
        print(f"   • Giant component: {len(giant_component)} nodes ({len(giant_component)/G.number_of_nodes()*100:.1f}%)")
    else:
        n_components = nx.number_connected_components(G)
        print(f"   • Số thành phần liên thông: {n_components}")
        
        giant_component = max(nx.connected_components(G), key=len)
        G_giant = G.subgraph(giant_component)
        print(f"   • Giant component: {len(giant_component)} nodes ({len(giant_component)/G.number_of_nodes()*100:.1f}%)")
    
    # 3. TÍNH ĐƯỜNG KÍNH VÀ BÁN KÍNH
    print("\n📏 3. ĐƯỜNG KÍNH VÀ BÁN KÍNH:")
    calculate_diameter_radius(G_giant)
    
    # 4. PHÂN PHỐI BẬC
    print("\n🎯 4. PHÂN PHỐI BẬC:")
    analyze_degree_distribution(G)
    
    # 5. HỆ SỐ PHÂN CỤM
    print("\n🔍 5. HỆ SỐ PHÂN CỤM:")
    analyze_clustering_coefficient(G)
    
    # 6. ĐỘ DÀI ĐƯỜNG ĐI TRUNG BÌNH
    print("\n🛣️  6. ĐỘ DÀI ĐƯỜNG ĐI TRUNG BÌNH:")
    analyze_average_path_length(G_giant)
    
    # 7. TẠO BÁO CÁO TỔNG HỢP
    print("\n📋 7. BÁO CÁO TỔNG HỢP:")
    create_summary_report(G, G_giant)
    
    print("\n✅ HOÀN THÀNH PHÂN TÍCH TỔNG QUAN")

def calculate_diameter_radius(G):
    """Tính đường kính và bán kính của mạng"""
    try:
        if G.is_directed():
            # Với đồ thị có hướng, tính trên đồ thị vô hướng
            G_undirected = G.to_undirected()
            if nx.is_connected(G_undirected):
                diameter = nx.diameter(G_undirected)
                radius = nx.radius(G_undirected)
                print(f"   • Đường kính: {diameter}")
                print(f"   • Bán kính: {radius}")
            else:
                print("   • Mạng không liên thông - không tính được đường kính/bán kính")
        else:
            if nx.is_connected(G):
                diameter = nx.diameter(G)
                radius = nx.radius(G)
                print(f"   • Đường kính: {diameter}")
                print(f"   • Bán kính: {radius}")
            else:
                print("   • Mạng không liên thông - không tính được đường kính/bán kính")
    except Exception as e:
        print(f"   • Lỗi khi tính đường kính/bán kính: {e}")

def analyze_degree_distribution(G):
    """Phân tích phân phối bậc"""
    if G.is_directed():
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        degrees = in_degrees + out_degrees
        print(f"   • Degree trung bình: {np.mean(degrees):.2f}")
        print(f"   • In-degree trung bình: {np.mean(in_degrees):.2f}")
        print(f"   • Out-degree trung bình: {np.mean(out_degrees):.2f}")
        print(f"   • Degree lớn nhất: {max(degrees)}")
        print(f"   • Degree nhỏ nhất: {min(degrees)}")
    else:
        degrees = [d for n, d in G.degree()]
        print(f"   • Degree trung bình: {np.mean(degrees):.2f}")
        print(f"   • Degree lớn nhất: {max(degrees)}")
        print(f"   • Degree nhỏ nhất: {min(degrees)}")
    
    # Phân tích phân phối
    degree_counts = Counter(degrees)
    print(f"   • Số nodes có degree = 1: {degree_counts.get(1, 0)}")
    print(f"   • Số nodes có degree > 10: {len([d for d in degrees if d > 10])}")
    
    # Power-law fitting đơn giản
    if len(degrees) > 10:
        try:
            from scipy import stats
            # Lọc degrees > 0 để tránh log(0)
            positive_degrees = [d for d in degrees if d > 0]
            if len(positive_degrees) > 5:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    np.log(np.arange(1, len(positive_degrees) + 1)), 
                    np.log(sorted(positive_degrees, reverse=True))
                )
                print(f"   • Power-law exponent (ước lượng): {-slope:.2f}")
                print(f"   • R²: {r_value**2:.3f}")
        except:
            print("   • Không thể ước lượng power-law exponent")

def analyze_clustering_coefficient(G):
    """Phân tích hệ số phân cụm - ĐÃ FIX HIỂN THỊ"""
    try:
        print("   • Đang tính hệ số phân cụm...")
        
        if G.is_directed():
            G_undirected = G.to_undirected()
        else:
            G_undirected = G
        
        clustering_global = nx.average_clustering(G_undirected)
        clustering_local = nx.clustering(G_undirected)
        
        # Lọc các nodes có clustering hợp lệ
        nodes_with_edges = [node for node in G_undirected.nodes() if G_undirected.degree(node) >= 1]
        valid_clustering_values = [clustering_local[node] for node in nodes_with_edges]
        
        print(f"   • Hệ số phân cụm toàn cục: {clustering_global:.4f}")
        print(f"   • Hệ số phân cụm trung bình: {np.mean(valid_clustering_values):.4f}")
        print(f"   • Hệ số phân cụm lớn nhất: {max(valid_clustering_values):.4f}")
        print(f"   • Hệ số phân cụm nhỏ nhất: {min(valid_clustering_values):.4f}")
        
        # Vẽ và LƯU hình NHƯNG KHÔNG HIỂN THỊ
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(valid_clustering_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Hệ số phân cụm cục bộ')
        plt.ylabel('Số nodes')
        plt.title('PHÂN BỐ HỆ SỐ PHÂN CỤM CỤC BỘ')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(valid_clustering_values)
        plt.ylabel('Hệ số phân cụm')
        plt.title('BOXPLOT HỆ SỐ PHÂN CỤM')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clustering_coefficient.png', dpi=300, bbox_inches='tight')
        plt.close()  # QUAN TRỌNG: Đóng figure thay vì show()
        print("   💾 Đã lưu: clustering_coefficient.png")
        
    except Exception as e:
        print(f"   • Lỗi khi tính hệ số phân cụm: {e}")
        print("   • Bỏ qua phần này và tiếp tục...")

def analyze_average_path_length(G):
    """Phân tích độ dài đường đi trung bình"""
    try:
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            print(f"   • Độ dài đường đi trung bình: {avg_path_length:.2f}")
            
            # Tính phân bố đường đi
            path_lengths = []
            nodes = list(G.nodes())
            # Lấy mẫu để tính nhanh
            sample_size = min(50, len(nodes))
            sample_nodes = np.random.choice(nodes, size=sample_size, replace=False)
            
            for i, source in enumerate(sample_nodes):
                lengths = nx.single_source_shortest_path_length(G, source)
                path_lengths.extend(list(lengths.values()))
            
            # Vẽ histogram
            plt.figure(figsize=(8, 4))
            plt.hist(path_lengths, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel('Độ dài đường đi')
            plt.ylabel('Tần suất')
            plt.title('PHÂN BỐ ĐỘ DÀI ĐƯỜNG ĐI')
            plt.grid(True, alpha=0.3)
            plt.savefig('path_length_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()  # FIX: Đóng figure
            print("   💾 Đã lưu: path_length_distribution.png")
            
        else:
            print("   • Mạng không liên thông - không tính được độ dài đường đi trung bình")
    except Exception as e:
        print(f"   • Lỗi khi tính độ dài đường đi: {e}")

def create_summary_report(G, G_giant):
    """Tạo báo cáo tổng hợp"""
    summary = {
        'Tổng số nodes': G.number_of_nodes(),
        'Tổng số edges': G.number_of_edges(),
        'Mật độ mạng': f"{nx.density(G):.6f}",
        'Đồ thị có hướng': G.is_directed(),
        'Kích thước giant component': f"{G_giant.number_of_nodes()} ({G_giant.number_of_nodes()/G.number_of_nodes()*100:.1f}%)",
    }
    
    # Thêm degree statistics
    if G.is_directed():
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        degrees = in_degrees + out_degrees
        summary['Degree trung bình'] = f"{np.mean(degrees):.2f}"
        summary['In-degree trung bình'] = f"{np.mean(in_degrees):.2f}"
        summary['Out-degree trung bình'] = f"{np.mean(out_degrees):.2f}"
    else:
        degrees = [d for n, d in G.degree()]
        summary['Degree trung bình'] = f"{np.mean(degrees):.2f}"
    
    print("   📊 BÁO CÁO TỔNG HỢP:")
    for key, value in summary.items():
        print(f"      {key}: {value}")
    
    # Xuất ra file CSV
    df_summary = pd.DataFrame(list(summary.items()), columns=['Chỉ số', 'Giá trị'])
    df_summary.to_csv('network_summary.csv', index=False, encoding='utf-8')
    print("   💾 Đã lưu: network_summary.csv")

if __name__ == "__main__":
    # Test function
    print("🧪 TEST BASIC ANALYSIS...")
    G = nx.erdos_renyi_graph(100, 0.1, seed=42)
    basic_network_analysis(G)