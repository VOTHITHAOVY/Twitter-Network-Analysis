# main.py - FILE CHÍNH CHẠY TOÀN BỘ PHÂN TÍCH - ĐÃ FIX
import matplotlib
matplotlib.use('Agg')  # QUAN TRỌNG: Tắt hiển thị đồ họa

from data.data_loader import load_network_data
from analysis.basic_analysis import basic_network_analysis
from analysis.centrality_analysis import centrality_analysis
from analysis.community_detection import community_analysis
from analysis.random_graph_comparison import random_graph_comparison
from visualization.gephi_export import export_to_gephi
from visualization.network_plots import create_all_visualizations

def main():
    print("=" * 60)
    print("🚀 BẮT ĐẦU PHÂN TÍCH MẠNG TWITTER HIGGS (KHÔNG HIỂN THỊ ĐỒ HỌA)")
    print("=" * 60)
    
    try:
        # 1. Load data
        print("\n📥 1. ĐANG LOAD DỮ LIỆU...")
        G = load_network_data()
        
        # 2. Phân tích cơ bản (Chương 2)
        print("\n📊 2. PHÂN TÍCH CƠ BẢN...")
        basic_network_analysis(G)
        
        # 3. Phân tích centrality (Chương 3)
        print("\n🎯 3. PHÂN TÍCH CENTRALITY...")
        centrality_analysis(G)
        
        # 4. Phân tích community (Chương 4)
        print("\n👥 4. PHÂN TÍCH COMMUNITY...")
        community_analysis(G)
        
        # 5. So sánh random graph (Chương 3)
        print("\n🔁 5. SO SÁNH RANDOM GRAPH...")
        random_graph_comparison(G)
        
        # 6. Visualization
        print("\n🎨 6. TẠO VISUALIZATION...")
        create_all_visualizations(G)
        
        # 7. Xuất Gephi (Chương 2)
        print("\n📤 7. XUẤT FILE GEPHI...")
        export_to_gephi(G)
        
        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH TẤT CẢ PHÂN TÍCH!")
        print("👉 Kiểm tra các file .png và .csv đã được tạo")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ CÓ LỖI XẢY RA: {e}")
        print("💡 Kiểm tra các file đã được tạo và chạy lại phần còn lại")

if __name__ == "__main__":
    main()