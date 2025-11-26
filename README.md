# ĐỀ TÀI PHÂN TÍCH MẠNG XÃ HỘI 

# Giảng Viên Hướng Dẫn : Đỗ Như Tài


# Nhóm Sinh Viên Thực Hiện

| STT | Họ Tên | MSSV | Vai trò |
|-----|---------|------|---------|
| 1 | **Võ Thị Thảo Vy** | 3124411356 | Trưởng nhóm |
| 2 | **Nguyễn Như Thảo** | 3124411281 | Thành viên |
| 3 | **Đặng Đào Đạt Thành** |3124411274 | Thành viên |
| 4 | **Lê Tân Phước Thọ** | 3124411294 | Thành viên |

---

##  Giới Thiệu Dự Án

Dự án **Phân Tích Mạng Xã Hội** cung cấp một hệ thống toàn diện để phân tích các mạng xã hội từ Twitter và các nền tảng khác. Hệ thống tự động hóa toàn bộ quy trình từ thu thập dữ liệu, phân tích đến trực quan hóa kết quả.

##  Tính Năng Chính

###  Đã Hoàn Thành

| Module | Tính Năng | Mô Tả |
|--------|-----------|--------|
| ** Data Loading** | Tự động nhận diện dataset | Hỗ trợ dataset thật và tự tạo |
| ** Basic Analysis** | Thống kê cơ bản | Nodes, edges, density, clustering |
| ** Degree Analysis** | Phân phối bậc | Power-law fitting, assortativity |
| ** Centrality Analysis** | Độ đo trung tâm | 6 loại centrality measures |
| ** Community Detection** | Phát hiện cộng đồng | Louvain, Girvan-Newman |
| ** Correlation Analysis** | Tương quan | Pearson, Spearman correlation |
| ** Visualization** | Trực quan hóa | Biểu đồ tự động |
| ** Report Generation** | Xuất báo cáo | CSV, charts, summary |

## 🛠 Cài Đặt

### Yêu Cầu Hệ Thống

| Thành Phần | Yêu Cầu Tối Thiểu | Khuyến Nghị |
|------------|-------------------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4GB | 8GB+ |
| Storage | 1GB | 2GB+ |

### Cài Đặt Thư Viện

```bash
# Clone repository
git clone https://github.com/your-username/social-network-analysis.git
cd social-network-analysis

# Cài đặt dependencies
pip install -r requirements.txt
```

### Requirements

```txt
networkx>=3.0
pandas>=1.5.0
matplotlib>=3.5.0
numpy>=1.23.0
scipy>=1.9.0
seaborn>=0.12.0
python-louvain>=0.16
powerlaw>=1.5
```

## 📁 Cấu Trúc Dự Án

```
social-network-analysis/
├──  main.py                       # File chính chạy toàn bộ
├──  requirements.txt              # Danh sách thư viện
├──  README.md                     # Hướng dẫn sử dụng
│
├──  analysis/                     # Phân tích chuyên sâu
│   ├── basic_analysis.py           # CHƯƠNG 2: Phân tích tổng quan
│   ├── degree_analysis.py          # Phân phối bậc & hồi quy
│   ├── centrality.py               # CHƯƠNG 3: Phân tích centrality
│   ├── correlation_analysis.py     # Tương quan centrality
│   ├── random_comparison.py        # CHƯƠNG 3: So sánh random graphs
│   ├── community.py                # CHƯƠNG 4: Community detection
│   ├── community_comparison.py     # So sánh thuật toán
│   └── k_core_analysis.py          # K-core decomposition
│
├──  data/                         # CHƯƠNG 2: Tổng quan & Tiền xử lý
│   ├── load_data.py               # Load dataset
│   ├── collect_twitter.py         # Thu thập từ Twitter API
│   ├── preprocess.py              # Tiền xử lý dữ liệu
│   └── validate_data.py           # Kiểm tra chất lượng dataset
│
├──  visualization/               # Trực quan hóa
│   ├── plot_charts.py             # Vẽ biểu đồ
│   ├── draw_network.py            # Vẽ mạng
│   └── export_gephi.py            # Xuất file Gephi
│
├──  reporting/                   # BÁO CÁO & SLIDE
│   ├── generate_report.py         # Tạo báo cáo Word/PDF
│   ├── create_slides.py           # Tạo slide tự động
│   └── chapter_templates/         # Template cho từng chương
│
├── 🔧 utils/                       # Tiện ích
│   ├── check_installation.py      # Kiểm tra thư viện
│   └── config.py                  # Cấu hình parameters
│
└── 📁 results/                    # Kết quả đầu ra
    ├── 📁 charts/                 # Biểu đồ .png
    ├── 📁 data/                   # Dữ liệu .csv
    ├── 📁 reports/                # Báo cáo Word/PDF
    └── 📁 slides/                 # Slide thuyết trình
```

##  Hướng Dẫn Sử Dụng

### Chạy Toàn Bộ Phân Tích

```bash
python main.py
```

Hệ thống sẽ tự động chạy qua 7 bước phân tích:

1. **Data Loading & Preprocessing** - Tải và tiền xử lý dữ liệu
2. **Basic Network Analysis** - Phân tích tổng quan mạng
3. **Degree Distribution Analysis** - Phân tích phân phối bậc
4. **Centrality Analysis** - Tính toán độ đo trung tâm
5. **Correlation Analysis** - Phân tích tương quan
6. **Community Detection** - Phát hiện cộng đồng
7. **Visualization** - Tạo biểu đồ và báo cáo

### Chạy Từng Module Riêng Lẻ

| Module | Lệnh | Mô Tả |
|--------|------|-------|
| Phân tích cơ bản | `python analysis/basic_analysis.py` | Thống kê nodes, edges, density |
| Phân tích degree | `python analysis/degree_analysis.py` | Phân phối bậc và power-law |
| Phân tích centrality | `python analysis/centrality.py` | Độ đo trung tâm |
| Community detection | `python analysis/community.py` | Phát hiện cộng đồng |

##  Dataset

### Higgs Twitter Dataset

| Thông Tin | Giá Trị |
|-----------|---------|
| Tên dataset | Higgs Retweet Network |
| Nguồn | Stanford SNAP |
| Sự kiện | Khám phá Higgs Boson (7/2012) |
| Số lượng users | ~456,000 |
| Số lượng retweets | ~1.7 triệu |
| File | higgs-retweet_network.edgelist |

### Dataset Mẫu

Khi không tìm thấy dataset thật, hệ thống tự động tạo dataset mẫu:

| Thông Số | Giá Trị |
|----------|---------|
| Nodes | 300-500 |
| Edges | 3,000-5,000 |
| Loại đồ thị | Directed, small-world |

##  Kết Quả Phân Tích

### Các Độ Đo Được Tính Toán

| Loại Phân Tích | Chỉ Số | Mô Tả |
|----------------|--------|-------|
| **Basic Metrics** | Nodes, Edges | Số lượng node và cạnh |
| | Density | Mật độ mạng |
| | Clustering Coefficient | Độ tập trung cụm |
| **Degree Analysis** | Degree Distribution | Phân phối bậc |
| | Power-law Exponent | Hệ số gamma |
| **Centrality** | Degree, Betweenness | Độ đo trung tâm |
| | Closeness, PageRank | Ảnh hưởng lan tỏa |
| **Community** | Modularity | Chất lượng cộng đồng |
| | Number of Communities | Số lượng cộng đồng |

### Output Files

| Thư Mục | File | Mô Tả |
|---------|------|-------|
| `results/charts/` | `network_basic_analysis.png` | Biểu đồ phân tích cơ bản |
| | `degree_analysis_comprehensive.png` | Phân tích phân phối bậc |
| | `centrality_distributions.png` | Phân bố centrality |
| `results/data/` | `basic_metrics.csv` | Chỉ số cơ bản |
| | `centrality_all_nodes.csv` | Centrality tất cả nodes |

## 📚 Nội Dung Đồ Án

### Chương 1: Tổng Quan Đề Tài
- Giới thiệu mạng xã hội và tầm quan trọng
- Mục tiêu phân tích mạng retweet
- Phạm vi và đối tượng nghiên cứu

### Chương 2: Phân Tích Tổng Quan Mạng
- Thu thập và tiền xử lý dữ liệu
- Thống kê cơ bản (nodes, edges, density)
- Phân phối degree và power-law fitting

### Chương 3: Phân Tích Cấu Trúc Mạng
- Centrality measures (Degree, Betweenness, Closeness, PageRank)
- So sánh với random graph models
- Xác định key players/influencers

### Chương 4: Phân Tích Cộng Đồng
- Community detection (Louvain, Girvan-Newman)
- So sánh thuật toán phát hiện cộng đồng
- K-core decomposition

### Chương 5: Kết Luận Và Đánh Giá
- Tổng kết kết quả phân tích
- Đề xuất hướng phát triển

##  Kết Quả Đạt Được

- 7 modules phân tích chuyên sâu  
- Tự động hóa toàn bộ pipeline  
- Visualization đầy đủ biểu đồ  
- Xử lý lỗi robust  
- Export kết quả đa dạng format  

##  Hướng Phát Triển

- [ ] Phân tích động (temporal analysis)
- [ ] Machine learning để dự đoán link
- [ ] Visualization 3D với Gephi
- [ ] Web dashboard để trực quan hóa



##  Acknowledgments

- **Dataset:** Stanford SNAP
- **Libraries:** NetworkX, Pandas, Matplotlib
- **Inspiration:** Social Network Analysis course

---

<div align="center">



</div>

