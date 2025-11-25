import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from datetime import datetime

def create_slides_content(config):
    """Tạo nội dung slide thuyết trình"""
    print("Creating presentation slides content...")
    
    try:
        G = config.get_network()
        metrics = getattr(config, 'METRICS', {})
        centrality = getattr(config, 'CENTRALITY', {})
        communities = getattr(config, 'COMMUNITIES', {})
    except:
        print("❌ No analysis data available")
        return None
    
    slides_content = {
        'title_slide': {
            'title': 'PHÂN TÍCH MẠNG XÃ HỘI',
            'subtitle': 'Social Network Analysis Project',
            'team': 'Nhóm: [Tên nhóm]',
            'course': 'DS307.N11 - Phân tích Mạng Xã Hội',
            'date': datetime.now().strftime('%d/%m/%Y')
        },
        'introduction': {
            'title': 'Giới Thiệu Đề Tài',
            'content': [
                'Phân tích mạng xã hội sử dụng Python',
                'Dataset: Higgs Twitter Retweet Network',
                f'Quy mô: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
                'Mục tiêu: Hiểu cấu trúc và động lực mạng xã hội'
            ]
        },
        'methodology': {
            'title': 'Phương Pháp Phân Tích',
            'content': [
                '1. Phân tích tổng quan mạng',
                '2. Phân phối degree và power-law',
                '3. Độ đo trung tâm (Centrality)',
                '4. Phát hiện cộng đồng',
                '5. Trực quan hóa kết quả'
            ]
        },
        'results_basic': {
            'title': 'Kết Quả Phân Tích Cơ Bản',
            'content': [
                f"Mật độ mạng: {metrics.get('density', 0):.4f}",
                f"Độ tập trung cụm: {metrics.get('avg_clustering', 0):.3f}",
                f"Tính liên thông: {'Có' if metrics.get('is_connected', False) else 'Không'}",
                f"Tính có hướng: {'Có' if G.is_directed() else 'Không'}"
            ]
        },
        'results_centrality': {
            'title': 'Phân Tích Trung Tâm Mạng',
            'content': ['Các node quan trọng nhất:']
        },
        'results_communities': {
            'title': 'Phát Hiện Cộng Đồng',
            'content': []
        },
        'conclusion': {
            'title': 'Kết Luận & Hướng Phát Triển',
            'content': [
                'Đã phân tích thành công cấu trúc mạng xã hội',
                'Xác định được các node ảnh hưởng và cộng đồng',
                'Hệ thống có thể áp dụng cho các mạng khác',
                'Hướng phát triển: Phân tích động, Dự đoán liên kết'
            ]
        }
    }
    
    # Thêm thông tin centrality
    if centrality:
        top_nodes_info = []
        for measure in ['degree', 'betweenness', 'pagerank']:
            if measure in centrality:
                top_node = max(centrality[measure].items(), key=lambda x: x[1])
                top_nodes_info.append(f"Node {top_node[0]} ({measure}: {top_node[1]:.3f})")
        
        if top_nodes_info:
            slides_content['results_centrality']['content'].extend(top_nodes_info)
    
    # Thêm thông tin communities
    if communities and 'analysis' in communities:
        comm_analysis = communities['analysis']
        slides_content['results_communities']['content'] = [
            f"Số cộng đồng: {comm_analysis.get('num_communities', 'N/A')}",
            f"Modularity: {comm_analysis.get('modularity', 0):.3f}",
            f"Cộng đồng lớn nhất: {comm_analysis.get('largest_community', 'N/A')} nodes",
            f"Cộng đồng nhỏ nhất: {comm_analysis.get('smallest_community', 'N/A')} nodes"
        ]
    
    return slides_content

def generate_slides_markdown(slides_content, output_path):
    """Tạo slide dưới dạng Markdown"""
    print("Generating Markdown slides...")
    
    markdown_content = f"""# {slides_content['title_slide']['title']}

## {slides_content['title_slide']['subtitle']}

**{slides_content['title_slide']['team']}**

{slides_content['title_slide']['course']}

{slides_content['title_slide']['date']}

---

## {slides_content['introduction']['title']}

{"  \n".join(f"- {item}" for item in slides_content['introduction']['content'])}

---

## {slides_content['methodology']['title']}

{"  \n".join(slides_content['methodology']['content'])}

---

## {slides_content['results_basic']['title']}

{"  \n".join(f"- {item}" for item in slides_content['results_basic']['content'])}

---

## {slides_content['results_centrality']['title']}

{"  \n".join(f"- {item}" for item in slides_content['results_centrality']['content'])}

---

## {slides_content['results_communities']['title']}

{"  \n".join(f"- {item}" for item in slides_content['results_communities']['content'])}

---

## {slides_content['conclusion']['title']}

{"  \n".join(f"- {item}" for item in slides_content['conclusion']['content'])}

---

# Cảm ơn!

**Questions?**
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"✅ Markdown slides saved: {output_path}")
    return markdown_content

def main(config=None):
    """Main function for slide creation"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("SLIDE CREATION")
    print("="*60)
    
    # Tạo nội dung slide
    slides_content = create_slides_content(config)
    
    if not slides_content:
        print("❌ Cannot create slides without analysis data")
        return None
    
    # Tạo thư mục output
    slides_dir = os.path.join(config.RESULTS_DIR, "slides")
    os.makedirs(slides_dir, exist_ok=True)
    
    # Tạo slide markdown
    md_path = os.path.join(slides_dir, "presentation.md")
    generate_slides_markdown(slides_content, md_path)
    
    # Tạo file hướng dẫn
    readme_content = """
PRESENTATION SLIDES

Files:
- presentation.md: Slide content in Markdown format

How to use:
1. Copy content to PowerPoint or Google Slides
2. Or use Marp (https://marp.app/) to convert to PDF/HTML
3. Or use with reveal.js for web presentation

Each '---' separator indicates a new slide.
"""
    
    with open(os.path.join(slides_dir, "README.txt"), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Presentation slides created in: {slides_dir}")
    print("📊 You can now use these slides for your presentation!")
    
    return slides_content

if __name__ == "__main__":
    main()