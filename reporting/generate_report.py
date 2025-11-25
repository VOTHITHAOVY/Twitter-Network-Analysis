import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import pandas as pd
from datetime import datetime

def generate_text_report(config, output_path):
    """Tạo báo cáo văn bản tổng hợp"""
    print("Generating text report...")
    
    try:
        G = config.get_network()
        metrics = getattr(config, 'METRICS', {})
        centrality = getattr(config, 'CENTRALITY', {})
        communities = getattr(config, 'COMMUNITIES', {})
        degree_analysis = getattr(config, 'DEGREE_ANALYSIS', {})
    except:
        print("❌ No analysis data available")
        return
    
    report_content = f"""
SOCIAL NETWORK ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges
{'='*60}

1. BASIC NETWORK STATISTICS
{'='*30}
"""
    
    # Basic metrics
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                report_content += f"{key}: {value:.4f}\n"
            else:
                report_content += f"{key}: {value}\n"
    
    # Degree analysis
    if degree_analysis and 'statistics' in degree_analysis:
        report_content += f"\n2. DEGREE ANALYSIS\n{'='*30}\n"
        stats = degree_analysis['statistics']
        for key, value in stats.items():
            if isinstance(value, float):
                report_content += f"{key}: {value:.4f}\n"
            else:
                report_content += f"{key}: {value}\n"
    
    # Centrality analysis
    if centrality:
        report_content += f"\n3. CENTRALITY ANALYSIS\n{'='*30}\n"
        # Lấy top nodes cho mỗi measure
        for measure_name, values in centrality.items():
            if isinstance(values, dict) and len(values) > 0:
                top_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]
                report_content += f"\n{measure_name.upper()} - Top 3 Nodes:\n"
                for node, score in top_nodes:
                    report_content += f"  Node {node}: {score:.4f}\n"
    
    # Community analysis
    if communities and 'analysis' in communities:
        comm_analysis = communities['analysis']
        report_content += f"\n4. COMMUNITY ANALYSIS\n{'='*30}\n"
        report_content += f"Number of communities: {comm_analysis.get('num_communities', 'N/A')}\n"
        report_content += f"Modularity: {comm_analysis.get('modularity', 0):.4f}\n"
        report_content += f"Largest community: {comm_analysis.get('largest_community', 'N/A')} nodes\n"
    
    # Lưu report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Text report saved: {output_path}")
    return report_content

def generate_csv_reports(config, output_dir):
    """Tạo các báo cáo CSV"""
    print("Generating CSV reports...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic metrics report
    if hasattr(config, 'METRICS') and config.METRICS:
        metrics_df = pd.DataFrame([config.METRICS])
        metrics_df.to_csv(os.path.join(output_dir, "basic_metrics.csv"), index=False)
        print("✅ Basic metrics CSV saved")
    
    # 2. Centrality report
    if hasattr(config, 'CENTRALITY') and config.CENTRALITY:
        centrality_df = pd.DataFrame(config.CENTRALITY)
        centrality_df.to_csv(os.path.join(output_dir, "centrality_all_nodes.csv"), index=False)
        
        # Top nodes report
        top_nodes_data = []
        for measure, values in config.CENTRALITY.items():
            if isinstance(values, dict):
                top_10 = sorted(values.items(), key=lambda x: x[1], reverse=True)[:10]
                for rank, (node, score) in enumerate(top_10, 1):
                    top_nodes_data.append({
                        'measure': measure,
                        'rank': rank,
                        'node': node,
                        'score': score
                    })
        
        if top_nodes_data:
            top_nodes_df = pd.DataFrame(top_nodes_data)
            top_nodes_df.to_csv(os.path.join(output_dir, "centrality_top_nodes.csv"), index=False)
            print("✅ Centrality reports CSV saved")
    
    # 3. Community report
    if hasattr(config, 'COMMUNITIES') and config.COMMUNITIES:
        communities_data = []
        partition = config.COMMUNITIES.get('partition', {})
        
        for node, comm_id in partition.items():
            communities_data.append({
                'node': node,
                'community': comm_id
            })
        
        if communities_data:
            communities_df = pd.DataFrame(communities_data)
            communities_df.to_csv(os.path.join(output_dir, "community_membership.csv"), index=False)
            print("✅ Community report CSV saved")

def main(config=None):
    """Main function for report generation"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("REPORT GENERATION")
    print("="*60)
    
    # Tạo thư mục output
    reports_dir = os.path.join(config.RESULTS_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Tạo báo cáo văn bản
    text_report_path = os.path.join(reports_dir, "analysis_report.txt")
    generate_text_report(config, text_report_path)
    
    # Tạo báo cáo CSV
    csv_dir = os.path.join(reports_dir, "csv")
    generate_csv_reports(config, csv_dir)
    
    print(f"\n✅ All reports generated in: {reports_dir}")
    return True

if __name__ == "__main__":
    main()