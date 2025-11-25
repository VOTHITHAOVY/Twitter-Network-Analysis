import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

def calculate_correlations(centrality_dict):
    """Calculate correlations between different centrality measures"""
    # Tạo DataFrame từ centrality measures
    measures_df = pd.DataFrame(centrality_dict)
    
    # Tính ma trận tương quan
    pearson_corr = measures_df.corr(method='pearson')
    spearman_corr = measures_df.corr(method='spearman')
    
    return measures_df, pearson_corr, spearman_corr

def plot_correlation_heatmap(corr_matrix, title, save_path=None):
    """Plot correlation heatmap"""
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title(f'{title} Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved correlation heatmap: {save_path}")
    plt.close()

def plot_centrality_scatter(measures_df, save_path=None):
    """Plot scatter matrix of centrality measures"""
    # Chọn 4 measures chính để tránh quá nhiều subplots
    main_measures = ['degree', 'betweenness', 'closeness', 'pagerank']
    available_measures = [m for m in main_measures if m in measures_df.columns]
    
    if len(available_measures) >= 2:
        plot_df = measures_df[available_measures]
        
        # Tạo scatter matrix
        sns.pairplot(plot_df, diag_kind='hist', corner=True)
        plt.suptitle('Scatter Matrix of Centrality Measures', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved scatter matrix: {save_path}")
        plt.close()

def identify_similar_measures(corr_matrix, threshold=0.9):
    """Identify highly correlated centrality measures"""
    highly_correlated = []
    
    measures = corr_matrix.columns
    for i in range(len(measures)):
        for j in range(i+1, len(measures)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                highly_correlated.append({
                    'measure1': measures[i],
                    'measure2': measures[j], 
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    return highly_correlated

def main(config=None):
    """Main function for correlation analysis"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Lấy dữ liệu centrality từ config
    centrality = config.CENTRALITY
    
    if not centrality:
        print("❌ No centrality data found. Run centrality analysis first.")
        return None
    
    print("✅ Centrality data loaded for correlation analysis")
    
    # Tính toán tương quan
    measures_df, pearson_corr, spearman_corr = calculate_correlations(centrality)
    
    print("\n📊 CORRELATION ANALYSIS RESULTS:")
    print(f"Dataset shape: {measures_df.shape}")
    print(f"\nPearson Correlation Range: [{pearson_corr.values.min():.3f}, {pearson_corr.values.max():.3f}]")
    print(f"Spearman Correlation Range: [{spearman_corr.values.min():.3f}, {spearman_corr.values.max():.3f}]")
    
    # Xác định các measures tương quan cao
    high_corr = identify_similar_measures(pearson_corr, threshold=0.8)
    if high_corr:
        print("\n🔗 HIGHLY CORRELATED MEASURES (r ≥ 0.8):")
        for pair in high_corr:
            print(f"  {pair['measure1']} ↔ {pair['measure2']}: {pair['correlation']:.3f}")
    else:
        print("\n📈 No highly correlated measures found (all r < 0.8)")
    
    # Tạo visualizations
    plot_correlation_heatmap(
        pearson_corr, 
        'Pearson',
        save_path=config.RESULTS_DIR + "/charts/correlation_pearson.png"
    )
    
    plot_correlation_heatmap(
        spearman_corr,
        'Spearman', 
        save_path=config.RESULTS_DIR + "/charts/correlation_spearman.png"
    )
    
    plot_centrality_scatter(
        measures_df,
        save_path=config.RESULTS_DIR + "/charts/centrality_scatter_matrix.png"
    )
    
    # Lưu kết quả
    results = {
        'measures_df': measures_df,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'highly_correlated': high_corr
    }
    
    return results

if __name__ == "__main__":
    # Test với sample data
    config = Config()
    
    # Tạo sample centrality data nếu chưa có
    if not config.CENTRALITY:
        from data.load_data import load_sample_network
        from analysis.centrality import calculate_centrality_measures
        
        G = load_sample_network()
        centrality = calculate_centrality_measures(G)
        config.set_centrality(centrality)
    
    main(config)