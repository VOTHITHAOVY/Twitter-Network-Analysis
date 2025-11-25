import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

def analyze_degree_distribution(G):
    """Comprehensive degree distribution analysis"""
    print("Analyzing degree distribution...")
    
    analysis = {}
    
    if G.is_directed():
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        total_degrees = [in_deg + out_deg for in_deg, out_deg in zip(in_degrees, out_degrees)]
        
        analysis['in_degrees'] = in_degrees
        analysis['out_degrees'] = out_degrees
        analysis['total_degrees'] = total_degrees
        
        # Statistics
        analysis['in_stats'] = {
            'mean': np.mean(in_degrees),
            'median': np.median(in_degrees),
            'std': np.std(in_degrees),
            'max': np.max(in_degrees),
            'min': np.min(in_degrees)
        }
        
        analysis['out_stats'] = {
            'mean': np.mean(out_degrees),
            'median': np.median(out_degrees),
            'std': np.std(out_degrees),
            'max': np.max(out_degrees),
            'min': np.min(out_degrees)
        }
        
    else:
        degrees = [d for _, d in G.degree()]
        analysis['degrees'] = degrees
        analysis['stats'] = {
            'mean': np.mean(degrees),
            'median': np.median(degrees),
            'std': np.std(degrees),
            'max': np.max(degrees),
            'min': np.min(degrees)
        }
    
    return analysis

def fit_power_law(degrees, method='MLE'):
    """Fit power law to degree distribution using different methods"""
    
    if len(degrees) < 10:
        return {'gamma': 0, 'r_squared': 0, 'p_value': 1, 'method': method}
    
    # Remove zero degrees for log scaling
    degrees_nonzero = [d for d in degrees if d > 0]
    
    if method == 'linear_regression':
        # Method 1: Linear regression on log-log scale
        unique_degrees, counts = np.unique(degrees_nonzero, return_counts=True)
        pdf = counts / len(degrees_nonzero)
        
        # Log-transform
        log_degrees = np.log(unique_degrees[pdf > 0])
        log_pdf = np.log(pdf[pdf > 0])
        
        if len(log_degrees) < 2:
            return {'gamma': 0, 'r_squared': 0, 'p_value': 1, 'method': method}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_degrees, log_pdf)
        gamma = -slope  # p(k) ~ k^(-gamma)
        
        return {
            'gamma': gamma,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'method': method
        }
    
    elif method == 'MLE':
        # Method 2: Maximum Likelihood Estimation (more accurate)
        try:
            from powerlaw import fit
            fit_result = fit(degrees_nonzero, discrete=True)
            return {
                'gamma': fit_result.alpha,
                'xmin': fit_result.xmin,
                'sigma': fit_result.sigma,
                'method': 'MLE'
            }
        except ImportError:
            print("⚠️  powerlaw package not installed, using linear regression")
            return fit_power_law(degrees, method='linear_regression')

def plot_degree_analysis(analysis, G, save_dir):
    """Create comprehensive degree analysis plots"""
    
    plt.figure(figsize=(15, 12))
    
    if G.is_directed():
        in_degrees = analysis['in_degrees']
        out_degrees = analysis['out_degrees']
        total_degrees = analysis['total_degrees']
        
        # 1. In-Degree distribution
        plt.subplot(3, 3, 1)
        plt.hist(in_degrees, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('In-Degree')
        plt.ylabel('Frequency')
        plt.title('In-Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        # 2. Out-Degree distribution
        plt.subplot(3, 3, 2)
        plt.hist(out_degrees, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Out-Degree')
        plt.ylabel('Frequency')
        plt.title('Out-Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3. Total Degree distribution
        plt.subplot(3, 3, 3)
        plt.hist(total_degrees, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Total Degree')
        plt.ylabel('Frequency')
        plt.title('Total Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        degrees_for_fit = total_degrees
        
    else:
        degrees = analysis['degrees']
        
        plt.subplot(3, 3, 1)
        plt.hist(degrees, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        degrees_for_fit = degrees
    
    # 4. Log-log plot for power law
    plt.subplot(3, 3, 4)
    unique_degrees, counts = np.unique(degrees_for_fit, return_counts=True)
    pdf = counts / len(degrees_for_fit)
    
    plt.loglog(unique_degrees, pdf, 'bo', alpha=0.6, markersize=4, label='Data')
    
    # Fit and plot power law
    power_law_fit = fit_power_law(degrees_for_fit, method='linear_regression')
    if power_law_fit['r_squared'] > 0:
        x_fit = np.linspace(min(unique_degrees[unique_degrees > 0]), max(unique_degrees), 100)
        y_fit = np.exp(power_law_fit['intercept']) * x_fit ** (-power_law_fit['gamma'])
        plt.loglog(x_fit, y_fit, 'r-', linewidth=2, 
                  label=f'Power law fit (γ={power_law_fit["gamma"]:.2f})')
    
    plt.xlabel('Degree (log)')
    plt.ylabel('Probability (log)')
    plt.title('Degree Distribution (Log-Log)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. CCDF plot
    plt.subplot(3, 3, 5)
    sorted_degrees = np.sort(degrees_for_fit)
    ccdf = 1 - np.arange(len(sorted_degrees)) / len(sorted_degrees)
    
    plt.loglog(sorted_degrees, ccdf, 'g-', alpha=0.7, linewidth=2)
    plt.xlabel('Degree (log)')
    plt.ylabel('CCDF (log)')
    plt.title('Complementary CDF')
    plt.grid(True, alpha=0.3)
    
    # 6. Degree correlation scatter plot
    plt.subplot(3, 3, 6)
    if G.is_directed():
        plt.scatter(in_degrees, out_degrees, alpha=0.5, s=10)
        plt.xlabel('In-Degree')
        plt.ylabel('Out-Degree')
        plt.title('In-Degree vs Out-Degree')
    else:
        # For undirected, plot degree vs neighbor degree
        avg_neighbor_degree = nx.average_neighbor_degree(G)
        degrees = dict(G.degree())
        x = [degrees[node] for node in avg_neighbor_degree.keys()]
        y = list(avg_neighbor_degree.values())
        plt.scatter(x, y, alpha=0.5, s=10)
        plt.xlabel('Degree')
        plt.ylabel('Average Neighbor Degree')
        plt.title('Degree vs Neighbor Degree')
    plt.grid(True, alpha=0.3)
    
    # 7. Statistics table
    plt.subplot(3, 3, 7)
    plt.axis('off')
    
    stats_text = "DEGREE STATISTICS\n\n"
    if G.is_directed():
        stats_text += "IN-DEGREE:\n"
        stats_text += f"Mean: {analysis['in_stats']['mean']:.2f}\n"
        stats_text += f"Median: {analysis['in_stats']['median']:.1f}\n"
        stats_text += f"Std: {analysis['in_stats']['std']:.2f}\n"
        stats_text += f"Max: {analysis['in_stats']['max']}\n\n"
        
        stats_text += "OUT-DEGREE:\n"
        stats_text += f"Mean: {analysis['out_stats']['mean']:.2f}\n"
        stats_text += f"Median: {analysis['out_stats']['median']:.1f}\n"
        stats_text += f"Std: {analysis['out_stats']['std']:.2f}\n"
        stats_text += f"Max: {analysis['out_stats']['max']}\n"
    else:
        stats_text += "DEGREE:\n"
        stats_text += f"Mean: {analysis['stats']['mean']:.2f}\n"
        stats_text += f"Median: {analysis['stats']['median']:.1f}\n"
        stats_text += f"Std: {analysis['stats']['std']:.2f}\n"
        stats_text += f"Max: {analysis['stats']['max']}\n"
    
    if 'gamma' in power_law_fit:
        stats_text += f"\nPOWER LAW:\n"
        stats_text += f"Gamma: {power_law_fit['gamma']:.2f}\n"
        if 'r_squared' in power_law_fit:
            stats_text += f"R²: {power_law_fit['r_squared']:.3f}\n"
    
    plt.text(0.1, 0.9, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='top', transform=plt.gca().transAxes)
    
    # 8. Degree-rank plot (Zipf's law)
    plt.subplot(3, 3, 8)
    sorted_degrees = np.sort(degrees_for_fit)[::-1]  # Descending
    ranks = np.arange(1, len(sorted_degrees) + 1)
    
    plt.loglog(ranks, sorted_degrees, 'o-', alpha=0.7, markersize=3)
    plt.xlabel('Rank (log)')
    plt.ylabel('Degree (log)')
    plt.title('Degree-Rank Plot (Zipf)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "degree_analysis_comprehensive.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved comprehensive degree analysis plots")

def test_degree_distribution_hypothesis(analysis, G):
    """Test if degree distribution follows power law"""
    if G.is_directed():
        degrees = analysis['total_degrees']
    else:
        degrees = analysis['degrees']
    
    power_law_fit = fit_power_law(degrees)
    
    hypothesis_test = {
        'is_scale_free': power_law_fit.get('gamma', 0) > 2.0 and power_law_fit.get('r_squared', 0) > 0.8,
        'power_law_gamma': power_law_fit.get('gamma', 0),
        'fit_quality': power_law_fit.get('r_squared', 0),
        'method': power_law_fit.get('method', 'unknown')
    }
    
    return hypothesis_test

def main(config=None):
    """Main function for degree analysis"""
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("DEGREE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    G = config.get_network()
    
    if G is None:
        print("❌ No network data found. Run data loading first.")
        return None
    
    print(f"✅ Network loaded: {G.number_of_nodes()} nodes")
    
    # Analyze degree distribution
    analysis = analyze_degree_distribution(G)
    
    # Display results
    print("\n📊 DEGREE STATISTICS:")
    print("-" * 40)
    
    if G.is_directed():
        print("\nIN-DEGREE:")
        print(f"  Mean: {analysis['in_stats']['mean']:.2f}")
        print(f"  Median: {analysis['in_stats']['median']:.1f}")
        print(f"  Std: {analysis['in_stats']['std']:.2f}")
        print(f"  Max: {analysis['in_stats']['max']}")
        
        print("\nOUT-DEGREE:")
        print(f"  Mean: {analysis['out_stats']['mean']:.2f}")
        print(f"  Median: {analysis['out_stats']['median']:.1f}")
        print(f"  Std: {analysis['out_stats']['std']:.2f}")
        print(f"  Max: {analysis['out_stats']['max']}")
        
        degrees_for_fit = analysis['total_degrees']
    else:
        print("\nDEGREE:")
        print(f"  Mean: {analysis['stats']['mean']:.2f}")
        print(f"  Median: {analysis['stats']['median']:.1f}")
        print(f"  Std: {analysis['stats']['std']:.2f}")
        print(f"  Max: {analysis['stats']['max']}")
        
        degrees_for_fit = analysis['degrees']
    
    # Fit power law
    print("\n🔬 POWER LAW FITTING:")
    power_law_fit = fit_power_law(degrees_for_fit)
    
    print(f"  Method: {power_law_fit['method']}")
    print(f"  Gamma (exponent): {power_law_fit['gamma']:.4f}")
    
    if 'r_squared' in power_law_fit:
        print(f"  R-squared: {power_law_fit['r_squared']:.4f}")
        print(f"  P-value: {power_law_fit['p_value']:.4f}")
    
    if 'xmin' in power_law_fit:
        print(f"  x_min: {power_law_fit['xmin']}")
    
    # Test hypothesis
    hypothesis_test = test_degree_distribution_hypothesis(analysis, G)
    
    print("\n💡 HYPOTHESIS TESTING:")
    if hypothesis_test['is_scale_free']:
        print("  ✅ Network appears to be SCALE-FREE (power-law distribution)")
        print("  → Characteristic: Few hubs with many connections, many nodes with few connections")
    else:
        print("  ❌ Network does not strongly follow power-law distribution")
        print("  → May be random or have different degree distribution")
    
    # Create visualizations
    plots_dir = os.path.join(config.RESULTS_DIR, "charts")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_degree_analysis(analysis, G, plots_dir)
    
    # Save results
    results = {
        'analysis': analysis,
        'power_law_fit': power_law_fit,
        'hypothesis_test': hypothesis_test
    }
    
    config.set_degree_analysis(results)
    
    # Additional insights
    print("\n🎯 NETWORK CHARACTERISTICS:")
    if G.is_directed():
        in_out_ratio = analysis['in_stats']['mean'] / analysis['out_stats']['mean'] if analysis['out_stats']['mean'] > 0 else 0
        if in_out_ratio > 1.5:
            print("  • Information consumption network: More incoming than outgoing links")
        elif in_out_ratio < 0.67:
            print("  • Information production network: More outgoing than incoming links")
        else:
            print("  • Balanced network: Similar incoming and outgoing links")
    
    if hypothesis_test['power_law_gamma'] > 0:
        if hypothesis_test['power_law_gamma'] < 2:
            print("  • Very heterogeneous: Extreme hub-dominated structure")
        elif hypothesis_test['power_law_gamma'] < 3:
            print("  • Heterogeneous: Significant hub presence")
        else:
            print("  • Moderate heterogeneity: Some hubs present")
    
    return results

if __name__ == "__main__":
    config = Config()
    
    # Load sample network if not exists
    if config.get_network() is None:
        from data.load_data import load_sample_network
        G = load_sample_network()
        config.set_network(G)
    
    main(config)