#!/usr/bin/env python3
"""
MAIN EXECUTION SCRIPT FOR SOCIAL NETWORK ANALYSIS
Run: python main.py
"""

import sys
import os
import time
import importlib
from tqdm import tqdm

# Thêm đường dẫn để import các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'networkx', 'pandas', 'numpy', 'matplotlib', 
        'scipy', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are satisfied!")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'results/charts',
        'results/data', 
        'results/reports',
        'results/slides'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")

class Config:
    """Configuration class for the project"""
    
    def __init__(self):
        # Directory paths
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.ANALYSIS_DIR = os.path.join(self.BASE_DIR, "analysis") 
        self.VISUALIZATION_DIR = os.path.join(self.BASE_DIR, "visualization")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        self.REPORTING_DIR = os.path.join(self.BASE_DIR, "reporting")
        self.UTILS_DIR = os.path.join(self.BASE_DIR, "utils")
        
        # Analysis parameters
        self.NETWORK = None
        self.METRICS = {}
        self.CENTRALITY = {}
        self.COMMUNITIES = {}
        self.DEGREE_ANALYSIS = {}
        
    def set_network(self, G):
        """Store the network graph"""
        self.NETWORK = G
        print(f"✅ Network stored: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
    def get_network(self):
        """Retrieve the network graph"""
        if self.NETWORK is None:
            print("❌ Network not loaded. Run data loading first.")
            return None
        return self.NETWORK
        
    def set_metrics(self, metrics):
        """Store network metrics"""
        self.METRICS = metrics
        print("✅ Basic metrics calculated and stored")
        
    def set_centrality(self, centrality):
        """Store centrality measures"""
        self.CENTRALITY = centrality
        print("✅ Centrality measures calculated and stored")
        
    def set_communities(self, communities):
        """Store community detection results"""
        self.COMMUNITIES = communities
        print("✅ Community detection completed and stored")
        
    def set_degree_analysis(self, degree_analysis):
        """Store degree analysis results"""
        self.DEGREE_ANALYSIS = degree_analysis
        print("✅ Degree analysis completed and stored")

def run_module(module_name, config, step_name):
    """Run a specific module with error handling"""
    try:
        print(f"\n{'='*60}")
        print(f"🚀 {step_name}")
        print(f"📦 Module: {module_name}")
        print('='*60)
        
        # Import module
        module = importlib.import_module(module_name)
        
        # Check if module has main function
        if hasattr(module, 'main'):
            result = module.main(config)
            print(f"✅ {step_name} completed successfully!")
            return result
        else:
            print(f"⚠️  Module {module_name} has no 'main' function, skipping...")
            return None
            
    except Exception as e:
        print(f"❌ Error in {step_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🎯 SOCIAL NETWORK ANALYSIS PROJECT")
    print("="*70)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup directories
    print("\n📂 Setting up directories...")
    setup_directories()
    
    # Initialize configuration
    config = Config()
    
    # Define execution pipeline
    pipeline = [
        {
            "name": "1. Data Loading & Preprocessing",
            "module": "data.load_data",
            "description": "Load network dataset and perform initial preprocessing"
        },
        {
            "name": "2. Basic Network Analysis", 
            "module": "analysis.basic_analysis",
            "description": "Calculate basic network metrics and properties"
        },
        {
            "name": "3. Degree Distribution Analysis",
            "module": "analysis.degree_analysis", 
            "description": "Analyze degree distribution and fit power law"
        },
        {
            "name": "4. Centrality Analysis",
            "module": "analysis.centrality",
            "description": "Calculate various centrality measures"
        },
        {
            "name": "5. Correlation Analysis",
            "module": "analysis.correlation_analysis",
            "description": "Analyze correlations between centrality measures"
        },
        {
            "name": "6. Community Detection",
            "module": "analysis.community", 
            "description": "Detect communities using various algorithms"
        },
        {
            "name": "7. Visualization",
            "module": "visualization.plot_charts",
            "description": "Generate charts and network visualizations"
        }
    ]
    
    # Execute pipeline
    print(f"\n📊 Starting analysis pipeline with {len(pipeline)} steps...")
    
    results = {}
    for step in tqdm(pipeline, desc="Analysis Progress"):
        result = run_module(step["module"], config, step["name"])
        results[step["module"]] = result
        time.sleep(1)  # Small delay for better visualization
    
    # Generate final report
    print(f"\n{'='*60}")
    print("📄 GENERATING FINAL REPORT")
    print('='*60)
    
    try:
        reporting_module = importlib.import_module("reporting.generate_report")
        if hasattr(reporting_module, 'main'):
            reporting_module.main(config)
        print("✅ Final report generated!")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print('='*70)
    
    # Display summary statistics
    if config.METRICS:
        print("\n📈 SUMMARY STATISTICS:")
        print(f"   • Nodes: {config.METRICS.get('nodes', 'N/A')}")
        print(f"   • Edges: {config.METRICS.get('edges', 'N/A')}")
        print(f"   • Density: {config.METRICS.get('density', 'N/A'):.6f}")
        print(f"   • Average Clustering: {config.METRICS.get('avg_clustering', 'N/A'):.4f}")
    
    if config.DEGREE_ANALYSIS and 'power_law' in config.DEGREE_ANALYSIS:
        pl = config.DEGREE_ANALYSIS['power_law']
        print(f"   • Power Law Gamma: {pl.get('gamma', 'N/A'):.4f}")
    
    print(f"\n📁 Results saved in: {config.RESULTS_DIR}")
    print("✨ Project completed! Check the results/ directory for outputs.")

if __name__ == "__main__":
    main()