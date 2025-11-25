import os

class Config:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        
        # Create directories
        os.makedirs(os.path.join(self.RESULTS_DIR, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_DIR, "data"), exist_ok=True)
        
        self.NETWORK = None
        self.METRICS = {}
        self.CENTRALITY = {}
        self.COMMUNITIES = {}
        self.DEGREE_ANALYSIS = {}
        
    def set_network(self, G):
        self.NETWORK = G
        
    def get_network(self):
        return self.NETWORK
        
    def set_metrics(self, metrics):
        self.METRICS = metrics
        
    def set_centrality(self, centrality):
        self.CENTRALITY = centrality
        
    def set_communities(self, communities):
        self.COMMUNITIES = communities
        
    def set_degree_analysis(self, degree_analysis):
        self.DEGREE_ANALYSIS = degree_analysis