#!/usr/bin/env python3
"""
Download Higgs Twitter Dataset automatically
Run: python data/download_dataset.py
"""

import os
import requests
import gzip
import shutil
from tqdm import tqdm
from urllib.request import urlretrieve

def download_higgs_dataset():
    """Download Higgs Twitter Retweet Network dataset"""
    
    # URLs for Higgs dataset (chọn một trong các nguồn)
    datasets = {
        'retweet': {
            'url': 'http://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz',
            'filename': 'higgs-retweet_network.edgelist.gz',
            'size': '1.2GB'
        },
        'mention': {
            'url': 'http://snap.stanford.edu/data/higgs-mention_network.edgelist.gz', 
            'filename': 'higgs-mention_network.edgelist.gz',
            'size': '0.8GB'
        },
        'reply': {
            'url': 'http://snap.stanford.edu/data/higgs-reply_network.edgelist.gz',
            'filename': 'higgs-reply_network.edgelist.gz', 
            'size': '0.9GB'
        }
    }
    
    print("=== HIGGS TWITTER DATASET DOWNLOADER ===")
    print("Available datasets:")
    for i, (key, info) in enumerate(datasets.items(), 1):
        print(f"{i}. {key.upper()} Network - {info['size']}")
    
    choice = input("\nChoose dataset (1-3, default=1): ").strip()
    choices = {'1': 'retweet', '2': 'mention', '3': 'reply'}
    selected = choices.get(choice, 'retweet')
    
    dataset_info = datasets[selected]
    url = dataset_info['url']
    filename = dataset_info['filename']
    filepath = os.path.join('data', filename)
    
    print(f"\nDownloading {selected.upper()} network...")
    print(f"URL: {url}")
    print(f"Save to: {filepath}")
    
    # Tạo thư mục data nếu chưa tồn tại
    os.makedirs('data', exist_ok=True)
    
    try:
        # Download với progress bar
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f'\rProgress: {percent}%', end='', flush=True)
        
        urlretrieve(url, filepath, progress_hook)
        print(f"\nDownload completed: {filename}")
        
        # Giải nén file
        print("Extracting file...")
        extracted_path = filepath.replace('.gz', '')
        with gzip.open(filepath, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Extracted to: {extracted_path}")
        return extracted_path
        
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def create_sample_dataset():
    """Tạo dataset mẫu nếu download thất bại"""
    print("\nCreating sample dataset for testing...")
    
    import networkx as nx
    import random
    
    # Tạo đồ thị mẫu có cấu trúc giống mạng xã hội
    G = nx.DiGraph()
    
    # Thêm các node
    for i in range(1000):
        G.add_node(i)
    
    # Thêm edges với phân phối power-law
    for i in range(1000):
        # Số lượng connections theo phân phối power-law
        num_connections = int(100 * (random.random() ** -0.8))
        num_connections = min(num_connections, 100)  # Giới hạn tối đa
        
        for _ in range(num_connections):
            target = random.randint(0, 999)
            if target != i:
                G.add_edge(i, target)
    
    # Lưu thành file edgelist
    sample_path = 'data/sample_retweet_network.edgelist'
    nx.write_edgelist(G, sample_path, data=False)
    
    print(f"Sample dataset created: {sample_path}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    return sample_path

if __name__ == "__main__":
    # Thử download dataset thật
    dataset_path = download_higgs_dataset()
    
    # Nếu thất bại, tạo dataset mẫu
    if not dataset_path:
        print("\nFalling back to sample dataset...")
        dataset_path = create_sample_dataset()
    
    print(f"\nDataset ready: {dataset_path}")