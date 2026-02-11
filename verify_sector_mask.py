import torch
import torch.nn as nn
import torch.nn.functional as F
from macro_dgrcl import DynamicGraphLearner


def test_sector_masking():
    print("Testing DynamicGraphLearner Sector Masking...")
    
    # 1. Setup Dummy Data
    N = 4  # 4 stocks
    Dim = 8 # Embedding dimension
    embeddings = torch.randn(N, Dim)
    
    # Define Sectors: 
    # Stock 0 & 1 are Sector A
    # Stock 2 & 3 are Sector B
    # Sector Mask [N, N]: True if same sector
    sector_mask = torch.tensor([
        [True,  True,  False, False], # 0
        [True,  True,  False, False], # 1
        [False, False, True,  True ], # 2
        [False, False, True,  True ]  # 3
    ])
    
    print("Sector Mask:\n", sector_mask)
    

    # 2. Initialize Graph Learner
    learner = DynamicGraphLearner(
        hidden_dim=Dim, # Real class takes hidden_dim
        top_k=3 # Try to connect to 3 neighbors
    )
    
    # 3. Forward Pass with Mask
    # We want to inspect the raw attention weights or resulting edges
    edge_index, _ = learner(
        embeddings, 
        sector_mask=sector_mask,
        return_weights=True
    )
    
    print("\nGenerated Edges (Src -> Dst):")
    reshaped_edges = edge_index.t()
    print(reshaped_edges)
    
    # 4. Verify No Cross-Sector Edges
    src, dst = edge_index
    
    valid_edges = 0
    invalid_edges = 0
    
    for i in range(edge_index.size(1)):
        s_idx = src[i].item()
        d_idx = dst[i].item()
        
        # Check if s and d are in same sector
        is_same_sector = sector_mask[s_idx, d_idx].item()
        
        if is_same_sector:
            valid_edges += 1
        else:
            invalid_edges += 1
            print(f"❌ INVALID EDGE FOUND: {s_idx} -> {d_idx}")
            
    print(f"\nSummary:")
    print(f"  Total Edges: {edge_index.size(1)}")
    print(f"  Valid Intra-Sector Edges: {valid_edges}")
    print(f"  Invalid Cross-Sector Edges: {invalid_edges}")
    
    assert invalid_edges == 0, "Found cross-sector edges! Masking failed."
    print("✅ TEST PASSED: All edges are strictly intra-sector.")

if __name__ == "__main__":
    test_sector_masking()
