import torch
import torch.nn as nn
from macro_dgrcl import MacroDGRCL
from train import compute_pairwise_ranking_loss, compute_log_scaled_mag_target

def trace_gradients():
    print("="*60)
    print("Masked Superset Gradient Trace")
    print("="*60)
    
    device = torch.device('cpu')
    torch.manual_seed(42)
    
    # Tiny dataset
    N_s = 5
    N_m = 3
    T = 10
    d_s = 8
    d_m = 4
    H = 32
    
    # Active mask: 3 active, 2 inactive
    active_mask = torch.tensor([True, True, True, False, False], device=device)
    
    model = MacroDGRCL(
        num_stocks=N_s,
        num_macros=N_m,
        stock_feature_dim=d_s,
        macro_feature_dim=d_m,
        hidden_dim=H
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    stock_feat = torch.randn(N_s, T, d_s, device=device)
    macro_feat = torch.randn(N_m, T, d_m, device=device)
    returns = torch.randn(N_s, device=device) * 0.05
    
    # --- Forward Pass ---
    print("\n[Forward Pass]")
    dir_logits, mag_preds = model(stock_feat, macro_feat, active_mask=active_mask)
    
    # --- Mask Verification check inside forward pass constraints ---
    print("\n[Safeguard Verification]")
    print(f"Logits for inactive stocks (should be strict 0.0):")
    print(dir_logits[~active_mask].detach().flatten().numpy())
    
    # --- Losses ---
    scores = dir_logits.squeeze(-1)
    loss_dir, _ = compute_pairwise_ranking_loss(scores, returns, active_mask=active_mask)
    
    mag_target = compute_log_scaled_mag_target(returns, active_mask=active_mask)
    loss_mag = nn.SmoothL1Loss()(mag_preds.squeeze(-1)[active_mask], mag_target[active_mask])
    
    total_loss = loss_dir + loss_mag
    print(f"\n[Losses]")
    print(f"Direction Loss: {loss_dir.item():.4f}")
    print(f"Magnitude Loss: {loss_mag.item():.4f}")
    print(f"Total Loss:     {total_loss.item():.4f}")
    
    # --- Backward Pass ---
    optimizer.zero_grad()
    total_loss.backward()
    
    # --- Gradient Trace ---
    print("\n[Gradient Trace - Norms per Component]")
    
    components = {
        "Temporal Encoder (Stock)": model.stock_encoder,
        "Temporal Encoder (Macro)": model.macro_encoder,
        "Stock LayerNorm": model.stock_embedding_norm,
        "Macro LayerNorm": model.macro_embedding_norm,
        "Dynamic Graph Learner": model.graph_learner,
        "Macro Propagation": model.mp_layers,
        "Multi-Task Head (Shared)": model.output_head.shared,
        "Multi-Task Head (Direction)": model.output_head.dir_head,
        "Multi-Task Head (Magnitude)": model.output_head.mag_head,
    }
    
    all_learning = True
    for name, module in components.items():
        grad_norms = []
        for p in module.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        
        if len(grad_norms) > 0:
            avg_grad = sum(grad_norms) / len(grad_norms)
            max_grad = max(grad_norms)
            status = "OK" if max_grad > 0 else "DEAD"
            if max_grad == 0:
                all_learning = False
            print(f"{name.ljust(30)} | Avg Grad: {avg_grad:.6f} | Max Grad: {max_grad:.6f} | [{status}]")
        else:
            print(f"{name.ljust(30)} | NO GRADIENTS FOUND")
            all_learning = False

    print("\n[Overall Learning Status]")
    if all_learning:
        print("SUCCESS: Gradients are flowing through all architectural components accurately.")
    else:
        print("FAIL: Gradient flow is broken in one or more components.")

if __name__ == "__main__":
    trace_gradients()
