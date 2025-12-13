import torch
import numpy as np
import matplotlib.pyplot as plt
import io

def get_saliency_map(model, signal_tensor, device):
    """
    Compute input gradients for the predicted class.
    signal_tensor: [1, 8, 1000]
    """
    model.eval()
    signal_tensor = signal_tensor.to(device)
    signal_tensor.requires_grad = True
    
    # Forward pass
    logits = model(signal_tensor)
    
    # Get top prediction
    probs = torch.sigmoid(logits)
    top_class_idx = torch.argmax(probs, dim=1).item()
    
    # Backward pass
    model.zero_grad()
    score = logits[0, top_class_idx]
    score.backward()
    
    # Get gradient
    saliency = signal_tensor.grad.data.abs() # [1, 8, 1000]
    saliency, _ = torch.max(saliency, dim=1) # [1, 1000] - Aggregate across channels? 
    # Or keep channel specific?
    # User requested: "Show 3 most important leads".
    # So we should keep channel dimension initially.
    
    saliency_per_lead = signal_tensor.grad.data.abs()[0] # [8, 1000]
    
    # Calculate importance per lead
    lead_importance = torch.sum(saliency_per_lead, dim=1) # [8]
    top_3_indices = torch.topk(lead_importance, 3).indices.cpu().numpy()
    
    return saliency_per_lead.cpu().numpy(), top_3_indices, top_class_idx

def plot_saliency(signal, saliency_map, top_leads, save_path=None):
    """
    Plot top 3 leads with saliency overlay.
    signal: [8, 1000] (numpy)
    saliency_map: [8, 1000] (numpy)
    top_leads: list of 3 indices
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # Assuming Hosp A ordering?
    # Wait, Hospital D uses DIFFERENT indices from raw 12-lead!
    # User indices: [0,2,3,4,5,6,7,11] -> I, III, aVR, aVL, aVF, V1, V2, V6
    hosp_d_lead_names = ['I', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V6']
    
    for i, lead_idx in enumerate(top_leads):
        ax = axes[i]
        sig = signal[lead_idx]
        sal = saliency_map[lead_idx]
        
        # Normalize saliency for alpha
        if sal.max() > 0:
            sal = sal / sal.max()
        
        ax.plot(sig, 'b-', label='ECG')
        # Overlay red dots where saliency is high
        # Or scatter plot
        # Using scatter for clear visibility
        high_sal_idx = np.where(sal > 0.3)[0] # Threshold
        ax.scatter(high_sal_idx, sig[high_sal_idx], c='r', s=10, alpha=sal[high_sal_idx]*0.8, label='Importance')
        
        ax.set_title(f"Lead {hosp_d_lead_names[lead_idx]}")
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        return save_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf
