import torch
import torch.nn as nn
import numpy as np
# from sklearn.utils.class_weight import compute_sample_weight

def compute_weights(y_true):
    y_np = y_true.cpu().detach().numpy().squeeze()
    labels = np.empty_like(y_np, dtype=int)
    thr1 = np.log1p(5)
    thr2 = np.log1p(25)
    thr3 = np.log1p(50)
    labels[y_np < thr1] = 1
    labels[(y_np >= thr1) & (y_np < thr2)] = 2
    labels[(y_np >= thr2) & (y_np < thr3)] = 3
    labels[y_np >= thr3] = 4
    # Extreme: 0.0040%, Heavy: 0.0173%, Moderate: 0.7135%, Weak: 99.2653%
    fixed_weights = {
        1: 1.0,  
        2: 10.0,
        3: 50.0,
        4: 100.0
    }
    sample_weights = np.empty_like(labels, dtype=float)
    for label, weight in fixed_weights.items():
        sample_weights[labels == label] = weight
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32, device=y_true.device)
    return sample_weights.unsqueeze(1)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        weights = compute_weights(y_true)
        # loss = torch.mean(torch.abs(y_true - y_pred))
        # print(f"y_pred.shape: {y_pred.shape}")
        # print(f"y_true.shape: {y_true.shape}")
        # print(f"weights.shape: {weights.shape}")
        # exit(0)
        # return loss
        return torch.sum(weights * torch.abs(y_true - y_pred)) / torch.sum(weights)