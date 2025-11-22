import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

      
class LECODU(nn.Module):
    def __init__(self, y_dim=4, s_dim=3, hidden_dim=512):
        super(LECODU, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((s_dim+1) * y_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, y_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
class L2DC(nn.Module):
    def __init__(self, y_dim=4, s_dim=3, hidden_dim=512):
        """
        exp_dim: number of users
        y_dim: number of classes
        hidden_dim: 

        """
        super(L2DC, self).__init__()
        self.s_dim = s_dim 
        self.y_dim = y_dim
        self.flatten = nn.Flatten()
        self.collaboration = nn.Sequential(
            nn.Linear(y_dim * (self.s_dim+1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )
        
    def forward(self, predictions, is_train=True):
        B, N, K = predictions.shape

        hm_preds = []
        for i in range(N-1):
            preds = torch.zeros((B, self.s_dim+1, self.y_dim)).to(predictions.device)
            preds[:, i, :] = predictions[:, i, :]
            preds[:, -1, :] = predictions[:,-1,:]
            hm_preds.append(self.collaboration(self.flatten(preds)))
        collaboration = torch.stack(hm_preds, dim=1)
        return collaboration

if __name__ == '__main__':
    model = L2DC(4, 3)
    pred = torch.rand((256, 4, 4))
    print(model(pred).shape)




        
