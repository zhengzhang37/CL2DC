import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from models.model import *
import copy
class CLFwDFR(nn.Module):
    def __init__(self, num_classes=10, num_experts=3, hidden_dim=512, args=None, cfg=None):
        super(CLFwDFR, self).__init__()
        if cfg.dataset.name == 'chaoyang2u' or cfg.dataset.name == 'chaoyang3u':
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(512, 4)
            model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/chaoyang/best_ckpt.pth'))
        elif cfg.dataset.name == 'chestxray':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 2)
            model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/chestxray/best_ckpt.pth'))
        elif cfg.dataset.name == 'micebone':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(512, 3)
            model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/micebone/best_wo_pretrained_ckpt.pth'))
        elif cfg.dataset.name == 'ham10000':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 7)
            model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/ham10000/best_ckpt.pth'))
        elif cfg.dataset.name == 'galaxyzoo':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 2)
            model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/galaxyzoo/best_ckpt.pth'))
        elif cfg.dataset.name == 'cifair100':
            model = DualNet(100)
            model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/cifair100/best_ckpt.pth'))
        for param in model.parameters():
            param.requires_grad = False
        if cfg.dataset.name == 'cifair100':
            model.net1.linear = nn.Linear(512, num_classes + num_experts)
            model.net2.linear = nn.Linear(512, num_classes + num_experts)
        else:
            model.fc = nn.Linear(512, num_classes + num_experts)
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output

def load_models(args=None, cfg=None, device=None):
    if cfg.dataset.name == 'chaoyang2u' or cfg.dataset.name == 'chaoyang3u':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 4)
        model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/chaoyang/best_ckpt.pth'))
    elif cfg.dataset.name == 'chestxray':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/chestxray/best_ckpt.pth'))
    elif cfg.dataset.name == 'micebone':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 3)
        model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/micebone/best_wo_pretrained_ckpt.pth'))
    elif cfg.dataset.name == 'ham10000':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 7)
        model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/ham10000/best_ckpt.pth'))
    elif cfg.dataset.name == 'galaxyzoo':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/galaxyzoo/best_ckpt.pth'))
    elif cfg.dataset.name == 'cifair100':
        model = DualNet(100)
        model.load_state_dict(torch.load('/home/zheng/work/learningtodefer/pretrained_models/cifair100/best_ckpt.pth'))
    gating = copy.deepcopy(model)
    poissonregressor = copy.deepcopy(model)
    for param in model.parameters():
        param.requires_grad = False
    if cfg.dataset.name == 'cifair100':
        # model.net1.linear = nn.Linear(512, num_classes + num_experts)
        # model.net2.linear = nn.Linear(512, num_classes + num_experts)
        gating.net1.linear = nn.Linear(512, cfg.dataset.num_users + 1)
        gating.net2.linear = nn.Linear(512, cfg.dataset.num_users + 1)
        poissonregressor.net1.linear = nn.Sequential(
                                    nn.Linear(512, 1),
                                    nn.Softplus()
                                    )
        poissonregressor.net2.linear = nn.Sequential(
                                    nn.Linear(512, 1),
                                    nn.Softplus()
                                    )
    else:
        # model.fc = nn.Linear(512, num_classes + num_experts)
        gating.fc = nn.Linear(512, cfg.dataset.num_users + 1)
        poissonregressor.fc = nn.Sequential(
                            nn.Linear(512, 1),
                            nn.Softplus()
                            )

    return model.to(device=device), gating.to(device=device), poissonregressor.to(device=device)

        