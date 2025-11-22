import json
import os
import sys
sys.path.append("../")
import logging
from pathlib import Path
import time
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
from torcheval import metrics
import copy
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import yaml
import math
from data_provider.chaoyang2u import *
from data_provider.chaoyang3u import *
from data_provider.chestxray import *
from data_provider.micebone import *
from data_provider.cifar100 import *
from models.model import DualNet
from types import SimpleNamespace
from omegaconf import OmegaConf
from models.mymodels import *
torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sklearn.metrics

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def create_model(args, cfg):
    if cfg.dataset.name == 'chaoyang2u' or cfg.dataset.name == 'chaoyang3u':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 4)
        model.load_state_dict(torch.load('../pretrained_models/chaoyang/best_ckpt.pth'))
    elif cfg.dataset.name == 'chestxray':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load('../pretrained_models/chestxray/best_ckpt.pth'))
    elif cfg.dataset.name == 'micebone':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 3)
        model.load_state_dict(torch.load('../pretrained_models/micebone/best_ckpt.pth'))
    elif cfg.dataset.name == 'cifair100':
        model = DualNet(100)
        model.load_state_dict(torch.load('../pretrained_models/cifair100/best_ckpt.pth'))
    for param in model.parameters():
        param.requires_grad = False
    if cfg.dataset.name == 'cifair100':
        dfr_net = copy.deepcopy(model)
        dfr_net.net1.linear = nn.Linear(512, 2 * cfg.dataset.num_users+1)
        dfr_net.net2.linear = nn.Linear(512, 2 * cfg.dataset.num_users+1)
    else:
        dfr_net = copy.deepcopy(model)
        dfr_net.fc = nn.Linear(512, 2 * cfg.dataset.num_users+1)
    clb_net = LECODU(y_dim=cfg.dataset.num_classes, s_dim=cfg.dataset.num_users, hidden_dim=512)
    return model.to(device=DEVICE), dfr_net.to(device=DEVICE), clb_net.to(device=DEVICE)

def smooth_labels(labels: torch.Tensor, alpha: float) -> torch.Tensor:
    """perform label smoothing

    Args:
        labels: one-hot labels
        alpha: smoothing factor (0 < alpha < 1)

    Returns:
        smoothed_labels:
    """
    num_classes = labels.shape[-1]
    return (1.0 - alpha) * labels + alpha / num_classes

def evaluate(
    dataset,
    gating: torch.nn.Module,
    clf: torch.nn.Module,
    clb: torch.nn.Module,
    cfg: DictConfig
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    Args:
        dataset:
        gating: the gating model
        clf: classifier
        cfg: configuration object

    Returns:

    """
    coverage = metrics.Mean(device=DEVICE)
    accuracy_accum = metrics.MulticlassAccuracy(
        num_classes=cfg.dataset.num_classes,
        device=DEVICE
    )
    # evaluate
    gating.eval()
    clf.eval()
    clb.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels, humans) in enumerate(dataset):
            x, y, t = inputs.to(device=DEVICE, non_blocking=True), labels.to(device=DEVICE, non_blocking=True), humans.to(device=DEVICE, non_blocking=True)
            # output of the gating model
            logits_p_z_x = gating(x=x)  # (batch, 2 * num_experts + 1)
            p_z_x = gumbel_softmax_sample(logits_p_z_x, 5)
            weighted_matrix = torch.zeros((cfg.dataset.num_users*2+1, cfg.dataset.num_users+1), dtype=torch.float32)
            weighted_matrix[:cfg.dataset.num_users+1,:] = torch.tril(torch.ones(cfg.dataset.num_users+1, cfg.dataset.num_users+1),diagonal=0)
            for i in range(cfg.dataset.num_users):
                weighted_matrix[i+cfg.dataset.num_users+1, 1:i+2] = 1
            weighted_matrix = weighted_matrix.to(device=DEVICE)
            # output of the classifier
            logits_p_t_x_clf = clf(x=x)  # (batch, num_classes)
            p_t_x_clf = gumbel_softmax_sample(logits_p_t_x_clf, 0.5)

            # annotations by human and classifier
            t = generate_experts(t, cfg.dataset.num_classes) # (batch, num_experts, num_classes)
            # smooth one-hot labels
            annotations = smooth_labels(labels=t, alpha=0.01)
            annotations = torch.concatenate(
                tensors=(p_t_x_clf[:, None, :], annotations[:, 0][:, None, :]),
                dim=1
            )  # (batch, num_experts + 1, num_classes)
            annotations = ((p_z_x @ weighted_matrix)).unsqueeze(-1) * annotations
            decision = clb(annotations) # (batch, num_experts, num_classes)
            
            coverage_flag = p_z_x[:, 0]
            coverage.update(coverage_flag)

            accuracy_accum.update(
                input=torch.argmax(input=decision, dim=1),
                target=y
            )
    return accuracy_accum.compute(), coverage.compute()

def generate_experts(humans, num_classes):
    experts = []
    b, n = humans.size()
    lst = list(range(n))
    random.shuffle(lst)
    sample_lst = lst
    expert_list = []
    for idx in sample_lst:
        expert_list.append(F.one_hot(humans[:, idx], num_classes=num_classes).float())
    experts = torch.cat([expert.unsqueeze(1) for expert in expert_list], dim=1)
    return experts

def main(cfg, args):
    """
    """
    cfg = OmegaConf.create(obj=cfg)
    OmegaConf.set_struct(conf=cfg, value=True)

    if cfg.dataset.name == 'chaoyang2u':
        loader = chaoyang2u_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    elif cfg.dataset.name == 'chaoyang3u':
        loader = chaoyang3u_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    elif cfg.dataset.name == 'chestxray':
        loader = chestxray_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers, args=cfg)
    elif cfg.dataset.name == 'micebone':
        loader = micebone_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    elif cfg.dataset.name == 'cifair100':
        loader = cifair100_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    train_loader = loader.run(mode='train')
    test_loader = loader.run(mode='test')

    # store dataset length to reweight prior
    with open_dict(config=cfg):
        cfg.dataset.length = cfg.dataset.num_users+1

    clf, gating, clb = create_model(args, cfg)
    
    optimiser = torch.optim.SGD(
        [
            {'params': gating.parameters()},
            {'params': clb.parameters()},
            # {'params': fp_encoder.parameters()}
        ],
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimiser, len(train_loader) * cfg.training.num_epochs, eta_min=0)
    # create a directory for storage (if not existed)
    logdir = os.path.join(cfg.experiment.logdir, cfg.dataset.name, 'coverage_'+str(args.coverage))
    modeldir = os.path.join(cfg.experiment.modeldir, cfg.dataset.name, 'coverage_'+str(args.coverage))
    if not os.path.exists(path=logdir):
        Path(logdir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path=modeldir):
        Path(modeldir).mkdir(parents=True, exist_ok=True)

    try:
        # tensorboard writer
        writer = SummaryWriter(
            log_dir=logdir,
            filename_suffix='learning_to_defer'
        )

        # TRAINING
        for epoch_id in tqdm(
            iterable=range(0, cfg.training.num_epochs, 1),
            desc='progress',
            ncols=80,
            leave=True,
            position=1,
            colour='green',
            disable=not cfg.training.progress_bar
        ):
            clf.eval()
            gating.train()
            clb.train()

            # batching and shuffling the dataset
            

            loss_accum = metrics.Mean(device=DEVICE)
            for batch_idx, (inputs, labels, humans) in enumerate(train_loader):
                x, y, t = inputs.to(device=DEVICE, non_blocking=True), labels.to(device=DEVICE, non_blocking=True), humans.to(device=DEVICE, non_blocking=True)

                with torch.autocast(device_type='cuda'):
                    # output of the gating model
                    logits_p_z_x = gating(x=x)  # (batch, 2 * num_experts + 1)
                    p_z_x = gumbel_softmax_sample(logits_p_z_x, 5)
                    weighted_matrix = torch.zeros((cfg.dataset.num_users*2+1, cfg.dataset.num_users+1), dtype=torch.float32)
                    weighted_matrix[:cfg.dataset.num_users+1,:] = torch.tril(torch.ones(cfg.dataset.num_users+1, cfg.dataset.num_users+1),diagonal=0)
                    for i in range(cfg.dataset.num_users):
                        weighted_matrix[i+cfg.dataset.num_users+1, 1:i+2] = 1
                    weighted_matrix = weighted_matrix.to(device=DEVICE)
                    # output of the classifier
                    logits_p_t_x_clf = clf(x=x)  # (batch, num_classes)
                    p_t_x_clf = gumbel_softmax_sample(logits_p_t_x_clf, 0.5)

                    # annotations by human and classifier
                    t = generate_experts(t, cfg.dataset.num_classes) # (batch, num_experts, num_classes)

                    # smooth one-hot labels
                    annotations = smooth_labels(labels=t, alpha=0.01)
                    annotations = torch.concatenate(
                        tensors=(p_t_x_clf[:, None, :], annotations[:, 0][:, None, :]),
                        dim=1
                    )  # (batch, num_experts + 1, num_classes)
                    annotations = (p_z_x @ weighted_matrix).unsqueeze(-1) * annotations
                    decision = clb(annotations) # (batch, num_experts, num_classes)
                    losses = F.cross_entropy(decision, y, reduction='none')
                    exp_coverage = p_z_x[:, 0].sum() / len(p_z_x) 
                    loss = losses.mean()
                    hyper_params = epoch_id + 1
                    loss = loss + hyper_params * args.h * (args.coverage - exp_coverage) ** 2

                    if torch.isnan(input=loss):
                        raise ValueError('Loss is NaN')

                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()
                    scheduler.step()
                    loss_accum.update(loss)
            
            # evaluation
            torch.save({
                        'gating_state_dict': gating.state_dict(), 
                        'clb_state_dict': clb.state_dict()
                        }
                        , os.path.join(modeldir, 'model.pth'))

            accuracy, coverage = evaluate(dataset=test_loader, gating=gating, clf=clf, clb=clb,cfg=cfg)
            table_header = "| Exp Coverage | Real Coverage | Accuracy |\n|------------|---------------|------------|\n"
            table_rows = [f"| {args.coverage} | {coverage} | {accuracy}|"]
            table_content = table_header + "\n".join(table_rows)

            writer.add_text("Experiment Results", table_content)
            
            # write to tensorboard
            writer.add_scalar(
                tag='loss',
                scalar_value=loss_accum.compute(),
                global_step=epoch_id + 1
            )
            writer.add_scalar(
                tag='accuracy',
                scalar_value=accuracy,
                global_step=epoch_id + 1
            )
            writer.add_scalar(
                tag='coverage',
                scalar_value=coverage,
                global_step=epoch_id + 1
            )
            
    finally:
        writer.close()

if __name__ == '__main__':
    # DEVICE = torch.device(device='cuda:1') if torch.cuda.is_available() else torch.device(device='cpu')
    
    parser = argparse.ArgumentParser(description='PyTorch Chaoyang Training')
    parser.add_argument('--coverage', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--h', type=int, default=0.5)
    args = parser.parse_args()
    cfgs = [
        # '../config/lecodu_chaoyang2u_conf.yaml', 
        # '../config/lecodu_chaoyang3u_conf.yaml', 
        '../config/lecodu_chestxray_conf.yaml', 
        '../config/lecodu_micebone_conf.yaml', 
        '../config/lecodu_cifair_conf.yaml'
        ]
    for cfg_path in cfgs:
        cfg = OmegaConf.load(cfg_path)
        DEVICE = torch.device(device='cuda:{:d}'.format(cfg.training.device)) if torch.cuda.is_available() else torch.device(device='cpu')
        coverage_list = [0.2, 0.4, 0.6, 0.8]

        for coverage in coverage_list:
            args.coverage = coverage
            main(cfg, args)
