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
# import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import yaml
import math
from data_provider.chaoyang import *
from data_provider.chestxray import *
from types import SimpleNamespace
from omegaconf import OmegaConf
torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

def evaluate(
    dataset,
    gating: torch.nn.Module,
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
    selection_accuracy = [metrics.MulticlassAccuracy(num_classes=cfg.dataset.num_classes, device=DEVICE) 
                        for _ in range(cfg.dataset.num_users + 1)]
    # evaluate
    gating.eval()

    for batch_idx, (inputs, labels, humans) in enumerate(dataset):
        # load data
        x, y, t = inputs.to(device=DEVICE), labels.to(device=DEVICE), humans.to(device=DEVICE)

        # gating
        logits_p_z_x = gating.forward(x=x)  # (batch, num_classes + 1)
        selected_expert_ids = torch.argmax(a=logits_p_z_x, dim=-1)  # (batch_size,)
        selected_expert_ids[selected_expert_ids != cfg.dataset.num_classes] = 0
        selected_expert_ids[selected_expert_ids == cfg.dataset.num_classes] = 1
        
        # coverage
        coverage_flag = (selected_expert_ids != 1)
        coverage.update(coverage_flag)

        # classifier
        logits_clf = F.softmax(logits_p_z_x[:, 0:-1], -1)  # (batch, num_classes)
        human_and_model_predictions = torch.concatenate(
            tensors=(
                logits_clf[:, None, :],
                F.one_hot(input=t, num_classes=cfg.dataset.num_classes)[:, None, :]
            ),
            dim=1
        )

        # defer
        queried_predictions = human_and_model_predictions[torch.arange(len(x)), selected_expert_ids, :]
        accuracy_accum.update(
            input=torch.argmax(input=queried_predictions, dim=-1),
            target=y
        )

        for expert_id in range(cfg.dataset.num_users + 1):
            expert_mask = selected_expert_ids == expert_id
            if expert_mask.sum() > 0:
                expert_preds = queried_predictions[expert_mask]
                expert_targets = y[expert_mask]
                selection_accuracy[expert_id].update(expert_preds, expert_targets)
    
    return accuracy_accum.compute(), coverage.compute(), [selection_accuracy[i].compute() for i in range(len(selection_accuracy))]


def RS(output, label, expert):
    eps_cst = 1e-8
    alpha = 0.5
    output = output.to(device=DEVICE)
    label = label.to(device=DEVICE)
    expert_pred = expert
    expert_correctness = (expert_pred == label).float()

    batch_size = output.size()[0]  # batch_size
    outputs_exp = torch.exp(output)
    new_loss = -torch.log2(
        (
            expert_correctness * outputs_exp[range(batch_size), -1]
            + outputs_exp[range(batch_size), label]
        )
        / (torch.sum(outputs_exp, dim=1) + eps_cst)
    )  # pick the values corresponding to the labels
    ce_loss = -torch.log2(
        (outputs_exp[range(batch_size), label])
        / (torch.sum(outputs_exp[range(batch_size), :-1], dim=1) + eps_cst)
    )
    loss = alpha * new_loss + (1 - alpha) * ce_loss
    return torch.sum(loss) / batch_size

def create_model(args):
    if args.dataset.name == 'chaoyang':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 4)
        model.load_state_dict(torch.load('../pretrained_models/chaoyang/best_ckpt.pth'))
    elif args.dataset.name == 'chestxray':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load('../pretrained_models/chestxray/best_ckpt.pth'))
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, args.dataset.num_users+args.dataset.num_classes)
    return model.to(device=DEVICE)

def main(cfg, args):
    """
    """
    cfg = OmegaConf.create(obj=cfg)
    OmegaConf.set_struct(conf=cfg, value=True)

    if cfg.dataset.name == 'chaoyang':
        loader = chaoyang_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    elif cfg.dataset.name == 'chestxray':
        loader = chestxray_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers, args=cfg)
    train_loader = loader.run(mode='train')
    test_loader = loader.run(mode='test')
    
    with open_dict(config=cfg):
        cfg.dataset.length = cfg.dataset.num_users+1

    model = create_model(cfg)

    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9, 
        nesterov=True,
        weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimiser, len(train_loader) * 300, eta_min=0)
    # create a directory for storage (if not existed)
    logdir = os.path.join(cfg.experiment.logdir, cfg.dataset.name, 'coverage_'+str(args.coverage)+'_RS')
    modeldir = os.path.join(cfg.experiment.modeldir, cfg.dataset.name, 'coverage_'+str(args.coverage)+'_RS')
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
        model.train()
        for epoch_id in tqdm(
            iterable=range(0, cfg.training.num_epochs, 1),
            desc='progress',
            ncols=80,
            leave=True,
            position=1,
            colour='green',
            disable=not cfg.training.progress_bar
        ):
            loss_accum = metrics.Mean(device=DEVICE)
            for batch_idx, (inputs, labels, expert) in enumerate(train_loader):
                inputs = inputs.to(device=DEVICE)
                labels = labels.to(device=DEVICE)
                expert = expert.to(device=DEVICE)
                optimiser.zero_grad()
                output = model(inputs).to(device=DEVICE)
                exp_coverage = F.softmax(input=output, dim=-1)[:,:-1].mean()
                loss = RS(output=output, label=labels, expert=expert)
                hyper_params = epoch_id + 1
                loss = loss + hyper_params * args.h * (args.coverage - exp_coverage) ** 2
                loss.backward()
                optimiser.step()
                scheduler.step()
                loss_accum.update(loss)
            # evaluation
            accuracy, coverage, selection_accuracy = evaluate(dataset=test_loader, gating=model, cfg=cfg)
            torch.save(model.state_dict(), os.path.join(modeldir, 'model.pth'))
            table_header = "| Exp Coverage | Real Coverage | Accuracy |\n|------------|---------------|------------|\n"
            table_rows = [f"| {args.coverage} | {coverage} | {accuracy} |"]
            table_content = table_header + "\n".join(table_rows)

            # write to tensorboard

            writer.add_text("Experiment Results", table_content)
            
            writer.add_scalar(
                tag='loss',
                scalar_value=loss_accum.compute(),
                global_step=epoch_id + 1
            )

            writer.add_scalars(
                main_tag='accuracy',
                tag_scalar_dict={
                    'system': accuracy,
                    **{'selection' + f'{i}': selection_accuracy[i] for i in range(len(selection_accuracy))}
                },
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
    cfg = OmegaConf.load('../config/rs_chaoyang_conf.yaml')
    DEVICE = torch.device(device='cuda:{:d}'.format(cfg.training.device)) if torch.cuda.is_available() else torch.device(device='cpu')
    coverage_list = [0.2, 0.4, 0.6, 0.8]
    for coverage in coverage_list:
        args.coverage = coverage
        main(cfg, args)