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
from data_provider.ham10000 import *
from data_provider.galaxyzoo import *
from models.model import DualNet
from types import SimpleNamespace
from omegaconf import OmegaConf
from models.collaboration import *
torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sklearn.metrics
import pandas as pd
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
        model.load_state_dict(torch.load('../pretrained_models/micebone/best_wo_pretrained_ckpt.pth'))
    elif cfg.dataset.name == 'ham10000':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 7)
        model.load_state_dict(torch.load('../pretrained_models/ham10000/best_ckpt.pth'))
    elif cfg.dataset.name == 'galaxyzoo':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load('../pretrained_models/galaxyzoo/best_ckpt.pth'))
    elif cfg.dataset.name == 'cifair100':
        model = DualNet(100)
        model.load_state_dict(torch.load('../pretrained_models/cifair100/best_ckpt.pth'))
    # for param in model.parameters():
    #     param.requires_grad = False
    if cfg.dataset.name == 'cifair100':
        dfr_net = copy.deepcopy(model)
        dfr_net.net1.linear = nn.Linear(512, 2 * cfg.dataset.num_users+1)
        dfr_net.net2.linear = nn.Linear(512, 2 * cfg.dataset.num_users+1)
    else:
        dfr_net = copy.deepcopy(model)
        dfr_net.fc = nn.Linear(512, 2 * cfg.dataset.num_users+1)
    clb_net = L2DC(y_dim=cfg.dataset.num_classes, s_dim=cfg.dataset.num_users, hidden_dim=512)

    return model.to(device=DEVICE), dfr_net.to(device=DEVICE),  clb_net.to(device=DEVICE)

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
    fp_encoder: torch.nn.Module,
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
    selection_coverage = [metrics.Mean(device=DEVICE) for _ in range(2 * cfg.dataset.num_users + 1)]
    selection_accuracy = [metrics.MulticlassAccuracy(num_classes=cfg.dataset.num_classes, device=DEVICE) 
                        for _ in range(2 * cfg.dataset.num_users + 1)]
    # evaluate
    gating.eval()
    clf.eval()
    clb.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels, humans) in enumerate(dataset):
            # load data
            x, y, t = inputs.to(device=DEVICE), labels.to(device=DEVICE), humans.to(device=DEVICE)
            # gating
            logits_p_z_x = gating.forward(x=x)  # (batch, num_experts + 1)
            selected_expert_ids = torch.argmax(a=logits_p_z_x, dim=-1)  # (batch_size,)
            
            # coverage
            coverage_flag = (selected_expert_ids == cfg.dataset.num_users)
            coverage.update(coverage_flag)

            # classifier
            logits_clf = F.softmax(clf.forward(x=x), -1)  # (batch, num_classes)
            human_and_model_predictions = torch.concatenate(
                tensors=(
                    F.one_hot(input=t, num_classes=cfg.dataset.num_classes),
                    logits_clf[:, None, :]
                ),
                dim=1
            )
            decision = clb(fp_encoder(x), human_and_model_predictions)
            decision = torch.concatenate(
                        tensors=(human_and_model_predictions, decision),
                        dim=1
                    )
            queried_predictions = decision[torch.arange(len(x)), selected_expert_ids, :]
            accuracy_accum.update(
                input=torch.argmax(input=queried_predictions, dim=1),
                target=y
            )

            for expert_id in range(2 * cfg.dataset.num_users + 1):
                expert_mask = selected_expert_ids == expert_id
                if expert_mask.sum() > 0:
                    expert_preds = queried_predictions[expert_mask]
                    expert_targets = y[expert_mask]
                    selection_accuracy[expert_id].update(expert_preds, expert_targets)
            for expert_id in range(2 * cfg.dataset.num_users + 1):
                coverage_flag = (selected_expert_ids == expert_id)
                selection_coverage[expert_id].update(coverage_flag)
    return accuracy_accum.compute(), coverage.compute(), [selection_accuracy[i].compute() for i in range(len(selection_accuracy))], [selection_coverage[i].compute() for i in range(len(selection_coverage))]

def main(cfg, args):
    """
    """
    cfg = OmegaConf.create(obj=cfg)
    OmegaConf.set_struct(conf=cfg, value=True)

    if cfg.dataset.name == 'chaoyang2u':
        loader = chaoyang2u_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    elif cfg.dataset.name == 'chaoyang3u':
        loader = chaoyang3u_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    elif cfg.dataset.name == 'ham10000':
        loader = ham10000_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    elif cfg.dataset.name == 'chestxray':
        loader = chestxray_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers, args=cfg)
    elif cfg.dataset.name == 'micebone':
        loader = micebone_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
    elif cfg.dataset.name == 'cifair100':
        loader = cifair100_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers, args=cfg)
    elif cfg.dataset.name == 'galaxyzoo':
        loader = galaxyzoo_dataloader(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)
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
        ],
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimiser, len(train_loader) * cfg.training.num_epochs, eta_min=0)
    # create a directory for storage (if not existed)
    logdir = os.path.join(cfg.experiment.logdir, cfg.dataset.name + '_' + str(cfg.dataset.num_users) + 'experts', 'coverage_'+str(args.coverage))
    modeldir = os.path.join(cfg.experiment.modeldir, cfg.dataset.name + '_' + str(cfg.dataset.num_users) + 'experts', 'coverage_'+str(args.coverage))
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
                    p_z_x = F.softmax(input=logits_p_z_x, dim=-1)

                    # output of the classifier
                    logits_p_t_x_clf = clf(x=x)  # (batch, num_classes)
                    p_t_x_clf = F.softmax(input=logits_p_t_x_clf, dim=-1)

                    # annotations by human and classifier
                    t = F.one_hot(
                        input=t,
                        num_classes=cfg.dataset.num_classes
                    )  # (batch, num_experts, num_classes)

                    # smooth one-hot labels
                    annotations = smooth_labels(labels=t, alpha=0.01)
                    annotations = torch.concatenate(
                        tensors=(annotations, p_t_x_clf[:, None, :]),
                        dim=1
                    )  # (batch, num_experts + 1, num_classes)

                    decision = clb(annotations) # (batch, num_experts, num_classes)
                    decision = torch.concatenate(
                        tensors=(annotations, decision),
                        dim=1
                    )
                    losses = torch.stack([F.cross_entropy(input=decision[:, i, :], target=y, reduction='none') 
                                            for i in range(2 * cfg.dataset.num_users + 1)
                                        ], dim=1)
                    losses = p_z_x * losses

                    exp_coverage = p_z_x[:,cfg.dataset.num_users].sum() / len(p_z_x) 
                    loss = losses.mean()
                    if cfg.training.dual_coverage:
                        penalty_term = torch.relu(exp_coverage - args.coverage_upper) ** 2 + torch.relu(args.coverage_lower - exp_coverage) ** 2
                        loss = loss + (epoch_id + 1) * args.h * penalty_term
                    elif cfg.training.exact_coverage:
                        penalty_term = (args.coverage - exp_coverage) ** 2
                        loss = loss + (epoch_id + 1) * args.h * penalty_term
                    else:
                        penalty_term = torch.relu(args.coverage - exp_coverage) ** 2
                        loss = loss + (epoch_id + 1) * args.h * penalty_term

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

            accuracy, coverage, selection_accuracy, selection_coverage = evaluate(dataset=test_loader, gating=gating, clf=clf, fp_encoder=fp_encoder, clb=clb,cfg=cfg)
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
            writer.add_scalars(
                main_tag='selection_coverage',
                tag_scalar_dict={
                    **{'coverage' + f'{i}': selection_coverage[i] for i in range(len(selection_coverage))}
                },
                global_step=epoch_id + 1
            )
            writer.add_scalars(
                main_tag='selection_accuracy',
                tag_scalar_dict={
                    'system': accuracy,
                    **{'selection' + f'{i}': selection_accuracy[i] for i in range(len(selection_accuracy))}
                },
                global_step=epoch_id + 1
            )
            
    finally:
        writer.close()

def compute_deferral_metrics(data_test):
    """_summary_

    Args:
        data_test (dict): dict data with fields 'defers', 'labels', 'hum_preds', 'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on all data
    'human_all_acc': human accuracy on all data
    'coverage': how often classifier predicts

    """
    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )
    results["human_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"], data_test["labels"]
    )
    results["coverage"] = 1 - np.mean(data_test["defers"])
    # get classifier accuracy when defers is 0
    results["classifier_nondeferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"][data_test["defers"] == 0],
        data_test["labels"][data_test["defers"] == 0],
    )
    # get human accuracy when defers is 1
    results["human_deferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"][data_test["defers"] == 1],
        data_test["labels"][data_test["defers"] == 1],
    )
    # get system accuracy
    results["system_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"] * (1 - data_test["defers"])
        + data_test["hum_preds"] * (data_test["defers"]),
        data_test["labels"],
    )
    return results

def compute_coverage_v_acc_curve(data_test):
    """

    Args:
        data_test (dict): dict data with field   {'defers': defers_all, 'labels': truths_all, 'hum_preds': hum_preds_all, 'preds': predictions_all, 'rej_score': rej_score_all, 'class_probs': class_probs_all}

    Returns:
        data (list): compute_deferral_metrics(data_test_modified) on different coverage levels, first element of list is compute_deferral_metrics(data_test)
    """
    # get unique rejection scores
    rej_scores = np.unique(data_test["rej_score"])
    # sort by rejection score
    # get the 100 quantiles for rejection scores
    rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
    # for each quantile, get the coverage and accuracy by getting a new deferral decision
    all_metrics = []
    all_metrics.append(compute_deferral_metrics(data_test))
    for q in rej_scores_quantiles:
        # get deferral decision
        defers = (data_test["rej_score"] > q).astype(int)
        copy_data = copy.deepcopy(data_test)
        copy_data["defers"] = defers
        # compute metrics
        metrics = compute_deferral_metrics(copy_data)
        all_metrics.append(metrics)
    return all_metrics

def post_hoc(cfg, args):
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
    test_loader = loader.run(mode='test')
    clf, gating, fp_encoder, clb = create_model(args, cfg)
    clf.eval()
    gating.eval()
    fp_encoder.eval()
    clb.eval()
    modeldir = os.path.join(cfg.experiment.modeldir, cfg.dataset.name, 'coverage_'+str(args.coverage), 'model.pth')
    gating.load_state_dict(torch.load(modeldir)['gating_state_dict'])
    clb.load_state_dict(torch.load(modeldir)['clb_state_dict'])
    defers_all = []
    truths_all = []
    hum_preds_all = []
    predictions_all = []  # classifier only
    rej_score_all = []  # rejector probability
    class_probs_all = []  # classifier probability
    rej_score_all = []
    with torch.no_grad():
        for batch_idx, (inputs, label, humans) in enumerate(test_loader):
            # load data
            x, y, t = inputs.to(device=DEVICE), label.to(device=DEVICE), humans.to(device=DEVICE)
            # gating
            outputs_class = F.softmax(clf.forward(x=x), -1)
            human_and_model_predictions = torch.concatenate(
                tensors=(
                    F.one_hot(input=t, num_classes=cfg.dataset.num_classes),
                    outputs_class[:, None, :]
                ),
                dim=1
            )
            decision = clb(fp_encoder(x), human_and_model_predictions)
            # total prediction
            decision = torch.concatenate(
                        tensors=(F.one_hot(input=t, num_classes=cfg.dataset.num_classes), 
                                decision, outputs_class[:, None, :],
                        ),
                        dim=1
                    )
            decision = torch.argmax(decision, -1)
            # gating
            outputs = F.softmax(gating.forward(x=x), -1)  # (batch, num_classes + 1)
            _, predicted = torch.max(outputs.data, 1)
            # machine
            max_probs, predicted_class = torch.max(outputs_class, 1)
            predictions_all.extend(predicted_class.cpu().numpy())

            # experts
            parts = torch.cat((outputs[:, :cfg.dataset.num_users], outputs[:,cfg.dataset.num_users+1:]), dim=1)
            defer_and_fusion_probs = torch.cat([parts, outputs[:, cfg.dataset.num_users][:, None]], dim=1)
            _, predicted_experts = torch.max(defer_and_fusion_probs, 1)
            

            defer_scores = [defer_and_fusion_probs.data[i][predicted_experts[i]].item() - defer_and_fusion_probs.data[i][-1].item() for i in range(len(defer_and_fusion_probs.data))]
            defer_binary = [int(defer_score >= 0) for defer_score in defer_scores]
            
            defers_all.extend(defer_binary)
            truths_all.extend(y.cpu().numpy())
            human_preds = [decision[i][predicted_experts[i]].item() for i in range(len(defer_and_fusion_probs.data))]
            hum_preds_all.extend(human_preds)
            class_probs_all.extend(outputs_class.cpu().numpy())
            for i in range(len(defer_and_fusion_probs.data)):
                rej_score_all.append(
                    defer_and_fusion_probs.data[i][predicted_experts[i]].item()
                    - defer_and_fusion_probs.data[i][-1].item()
                )
    defers_all = np.array(defers_all)
    truths_all = np.array(truths_all)
    hum_preds_all = np.array(hum_preds_all)
    predictions_all = np.array(predictions_all)
    rej_score_all = np.array(rej_score_all)
    class_probs_all = np.array(class_probs_all)
    # print(rej_score_all)
    data = {
        "defers": defers_all,
        "labels": truths_all,
        "hum_preds": hum_preds_all,
        "preds": predictions_all,
        "rej_score": rej_score_all,
        "class_probs": class_probs_all,
    }    

    all_metrics = compute_coverage_v_acc_curve(data)
    with open("../data/cl2dc_cifair100_coverage_"+ str(args.coverage) + ".pkl", "wb") as f:
        pickle.dump(all_metrics, f)




if __name__ == '__main__':
    # DEVICE = torch.device(device='cuda:1') if torch.cuda.is_available() else torch.device(device='cpu')
    
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--coverage', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--h', type=int, default=0.5)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfgs = [
        # '../config/l2dc_chaoyang2u_conf.yaml', 
        # '../config/l2dc_chaoyang3u_conf.yaml', 
        # '../config/l2dc_chestxray_conf.yaml', 
        # '../config/l2dc_micebone_conf.yaml', 
        # '../config/l2dc_cifair_conf.yaml',
        # '../config/l2dc_ham10000_conf.yaml',
        '../config/l2dc_galaxyzoo_conf.yaml'
    ]
    for cfg_path in cfgs:
        cfg = OmegaConf.load(cfg_path)
        DEVICE = torch.device(device='cuda:{:d}'.format(cfg.training.device)) if torch.cuda.is_available() else torch.device(device='cpu')
        coverage_list = [0.2, 0.4, 0.6, 0.8, 0]
        for num_user in num_users:
            cfg.dataset.num_users = num_user
            for coverage in coverage_list:
                args.coverage = coverage
                main(cfg, args)