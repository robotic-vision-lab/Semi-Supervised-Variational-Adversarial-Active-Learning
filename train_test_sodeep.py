from config import *
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import models.resnet as resnet
import torch.nn as nn
from sodeep import load_sorter, SpearmanLoss
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    # ###original implementation###
    # one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
    # if reduction == 'mean':
    #     loss = torch.sum(torch.clamp(margin - one * input, min=0))
    #     loss = loss / input.size(0)  # Note that the size of input is already halved
    # elif reduction == 'none':
    #     loss = torch.clamp(margin - one * input, min=0)
    # else:
    #     NotImplementedError()

    return loss


def test(models, epoch, method, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.to(device)
                labels = labels.to(device)

            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    # return 100 * correct / total
    return correct / total


def test_tsne(models, epoch, method, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'train'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    out_vec = torch.zeros(0)
    label = torch.zeros(0).long()
    with torch.no_grad():
        for (inputs, labels) in dataloaders:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.to(device)
                labels = labels.to(device)

            scores, _, _ = models['backbone'](inputs)
            preds = scores.cpu()
            labels = labels.cpu()
            out_vec = torch.cat([out_vec, preds])
            label = torch.cat([label, labels])
        out_vec = out_vec.numpy()
        label = label.numpy()
    return out_vec, label


iters = 0


def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, args):
    """Note for future exploration: separate the two loss backpropogation!!!"""

    models['backbone'].train()
    if method == 'lloss' or 'TA-VAAL':
        models['module'].train()
    global iters
    m_backbone_loss_sum, train_acc_sum, total = 0.0, 0.0, 0
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[0].to(device)
            labels = data[1].to(device)

        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss' or 'TA-VAAL':
            optimizers['module'].zero_grad()

        scores, _, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if method == 'lloss' or 'TA-VAAL':
            if epoch > epoch_loss:  # 120
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)  # nn.CrossEntropyLoss(reduction='none')
            m_backbone_loss_sum += m_backbone_loss.detach().cpu().item()

            # m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            # loss = m_backbone_loss + WEIGHT * m_module_loss
            # print(pred_loss.shape)
            # print(target_loss.shape)

            # ranking_criterion = SpearmanLoss("exa")
            # ranking_criterion = SpearmanLoss(*load_sorter('weights/best_model.pth.tar')).to(device)  # lstm_large, length: 100
            # ranking_criterion = SpearmanLoss(*load_sorter('weights/best_model_grup.pth.tar')).to(device)
            # ranking_criterion = SpearmanLoss(*load_sorter('weights/best_model_grup_L0029.pth.tar')).to(device)
            # ranking_criterion = SpearmanLoss(*load_sorter('weights/best_model_gruc.pth.tar')).to(device)
            ranking_criterion = SpearmanLoss(*load_sorter(args.weight_path)).to(device)
            ranking_loss = ranking_criterion(pred_loss, target_loss)
            # print(ranking_loss)
            loss = m_backbone_loss + WEIGHT * ranking_loss

        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss' or 'TA-VAAL':
            optimizers['module'].step()

        train_acc_sum += (scores.argmax(dim=1) == labels).sum().cpu().item()
        total += labels.shape[0]

    # return loss, pred_loss, target_loss
    return train_acc_sum / total, loss, m_backbone_loss_sum / len(dataloaders['train']), ranking_loss
    # return loss


def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, args):
    # print('>> Train a Model.')
    best_acc = 0.
    period_time, total_time = 0., 0.

    for epoch in range(num_epochs):
        start_time = time.time()
        best_loss = torch.tensor([0.5]).to(device)
        # running_loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, args)
        train_acc, loss, m_backbone_loss, ranking_loss = train_epoch(models, method, criterion, optimizers, dataloaders,
                                                                     epoch, epoch_loss, args)
        # running_loss, pred_loss, target_loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, args)

        schedulers['backbone'].step()
        if method == 'lloss' or 'TA-VAAL':
            schedulers['module'].step()

        # if epoch % 50 == 0:
        #     print('prediction loss shape: {}\t target loss shape: {}'.format(pred_loss.shape, target_loss.shape))

        # if False and epoch % 20 == 7:
        epoch_time = time.time() - start_time
        total_time += epoch_time
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            acc = test(models, epoch, method, dataloaders, mode='test')
            # acc = test(models, dataloaders, mc, 'test')
            if best_acc < acc:
                best_acc = acc
            print('Epoch: {}, Loss: {:.4f}, Backbone loss: {:.4f}, Ranking loss: {:.4f}, Training Acc: {:.2%},'
                  ' Test Acc: {:.2%}, Best Acc: {:.2%}, Time: {:.1f}s'.format(epoch + 1, loss, m_backbone_loss,
                                                                              ranking_loss, train_acc, acc, best_acc,
                                                                              total_time - period_time))
            period_time = total_time

    # print('>> Finished.')
