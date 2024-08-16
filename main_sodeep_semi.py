# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
from sklearn.cluster import KMeans
# Custom
import models.resnet as resnet
# from models.resnet import vgg11
from models.query_models import LossNet
from train_test_sodeep import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *
import time

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lambda_loss", type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s", "--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n", "--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r", "--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d", "--dataset", type=str, default="cifar10im",
                    help="")
parser.add_argument("-e", "--no_of_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-m", "--method_type", type=str, default="TA-VAAL",
                    help="")
parser.add_argument("-tr", "--trials", type=int, default=5,
                    help="Number of trials")
parser.add_argument("-c", "--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t", "--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-dp", "--data_path", type=str, default='./data', help='Path to where the data is')
parser.add_argument("-wp", "--weight_path", type=str, default='./data',
                    help='Path to where the weight of ranking model is')
parser.add_argument("-bs", "--batch_size", type=int, default='128', help='Batch size')
parser.add_argument('--num_semi_epochs', type=int, default=100,
                    help='Number of epochs for semi-supervised training')

args = parser.parse_args()


def alpha_weight(epoch, T1, T2, af):
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
        return ((epoch - T1) / (T2 - T1)) * af


### Main ###
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = args.method_type
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss', 'VAAL', 'TA-VAAL']
    datasets = ['cifar10', 'cifar10im', 'cifar100', 'fashionmnist', 'svhn']
    assert method in methods, 'No method %s! Try options %s' % (method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s' % (args.dataset, datasets)
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL'
    '''
    results = open(
        'results_' + str(args.method_type) + "_" + "sodeep_semi_" + args.dataset + '_main' + str(args.cycles) + str(
            args.total) + '.txt',
        'w')
    print("Dataset: %s" % args.dataset)
    print("Method type:%s" % method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        TRIALS = args.trials
        CYCLES = args.cycles

    for trial in range(TRIALS):
        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset, args.data_path)
        print('The size of the entire dataset is {}'.format(len(data_train)))
        ADDENDUM = adden
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        if args.total:
            labeled_set = indices
        else:
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]  # unlabeled_set = indices[ADDENDUM:]

        # train_loader = DataLoader(data_train, batch_size=BATCH,
        #                           sampler=SubsetRandomSampler(labeled_set),
        #                           pin_memory=True, drop_last=True)
        # test_loader = DataLoader(data_test, batch_size=BATCH)

        train_loader = DataLoader(data_train, batch_size=args.batch_size,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True, drop_last=True)
        unlabeled_loader = DataLoader(data_train, batch_size=args.batch_size,
                                      sampler=SubsetRandomSampler(unlabeled_set),
                                      pin_memory=True, drop_last=True)
        test_loader = DataLoader(data_test, batch_size=args.batch_size)

        dataloaders = {'train': train_loader, 'test': test_loader}

        for cycle in range(CYCLES):
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]  # SUBSET = 10000
            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                if args.dataset == "fashionmnist":
                    resnet18 = resnet.ResNet18fm(num_classes=NO_CLASSES).to(device)
                else:
                    # resnet18    = vgg11().to(device)
                    resnet18 = resnet.ResNet18(num_classes=NO_CLASSES).to(device)
                if method == 'lloss' or 'TA-VAAL':
                    # loss_module = LossNet(feature_sizes=[16,8,4,2], num_channels=[128,128,256,512]).to(device)
                    loss_module = LossNet().to(device)

            models = {'backbone': resnet18}  # shouldn't this be defined outside cycles???
            if method == 'lloss' or 'TA-VAAL':
                models = {'backbone': resnet18, 'module': loss_module}
            torch.backends.cudnn.benchmark = True

            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')  # need a vector of losses instead of a scalar
            # criterion = nn.CrossEntropyLoss()

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            if method == 'lloss' or 'TA-VAAL':
                optim_module = optim.SGD(models['module'].parameters(), lr=LR,
                                         momentum=MOMENTUM, weight_decay=WDECAY)
                sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and testing
            print('Trial {}/{} || Cycle {}/{} || Label set size {}:'.format(trial + 1, TRIALS, cycle + 1,
                                                                            CYCLES, len(labeled_set)))
            training_epochs = args.no_of_epochs + cycle * 20
            train(models, method, criterion, optimizers, schedulers, dataloaders, training_epochs, EPOCHL, args)
            # train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL, args)

            """KMeans clustering"""
            test_features_cpu = np.load("saved_models/test_features_cpu_0621.npy", allow_pickle=True)
            kmeans = KMeans(n_clusters=10, random_state=0).fit(test_features_cpu)
            confusion_matrix = np.zeros((10, 10), dtype=np.int32)
            for cluster_label, true_label in zip(kmeans.labels_, data_test.targets):
                confusion_matrix[true_label, cluster_label] += 1
            cluster_to_class = {}  # {0: 5, 1: 2, 2: 8, 3: 7, 4: 9, 5: 0, 6: 4, 7: 1, 8: 3, 9: 6}
            for j in range(10):
                true_label = np.argmax(confusion_matrix[:, j])
                cluster_to_class[j] = true_label

            """semi-supervised training with pseudo-labeling"""
            print('Semi-supervised training for another %d epochs...' % args.num_semi_epochs)
            step, T1, T2, alpha = 100, 100, 7500, 3
            threshold = 0.95
            print('threshold =', threshold)

            # model.train()
            for epoch in range(args.num_semi_epochs):
                start = time.time()
                unlabeled_loss_sum, unlabeled_acc_sum, unlabeled_total = 0.0, 0.0, 0
                labeled_loss_sum, labeled_acc_sum, labeled_total = 0.0, 0.0, 0
                unlabeled_count = 1

                for unlb_batch_idx, (unlabeled_inputs, _) in enumerate(unlabeled_loader):
                    # Forward Pass to get the pseudo labels
                    unlabeled_inputs = unlabeled_inputs.to(device)

                    optimizers['backbone'].zero_grad()

                    # Forward pass for unlabeled data

                    models['backbone'].eval()
                    with torch.no_grad():
                        outputs_unlabeled, _, _ = models['backbone'](unlabeled_inputs)
                        # print(len(outputs_unlabeled))
                        # print(outputs_unlabeled[0].shape)
                        # break

                        clustering = kmeans.predict(outputs_unlabeled.cpu().numpy())
                        kmeans_mapping = [cluster_to_class[cluster_label] for cluster_label in clustering]

                        _, pseudo_labels = torch.max(outputs_unlabeled, 1)  # [0]:values, [1]:indices
                        probs = torch.softmax(outputs_unlabeled, dim=1)
                        max_probs, _ = torch.max(probs, 1)
                        mask_cluster = (torch.tensor(kmeans_mapping).to(device) == pseudo_labels)
                        mask_pseudo = max_probs.ge(threshold)
                        mask = mask_pseudo & mask_cluster

                    models['backbone'].train()
                    outputs_unlabeled, _, _ = models['backbone'](unlabeled_inputs)
                    # loss_unlabeled = criterion(outputs_unlabeled, pseudo_labels) * mask
                    loss_unlabeled = criterion(outputs_unlabeled, pseudo_labels)
                    # loss_unlabeled = loss_unlabeled.sum() / loss_unlabeled.shape[0]  # = mean
                    loss_unlabeled = loss_unlabeled.mean()
                    # print(loss_unlabeled.shape)
                    # print(loss_unlabeled)
                    # print(loss_unlabeled2)
                    # break

                    loss_unlabeled = loss_unlabeled * mask
                    # print(loss_unlabeled.shape)
                    # break
                    loss_unlabeled = alpha_weight(step, T1, T2, alpha) * loss_unlabeled
                    loss_unlabeled = loss_unlabeled.mean()

                    unlabeled_loss_sum += loss_unlabeled.detach().cpu().item()

                    loss_unlabeled.backward()
                    optimizers['backbone'].step()

                    # For every 50 batches train one epoch on labeled data
                    if unlb_batch_idx % 50 == 0:

                        # Normal training procedure on labeled data
                        for batch_idx, (inputs, labels) in enumerate(train_loader):
                            inputs, labels = inputs.to(device), labels.to(device)

                            output, _, _ = models['backbone'](inputs)
                            labeled_loss = criterion(output, labels)
                            labeled_loss = labeled_loss.mean()

                            optimizers['backbone'].zero_grad()
                            labeled_loss.backward()
                            optimizers['backbone'].step()

                            labeled_loss_sum += labeled_loss.detach().cpu().item()
                            # running_loss += labeled_loss.item()

                        # Now we increment step by 1
                        step += 1

                acc = test(models, EPOCH, method, dataloaders, mode='test')
                schedulers['backbone'].step()
                if (epoch + 1) % 10 == 0:
                    print('E {}: α: {:.3f} | Ub Loss: {:.4f} | Lb Loss: {:.4f} | Test Acc: {:.4f} | Time: {:.1f}s'.format(
                        epoch + 1, alpha_weight(step, T1, T2, alpha), unlabeled_loss_sum / len(unlabeled_loader),
                        labeled_loss_sum / len(train_loader), acc, (time.time() - start) * 5))

            acc = test(models, EPOCH, method, dataloaders, mode='test')
            # print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
            #                                                                             CYCLES, len(labeled_set), acc))
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            np.array([method, trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")

            if cycle == (CYCLES - 1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)

            # Update the labeled dataset and the unlabeled dataset, respectively
            new_list = list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
            # print(len(new_list), min(new_list), max(new_list))
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy())
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=args.batch_size,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
            # dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
            #                                   sampler=SubsetRandomSampler(labeled_set),
            #                                   pin_memory=True)

    results.close()
