# import os
import torch
# import random
from copy import deepcopy
# from tqdm.auto import tqdm
# import numpy as np

from utils import get_config_from_name, prepare_experiment_config,\
    reset_bn_stats, get_merging_fn
from model_merger import ModelMerge
from models import my_vgg_bn
from lmc_utils import interpolate_state_dicts, repair
from graphs.base_graph import NodeType
import argparse


def validate(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: \
        %d ' % (100 * correct / total))
    return loss_sum / total, correct / total


# python vgg16_bn_experiment.py --config cifar100_my_vgg16_bn --save_path cifar100_vgg16_bn_acc.pth
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cifar10_my_vgg16')
    parser.add_argument('--save_path', type=str, default='vgg16_bn_acc.pth')
    args = parser.parse_args()
    
    config_name = args.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)  # 返回config文件中的字典，添加了一个device键值对
    cur_config = deepcopy(raw_config)
    config = prepare_experiment_config(cur_config)

    train_loader = config['data']['train']['full']
    test_loader = config['data']['test']['full']

    base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
    Grapher = config['graph']

    criterion = torch.nn.CrossEntropyLoss()

    merging_fn_s = ['match_tensors_permute', 'match_tensors_zipit',
                    'match_tensors_identity']
    # get all prefix nodes
    graph = Grapher(deepcopy(base_models[0])).graphify().G
    prefix_nodes = []
    for node in graph.nodes:
        info = graph.nodes[node]
        if info['type'] == NodeType.PREFIX:
            prefix_nodes.append(node)
    res_dict = {merging_fn: {'merger': [],
                             'merger_reset': []} for merging_fn in
                merging_fn_s}

    for merging_fn in merging_fn_s:
        print(f"merging_fn: {merging_fn}")
        stop_at = None
        prefix_nodes = [None]+prefix_nodes
        for start_at in prefix_nodes:
            print(f"start_at: {start_at}")
            graphs = [Grapher(deepcopy(base_model)).graphify() for base_model
                      in base_models]
            Merge = ModelMerge(*graphs, device=device)
            Merge.transform(
                deepcopy(config['models']['new']), 
                train_loader, 
                transform_fn=get_merging_fn(merging_fn),
                metric_classes=config['metric_fns'],
                stop_at=stop_at,
                start_at=start_at
            )
            merger_acc = validate(Merge, test_loader, criterion, device)[1]
            print(f"merger_acc: {merger_acc}")

            # reset
            reset_bn_stats(Merge, train_loader)

            merger_reset_acc = validate(Merge, test_loader, criterion, device)[1]
            print(f"merger_reset_acc: {merger_reset_acc}")

            res_dict[merging_fn]['merger'].append(merger_acc)
            res_dict[merging_fn]['merger_reset'].append(merger_reset_acc)

    torch.save(res_dict, args.save_path)


if __name__ == '__main__':
    main()
