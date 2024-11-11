# import os
import torch
# import random
from copy import deepcopy
# from tqdm.auto import tqdm
# import numpy as np

from utils import get_config_from_name, prepare_experiment_config,\
    reset_bn_stats, get_merging_fn
from model_merger import ModelMerge
from models import my_vgg
from lmc_utils import interpolate_state_dicts, repair


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


def partial_zip_model(x, start_at_models, merged_model, nodes, merge_node):
    if start_at_models is None:
        return merged_model(x)
    merge_node_id = int(nodes[merge_node]['layer'].split('.')[-1])
    coefs = [1/len(start_at_models)] * len(start_at_models)
    unzip_output = None
    for i, start_at_model in enumerate(start_at_models):
        if unzip_output is None:
            unzip_output = start_at_model.features[:merge_node_id+1](x) * coefs[i]
        else:
            unzip_output += start_at_model.features[:merge_node_id+1](x) * coefs[i]
    zip_conv_output = merged_model.features[merge_node_id+1:](unzip_output)

    out = merged_model.avgpool(zip_conv_output)
    out = torch.flatten(out, 1)
    out = merged_model.classifier(out)
    return out


def main():
    config_name = 'cifar10_my_vgg16'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    cur_config = deepcopy(raw_config)
    config = prepare_experiment_config(cur_config)

    train_loader = config['data']['train']['full']
    test_loader = config['data']['test']['full']

    base_models = [reset_bn_stats(base_model, train_loader) for base_model in
                   config['models']['bases']]
    Grapher = config['graph']

    criterion = torch.nn.CrossEntropyLoss()

    merging_fn_s = ['match_tensors_permute', 'match_tensors_zipit',
                    'match_tensors_identity']
    prefix_nodes = [3, 6, 10, 13, 17, 20, 23, 27, 30, 33, 37, 40, 43]
    # prefix_nodes = []
    res_dict = {merging_fn: {'bias_corr': [], 'zero_bias': [], 'rescale': [],
                             'repair': [], 'merger': [],
                             'manual_acc': []} for merging_fn in merging_fn_s}

    for merging_fn in merging_fn_s:
        print(f"merging_fn: {merging_fn}")
        graphs = [Grapher(deepcopy(base_model)).graphify() for
                  base_model in base_models]
        stop_at = None
        for start_at in [None]+prefix_nodes:
            print(f"start_at: {start_at}")
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

            # clear hooks
            Merge.clear_hooks()

            # manual validate merger
            merged_model = my_vgg.my_vgg16()
            merged_sd = Merge.get_merged_state_dict()
            merged_model.load_state_dict(deepcopy(merged_sd))
            merged_model = merged_model.to(device)
            correct = 0
            total = 0
            nodes = graphs[0].G.nodes
            merge_node = start_at - 2 if start_at is not None else None
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    if start_at is None:
                        outputs = merged_model(images)
                    else:
                        outputs = partial_zip_model(images,
                                                    Merge.start_at_models,
                                                    merged_model, nodes,
                                                    merge_node)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            manual_merger_acc = correct / total
            print(f"manual_merger_acc: {manual_merger_acc}")

            # bias correction
            merged_model = my_vgg.my_vgg16()
            merged_sd = deepcopy(Merge.get_merged_state_dict())
            merged_sd = interpolate_state_dicts(merged_sd, merged_sd,
                                                weight=0.5, bias_norm=True)
            merged_model.load_state_dict(merged_sd)
            merged_model = merged_model.to(device)
            correct = 0
            total = 0
            nodes = graphs[0].G.nodes
            merge_node = start_at - 2 if start_at is not None else None
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    if start_at is None:
                        outputs = merged_model(images)
                    else:
                        outputs = partial_zip_model(images,
                                                    Merge.start_at_models,
                                                    merged_model, nodes,
                                                    merge_node)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            bias_corr_acc = correct / total
            print(f"bias_corr_acc: {bias_corr_acc}")

            # zero bias
            merged_model = my_vgg.my_vgg16()
            merged_sd = deepcopy(Merge.get_merged_state_dict())
            for k in merged_sd.keys():
                if 'bias' in k:
                    merged_sd[k].fill_(0)
            merged_model.load_state_dict(merged_sd)
            merged_model = merged_model.to(device)
            correct = 0
            total = 0
            nodes = graphs[0].G.nodes
            merge_node = start_at - 2 if start_at is not None else None

            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    if start_at is None:
                        outputs = merged_model(images)
                    else:
                        outputs = partial_zip_model(images,
                                                    Merge.start_at_models,
                                                    merged_model, nodes,
                                                    merge_node)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            zero_bias_acc = correct / total
            print(f"zero_bias_acc: {zero_bias_acc}")

            # repair
            repaired_merged_model = repair(train_loader, base_models,
                                           Merge.merged_model, device,
                                           name='vgg16', variant='repair')
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    if start_at is None:
                        outputs = repaired_merged_model(images)
                    else:
                        outputs = partial_zip_model(images,
                                                    Merge.start_at_models,
                                                    repaired_merged_model,
                                                    nodes, merge_node)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            repair_acc = correct / total
            print(f"repair_acc: {repair_acc}")

            # rescale
            repaired_merged_model = repair(train_loader, base_models,
                                           Merge.merged_model, device,
                                           name='vgg16', variant='rescale')
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    if start_at is None:
                        outputs = repaired_merged_model(images)
                    else:
                        outputs = partial_zip_model(images,
                                                    Merge.start_at_models,
                                                    repaired_merged_model,
                                                    nodes, merge_node)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            rescale_acc = correct / total
            print(f"rescale_acc: {rescale_acc}")

            res_dict[merging_fn]['bias_corr'].append(bias_corr_acc)
            res_dict[merging_fn]['zero_bias'].append(zero_bias_acc)
            res_dict[merging_fn]['repair'].append(repair_acc)
            res_dict[merging_fn]['rescale'].append(rescale_acc)
            res_dict[merging_fn]['merger'].append(merger_acc)
            res_dict[merging_fn]['manual_acc'].append(manual_merger_acc)

    # # test ensemble
    # print("Testing ensemble")
    # for images, labels in test_loader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     correct = 0
    #     total = 0

    #     outputs = torch.zeros(labels.size(0), 10).to(device)
    #     for model in base_models:
    #         outputs += model(images) / len(base_models)
    #     _, predicted = torch.max(outputs, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    # ensemble_acc = correct / total
    # print(f"ensemble_acc: {ensemble_acc}")

    # # test direct average
    # mid_sd = interpolate_state_dicts(base_models[0].state_dict(),
    #                                  base_models[1].state_dict(), weight=0.5)
    # mid_model = my_vgg.my_vgg16()
    # mid_model.load_state_dict(mid_sd)
    # mid_model = mid_model.to(device)
    # direct_acc = validate(mid_model, test_loader, criterion, device)[1]
    # print(f"direct_acc: {direct_acc}")

    # res_dict['ensemble'] = ensemble_acc
    # res_dict['direct_avg'] = direct_acc

    torch.save(res_dict, "vgg16_acc.pth")


if __name__ == '__main__':
    main()
