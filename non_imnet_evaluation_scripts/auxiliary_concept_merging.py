import os
import torch
import random

from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np

from utils import *
from model_merger import ModelMerge

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)



def run_auxiliary_experiment(merging_fn, experiment_config, device, stop_at=None, csv_file='', start_at=None):
    # for pair in tqdm(pairs, desc='Evaluating Pairs...'):
        # experiment_config = inject_pair(experiment_config, pair)
    config = prepare_experiment_config(experiment_config)
    train_loader = config['data']['train']['full']
    base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
    
    Grapher = config['graph']
    graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
    Merge = ModelMerge(*graphs, device=device)
    Merge.transform(
        deepcopy(config['models']['new']), 
        train_loader, 
        transform_fn=get_merging_fn(merging_fn),
        metric_classes=config['metric_fns'],
        stop_at=stop_at,
        start_at=experiment_config['start_at']
    )
    reset_bn_stats(Merge, train_loader)
    
    results = evaluate_model(experiment_config['eval_type'], Merge, config)
    results['Time'] = Merge.compute_transform_time
    results['Merging Fn'] = merging_fn
    # for idx, split in enumerate(pair):
    #     results[f'Split {CONCEPT_TASKS[idx]}'] = split
    results ['Bases'] = experiment_config['model']['bases']
    write_to_csv(results, csv_file=csv_file)
    print(results)
    
    # sd_merged = Merge.get_merged_state_dict()
    # if 'save_path' in experiment_config:
    #     torch.save(sd_merged, experiment_config['save_path'])
    #     print(f'Saved merged model to {experiment_config["save_path"]}')
        
    return results


if __name__ == "__main__":
    # config_name = 'cifar5_vgg'
    config_name = 'cifar10_my_vgg16'
    skip_pair_idxs = []
    merging_fns = [
        'match_tensors_permute',
        # 'match_tensors_identity',
    ]
    stop_at = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)  # 返回config文件中的字典，添加了一个device键值对
    # raw_config['start_at'] = 37
    # model_dir = raw_config['model']['dir']
    # model_name = raw_config['model']['name']
    # run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)

    csv_file = os.path.join(
        './csvs',
        raw_config['dataset']['name'],
        raw_config['model']['name'],
        raw_config['eval_type'],
        'auxiliary_functions.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    prefix_nodes = [3, 6, 10, 13, 17, 20, 23, 27, 30, 33, 37, 40, 43]

    cur_config = deepcopy(raw_config)

    start_acc_pair = {}
    for prefix_node in prefix_nodes:
        cur_config['start_at'] = prefix_node

        with torch.no_grad():
            for merging_fn in merging_fns:
                node_results = run_auxiliary_experiment(
                    merging_fn=merging_fn,
                    experiment_config=cur_config,
                    device=device,
                    csv_file=csv_file,
                    stop_at=stop_at
                )
            start_acc_pair[prefix_node] = node_results['Accuracy']
    print(start_acc_pair)
