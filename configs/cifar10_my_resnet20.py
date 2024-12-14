config = {
    'dataset': {
        'name': 'cifar10',
        'shuffle_train': True
    },
    'model': {
        'name': 'my_resnet20',
        'bases': [
            './checkpoints/cifar10_my_resnet20_1.pt',
            './checkpoints/cifar10_my_resnet20_2.pt',
        ]
    },
    'eval_type': 'logits_same_task',
    'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance', 'mean'],
    'save_path': './checkpoints/cifar10_my_resnet20_permute_1_2.pt',
}