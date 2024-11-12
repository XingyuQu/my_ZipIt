config = {
    'dataset': {
        'name': 'cifar100',
        'shuffle_train': True
    },
    'model': {
        'name': 'my_vgg16_bn',
        'bases': [
            './checkpoints/cifar100_my_vgg16_bn_1.pt',
            './checkpoints/cifar100_my_vgg16_bn_2.pt',
        ]
    },
    'eval_type': 'logits_same_task',
    'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance', 'mean'],
    'save_path': './checkpoints/cifar100_my_vgg16_bn_permute_1_2.pt',
}