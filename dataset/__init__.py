import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.coord_dataset import tracking_dataset_pretrain, tracking_dataset, tracking_dataset_eval, tracking_dataset_infer

from dataset.randaugment import RandomAugment


def _resolve_resize_dims(image_res):
    if isinstance(image_res, int):
        return (image_res, image_res)
    if isinstance(image_res, (list, tuple)) and len(image_res) == 2:
        width, height = image_res
        return (height, width)
    raise ValueError(f"image_res must be an int or a (width, height) tuple but got {image_res} and the type is {type(image_res)}")


def create_dataset(dataset, config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    resize_hw = _resolve_resize_dims(config['image_res'])

    tracking_transform = transforms.Compose([
        transforms.Resize(resize_hw, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    saliency_transform = transforms.Compose([
        transforms.Resize(resize_hw, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])

    if dataset == 'pretrain':
        dataset = tracking_dataset_pretrain(config['train_file'], config['image_root'], tracking_transform, max_words=config["max_words"])
        

    elif dataset == 'tracking':
        dataset = tracking_dataset(config['train_file'], 
                                   config['image_root'], 
                                   tracking_transform, 
                                   saliency_transform, 
                                   max_words=config["max_words"])
        

    elif dataset == 'eval_tracking':
        dataset = tracking_dataset_eval(config['train_file'], config['eval_image_root'], tracking_transform, max_words=config["max_words"])
    
    elif dataset == 'inference':
        dataset = tracking_dataset_infer(config['eval_image_root'], tracking_transform, max_words=config["max_words"])
    
    return dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
