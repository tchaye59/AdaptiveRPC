import os.path

from torchvision import datasets, transforms

# Load data function
from datasets.tiny_imagenet.tiny import TinyImageNetDataset
from datasets.utils import noniid, iid


def load_tiny_imagenet(args):
    """The load function of the dataset

    Args:
       args: The arguments.

    Returns:
        train_dataset, test_dataset: the datasets.
    """
    transform_train = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ]),
        ]
    )
    sub_targets = args.sub_targets
    path = os.path.join(args.data_path, "tiny_imagenet")
    train_dataset = TinyImageNetDataset(path, download=True, sub_targets=sub_targets, transform=transform_train)

    test_dataset = TinyImageNetDataset(path, mode="val", sub_targets=sub_targets, transform=transform_test)
    return train_dataset, test_dataset


def split_tiny(args, dataset, num_clients):
    if args.split_mode == 'iid':
        return iid(args, dataset, num_clients)
    return noniid(args, dataset, num_clients)


if __name__ == "__main__":
    from utils.options import args_parser
    from models.nets import MobileNet

    args = args_parser()
    train_dataset, test_dataset = load_tiny_imagenet(args)
    a = train_dataset[0]
    a = test_dataset[0]

    a = a[0].unsqueeze(0)
    m = MobileNet(None)
    x = m(a)
    pass
