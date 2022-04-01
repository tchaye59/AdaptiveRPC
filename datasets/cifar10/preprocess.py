from torchvision import datasets, transforms

# Load data function
from datasets.utils import noniid, iid


def load_cifar10(args):
    """The load function of the dataset

    Args:
       args: The arguments.

    Returns:
        train_dataset, test_dataset: the datasets.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)

    test_dataset = datasets.CIFAR10(args.data_path, train=False, transform=transform_test)
    return train_dataset, test_dataset


def split_cifar10(args, dataset, num_clients):
    if args.split_mode == 'iid':
        return iid(args, dataset, num_clients)
    return noniid(args, dataset, num_clients)
