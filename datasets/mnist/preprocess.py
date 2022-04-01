from torchvision import datasets, transforms


# Load data function
from datasets.utils import iid, noniid


def load_mnist(args):
    """The load function of the dataset

    Args:
       args: The arguments.

    Returns:
        train_dataset, test_dataset: the datasets.
    """

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        args.data_path, train=True, download=True, transform=transform_train
    )

    test_dataset = datasets.MNIST(args.data_path, train=False, transform=transform_test)

    return train_dataset, test_dataset


def split_mnist(args, dataset, num_clients):
    if args.split_mode == 'iid':
        return iid(args, dataset, num_clients)
    return noniid(args, dataset, num_clients)
