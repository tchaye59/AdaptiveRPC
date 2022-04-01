import argparse

from datasets.cifar10.preprocess import *
from datasets.fashionmnist.preprocess import *
from datasets.mnist.preprocess import *

# load data
from datasets.tiny_imagenet.preprocess import load_tiny_imagenet, split_tiny


def load_data(args):
    """The load function of the dataset

    Args:
       args: The arguments.

    Returns:
        train_dataset, test_dataset: the datasets.
    """
    if args.dataset == "cifar10":
        train_dataset, test_dataset = load_cifar10(args)

    elif args.dataset == "mnist":
        train_dataset, test_dataset = load_mnist(args)

    elif args.dataset == "fashionmnist":
        train_dataset, test_dataset = load_fashionmnist(args)
    elif args.dataset == "tiny_imagenet":
        train_dataset, test_dataset = load_tiny_imagenet(args)
    else:
        raise NotImplemented("dataset not found")
    return train_dataset, test_dataset


# Split function
def split_dataset(args, dataset, clients):
    """The split function

    Args:
       args: The arguments.
       dataset: The dataset to split
       clients: The number of clients
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    sub_datasets = [[] for i in range(clients)]

    if args.dataset == "cifar10":
        sub_datasets = split_cifar10(args, dataset, clients)

    elif args.dataset == "mnist":
        sub_datasets = split_mnist(args, dataset, clients)

    elif args.dataset == "fashionmnist":
        sub_datasets = split_fashionmnist(args, dataset, clients)

    elif args.dataset == "tiny_imagenet":
        sub_datasets = split_tiny(args, dataset, clients)

    return sub_datasets


def parse_args():
    # Logging setup
    parser = argparse.ArgumentParser(description="parameters.")

    parser.add_argument(
        "--data_path", default="../data/",
        help="dataset key",
    )

    parser.add_argument(
        "--split_mode", default='niid',
        help="dataset key",
    )

    parser.add_argument(
        "--dataset", default="tiny_imagenet",
        help="dataset key",
    )

    parser.add_argument(
        "--min_size_rate", default=0.1,
        help="dataset key",
    )

    parser.add_argument(
        "--sub_targets", default=10,
        help="dataset key",
    )

    parser.add_argument(
        "--max_category", default=3,
        help="dataset key",
    )

    parser.add_argument("--min_size", default=600, help="dataset key", )
    parser.add_argument("--max_size", default=1000, help="dataset key", )

    parser.add_argument(
        "--batch_size", default=32,
        help="dataset key",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_dataset, test_dataset = load_data(args)
    x = train_dataset[0][0]
    x = test_dataset[0][0]

    print(train_dataset[0][0].shape)
    print(test_dataset[0][0].shape)

    num_clients = 100

    sub_datasets = split_dataset(args, train_dataset, num_clients)
    pass
