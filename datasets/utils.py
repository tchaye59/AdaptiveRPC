import argparse
import random

import numpy as np


def grouby_target(idxs_labels, labels):
    # Groups indexes by target
    groups = [[] for _ in range(len(labels))]
    [groups[idxs_labels[i, 1]].append(idxs_labels[i, 0]) for i in range(idxs_labels.shape[0])]
    return groups


def noniid(args, dataset, num_clients):
    """
    Sample non-I.I.D clients data from dataset
    :param args:
    :param num_clients:
    :param dataset:
    :return:
    """
    dict_users = {i: [] for i in range(num_clients)}
    train_idxs = np.arange(len(dataset))
    train_labels = np.array(dataset.targets)
    labels = np.unique(train_labels)

    # First, concatenate indexes and targets. After repeat and shuffle
    idxs_labels = np.concatenate([train_idxs[:, np.newaxis], train_labels[:, np.newaxis]], axis=-1)
    np.random.shuffle(idxs_labels)

    # Groups indexes by target
    groups = [[[], 0] for _ in range(len(labels))]
    [groups[idxs_labels[i, 1]][0].append(idxs_labels[i, 0]) for i in range(idxs_labels.shape[0])]

    # The samples size must be different per client
    sizes = np.array([args.min_size for _ in range(num_clients)])
    if args.max_size > args.min_size:
        sizes += np.random.randint(0, args.max_size - args.min_size, num_clients)

    # Divide and assign
    print("Divide and assign samples to clients")
    g_idx = 0
    for client_idx, size in enumerate(sizes):
        # select sub groups
        sub_groups = []
        for i in range(args.max_category):
            sub_groups.append(groups[(client_idx+i) % len(groups)])

        while len(dict_users[client_idx]) < size:
            for sub_group_idx, (x_indexes, pointer) in enumerate(sub_groups):
                if pointer == len(x_indexes):
                    random.shuffle(x_indexes)
                    pointer = 0
                dict_users[client_idx].append(x_indexes[pointer])
                # increment the pointer
                pointer += 1
                sub_groups[sub_group_idx][1] = pointer

        g_idx += args.max_category

    # print(sum([len(x) for x in dict_users.values()]))
    return [x for x in dict_users.values()]


def iid(args, dataset, num_clients):
    """
    Sample non-I.I.D clients data from dataset
    :param args:
    :param num_clients:
    :param dataset:
    :return:
    """
    dict_users = {i: [] for i in range(num_clients)}
    train_idxs = np.arange(len(dataset))
    train_labels = np.array(dataset.targets)
    labels = np.unique(train_labels)

    # First, concatenate indexes and targets. After shuffle
    idxs_labels = np.concatenate([train_idxs[:, np.newaxis], train_labels[:, np.newaxis]], axis=-1)
    np.random.shuffle(idxs_labels)

    # Groups indexes by target
    groups = [[[], 0] for _ in range(len(labels))]
    [groups[idxs_labels[i, 1]][0].append(idxs_labels[i, 0]) for i in range(idxs_labels.shape[0])]

    # Divide and assign
    print("Divide and assign samples to clients")
    for client_idx in range(num_clients):
        while len(dict_users[client_idx]) < args.min_size:
            for group_idx, (x_indexes, pointer) in enumerate(groups):
                if pointer == len(x_indexes):
                    pointer = 0
                dict_users[client_idx].append(x_indexes[pointer])
                # increment the pointer
                pointer += 1
                groups[group_idx][1] = pointer

    #print(sum([len(x) for x in dict_users.values()]))
    return [x for x in dict_users.values()]


def parse_args():
    # Logging setup
    parser = argparse.ArgumentParser(description="parameters.")

    parser.add_argument(
        "--split_mode", default='niid',
        help="dataset key",
    )

    parser.add_argument(
        "--iid_share", default=True,
        help="dataset key",
    )

    parser.add_argument(
        "--add_error", default=False,
        help="dataset key",
    )

    parser.add_argument("--data_path", default="./data/", help="dataset key", )

    parser.add_argument(
        "--iid_rate", default=1,
        help="dataset key",
    )

    parser.add_argument(
        "--min_size", default=500,
        help="dataset key",
    )

    parser.add_argument(
        "--max_size", default=500,
        help="dataset key",
    )

    parser.add_argument(
        "--global_dataset", default=False,
        help="dataset key",
    )

    parser.add_argument(
        "--dataset", default="tiny_imagenet",
        help="dataset key",
    )

    parser.add_argument('--sub_targets', type=int, default=20,
                        help='Only use a sub group of the available classes(-1 mean all classes)')

    # noniid keys
    parser.add_argument(
        "--min_size_rate", default=0.1,
        help="dataset key",
    )

    parser.add_argument(
        "--data_repeat", default=5,
        help="dataset key",
    )

    parser.add_argument(
        "--max_category", default=3,
        help="dataset key",
    )

    parser.add_argument(
        "--umbalance_weights", default=[1, 3, 5, 10, ],
        help="dataset key",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from dataloader import load_data, split_dataset

    train_ds, test_ds = load_data(args)

    x = train_ds[0]

    num_clients = 100
    x = noniid(args, train_ds, num_clients)

    x = sub_datasets = split_dataset(args, train_ds)
    pass
