import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--resume', type=bool, default=True, help="Resume the training from the last run")
    parser.add_argument('--rounds', type=int, default=100000, help="The maximum number of iterations")
    parser.add_argument('--epochs', type=int, default=25,
                        help="The number of local epochs for the fixed strategy and tau_0 in adacomm")
    parser.add_argument('--epochs1', type=int, default=1, help="Represents tau_2 in our strategy")
    parser.add_argument('--epochs2', type=int, default=2, help="Represents tau_3 in our strategy")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--max_time', type=float, default=46, help="Max training time budgets in minutes")
    parser.add_argument('--dropout', type=float, default=0.0, help="Workers dropout rate")

    # model arguments
    parser.add_argument('--model', type=str, default='mobilenet', choices=["mobilenet", ], help='model name')
    parser.add_argument('--log_dir', type=str, default='./logs', help="log folder")
    parser.add_argument('--strategy', type=str, default='rpc', choices=["adacomm", "fixed", "rpc"],
                        help="The strategy to be executed")
    parser.add_argument("--split_mode", default='iid', help="The data splitting mechanism", choices=["iid", "niid"])

    # other arguments
    parser.add_argument('--dataset', type=str, default='tiny_imagenet', help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', default=False, action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (default: 1)')

    parser.add_argument('--sub_targets', type=int, default=5,
                        help='Only use a sub group of the available classes(-1 mean all classes)')

    parser.add_argument("--data_path", default="./data/", help="Where to download and store the data", )
    parser.add_argument("--min_size", default=500, help="dataset key", )
    parser.add_argument("--max_size", default=600, help="dataset key", )
    parser.add_argument("--max_category", default=3, help="The number of classes each work can see in  niid sampling", )
    parser.add_argument("--batch_size", default=64, help="The training batch size", )

    args = parser.parse_args()

    return args
