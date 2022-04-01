import os.path
import sys
import time

import numpy as np
import torch
from mpi4py import MPI
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloader import load_data, split_dataset
from models.nets import CNNCifar, CNNMnist, MLP, MobileNet
from utils.options import args_parser
from utils.utils import FedAvg, LocalUpdate, AdaCommStrategy, CommStrategy, FixedCommStrategy, \
    evaluate, seed_all, RPCStrategy
import dill

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
SIZE = comm.Get_size()

print(f"Rank = {rank} | Size = {SIZE}")

SERVER = 0
N_CLIENTS = SIZE - 1


def log(msg):
    print(msg)
    sys.stdout.flush()


def run_server(args, model: torch.nn.Module, train_dataset, test_dataset, comm_strategy: CommStrategy):
    strategy = args.strategy
    writer = SummaryWriter(log_dir=args.log_dir + "/" + args.dataset + "/" + strategy + "/" + args.model)
    start_time = time.time()

    # Init the clients dataset
    data_idx = [None, ] + split_dataset(args, train_dataset, N_CLIENTS)  # The first is the server
    log(f"Chunks sizes : {[len(x) for x in data_idx[1:]]}")
    comm.scatter(data_idx, root=SERVER)

    global_weight = model.state_dict()

    stop = False
    round = 1
    tau = comm_strategy.reset()
    tau_key = f"{args.dataset}/{strategy}/Global/tau"
    store_name = os.path.join(args.log_dir, f"{args.strategy}_store.dill")
    model_path = os.path.join(args.log_dir, f"{args.strategy}_model.pth")

    if args.resume and os.path.exists(store_name):
        store = dill.load(open(store_name, "rb"))
        round = store['round']
        comm_strategy = store['comm_strategy']
        tau = store['tau']
        start_time -= store['duration']
        global_weight = torch.load(model_path)

    while not stop:
        workers_droupout_mask = np.random.choice([False, True], size=SIZE, replace=True, p=(args.dropout, 1 - args.dropout))
        taus = np.array([tau for _ in range(SIZE)]) * workers_droupout_mask.astype(np.uint8)
        workers_droupout_mask = workers_droupout_mask[1:]

        log(f"Round {round}/{args.rounds} | Tau : {tau}")
        comm.scatter(taus, root=SERVER)
        comm.bcast(global_weight, root=SERVER)

        data = comm.gather(None, root=SERVER)
        data = data[1:]  # remove the first item since it is the server
        clients_weights = np.array([item[0] for item in data])[workers_droupout_mask]
        sizes = np.array([item[1] for item in data]).flatten()[workers_droupout_mask]
        losses = np.stack([item[2] for item in data])[workers_droupout_mask, :, 0]
        loss = np.array([item[2][-1] for item in data]).flatten()[workers_droupout_mask]
        acc = np.array([item[3] for item in data]).flatten()[workers_droupout_mask]
        steps = np.array([item[4] for item in data]).flatten()[workers_droupout_mask]

        # Global averaging
        global_weight = FedAvg(list(clients_weights), sizes)
        model.load_state_dict(global_weight)

        # Test the model
        eval_info = evaluate(args, test_dataset, model)

        duration = time.time() - start_time
        # save logs
        # global_acc
        global_acc = (sizes * acc).sum() / sizes.sum()
        key = 'acc'
        key = f"{args.dataset}/{strategy}/Global/{key}"
        writer.add_scalar(key, global_acc, round)
        writer.add_scalar(key + "_time", global_acc, duration)
        log(f"{key}, {global_acc}")

        # global_loss
        global_loss = (sizes * loss).sum() / sizes.sum()
        key = 'loss'
        key = f"{args.dataset}/{strategy}/Global/{key}"
        writer.add_scalar(key, global_loss, round)
        writer.add_scalar(key + "_time", global_loss, duration)
        log(f"{key}, {global_loss}")

        # Test logs
        for key, val in eval_info.items():
            key = f"{args.dataset}/{strategy}/Global/test_{key}"
            writer.add_scalar(key, val, round)
            writer.add_scalar(key + "_time", val, duration)
            log(f"{key}, {val}")

        # Stop
        stop = round > args.rounds
        if stop:
            log("Earlier stopping! Max iteration reached.")
        if duration >= (args.max_time * 60):
            stop = True
            log("Earlier stopping! Max time budget reached.")
        # Tell the workers to continue or stop
        stop = comm.bcast(stop, root=SERVER)
        # Update the communication period
        if args.strategy == "rpc":
            tau = comm_strategy.step(global_loss)
        elif args.strategy == "adacomm":
            tau = comm_strategy.step(global_loss)
        else:
            tau = comm_strategy.step()
        writer.add_scalar(tau_key + "_time", tau, duration)
        writer.add_scalar(tau_key, tau, round)

        store = {
            "comm_strategy": comm_strategy,
            "round": round,
            "tau": tau,
            "duration": duration,
        }
        torch.save(global_weight, model_path)
        dill.dump(store, open(store_name, "wb"))
        round += 1


def run_client(args, model, dataset):
    # dataset
    idxs = comm.scatter(None, root=SERVER)
    trainner = LocalUpdate(args=args, dataset=dataset, idxs=idxs)
    data_size = len(idxs)
    stop = False
    while not stop:
        # Get initial weights from server
        tau = comm.scatter(None, root=SERVER)
        local_weight = comm.bcast(None, root=SERVER)
        model.load_state_dict(local_weight)
        # Do local updates
        info = trainner.train(model, tau)
        # Send to the server
        data = model.state_dict(), data_size, *info
        comm.gather(data, root=SERVER)

        # stop or repeat
        stop = comm.bcast(None, root=SERVER)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset and split users
    train_dataset, test_dataset = load_data(args)
    # seed all
    if args.seed > 0:
        seed_all(args.seed, rank)

    # build model
    model = None
    num_classes = None
    if args.model == 'cnn' and args.dataset == 'cifar10':
        model = CNNCifar(args=args).to(args.device)
        num_classes = 10
    elif args.model == 'cnn' and args.dataset == 'mnist':
        model = CNNMnist(args=args).to(args.device)
        num_classes = 10
    elif args.model == 'mobilenet' and args.dataset == 'tiny_imagenet':
        num_classes = args.sub_targets
        num_classes = num_classes if num_classes > 1 else 200
        model = MobileNet(args=args, num_classes=num_classes).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        num_classes = 10
        img_size = train_dataset[0][0].shape
        for x in img_size:
            len_in *= x
        model = MLP(dim_in=len_in, dim_hidden=200, dim_out=num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # Set comm strategy
    if args.strategy == "adacomm":
        comm_strategy = AdaCommStrategy(args)
    elif args.strategy == "rpc":
        comm_strategy = RPCStrategy(args)
    else:
        comm_strategy = FixedCommStrategy(args)

    if rank == SERVER:
        log(f">>>>>>>>>>>>>>>>>>>>Model: {args.model} | STRATEGY: {args.strategy} | DATASET: {args.dataset} | SPLIT "
            f"MODE: {args.split_mode}<<<<<<<<<<<<<<<<<<<<")
        run_server(args, model, train_dataset, test_dataset, comm_strategy)
    else:
        run_client(args, model, train_dataset)
