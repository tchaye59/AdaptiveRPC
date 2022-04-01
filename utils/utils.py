import copy
import random
import sys
import time
import math as m

import numpy as np
import torchmetrics as tm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def seed_all(seed, rank):
    worker_seed = seed + rank
    # torch.use_deterministic_algorithms(True)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def log(msg):
    print(msg)
    sys.stdout.flush()


def FedAvg(w, sizes):
    total_size = sum(sizes)
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] *= sizes[0]

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += sizes[i] * w[i][k]
        w_avg[k] = torch.div(w_avg[k], total_size)
    return w_avg


def evaluate(args, dataset, net):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("Evaluation")
    net.eval()
    accracy_metric = tm.Accuracy()
    loss_metric = tm.MeanMetric()
    loss_func = nn.CrossEntropyLoss()
    for images, labels in tqdm(loader):
        images, labels = images.to(args.device), labels.to(args.device)
        log_probs = net(images)
        loss = loss_func(log_probs, labels)
        loss_metric.update(loss.detach().cpu())
        accracy_metric.update(log_probs.detach().cpu(), labels.cpu())

    return {"loss": loss_metric.compute().numpy(), "acc": accracy_metric.compute().numpy()}


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        local_dataset = torch.utils.data.Subset(dataset, idxs)
        self.train_loader = DataLoader(local_dataset, batch_size=self.args.batch_size, shuffle=True)

    def eval(self, net):
        net.eval()
        # train and update
        accracy_metric = tm.Accuracy()
        loss_metric = tm.MeanMetric()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels.cpu())
                loss_metric.update(loss.detach().cpu())
                accracy_metric.update(log_probs.detach().cpu(), labels.cpu())
        return {"loss": loss_metric.compute().numpy(), "acc": accracy_metric.compute().numpy()}

    def train(self, net, tau):
        if tau == 0:
            return 0, 0, 0, 0
        losses = []
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        accracy_metric = tm.Accuracy()
        loss_metric = tm.MeanMetric()
        for iter in range(tau):
            loss_metric.reset()
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss_metric.update(loss.detach().cpu())
                accracy_metric.update(log_probs.detach().cpu(), labels.cpu())
                loss.backward()
                optimizer.step()

            losses.append(loss_metric.compute().numpy())

        steps = iter + 1
        return np.array(losses), accracy_metric.compute().numpy(), steps


class CommStrategy(object):
    def __init__(self, initial_tau):
        self.initial_tau = initial_tau

    def step(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        return self.initial_tau


class FixedCommStrategy(CommStrategy):
    def __init__(self, args):
        super().__init__(args.epochs)
        self.tau = args.epochs

    def step(self, *args, **kwargs):
        return self.tau


class AdaCommStrategy(CommStrategy):
    def __init__(self, args):
        super().__init__(args.epochs)
        self.period = self.initial_tau
        self.tau = args.epochs
        self.start_loss = None

    def step(self, loss):
        if self.start_loss is None:
            self.start_loss = loss
        else:
            self.tau = m.ceil(m.sqrt(loss / self.start_loss) * self.initial_tau)
        return self.tau


class RPCStrategy(CommStrategy):
    def __init__(self, args):
        super().__init__(args.epochs1)
        self.args = args
        self.tau = args.epochs1
        self.tau2 = args.epochs1
        self.tau3 = self.args.epochs2
        # The first element is just a place holder
        self.losses = [None, ]
        self.taus = [None, ]
        self.tau_min = 1
        self.tau_max = 100
        self.k = 1
        self.start_point_update_step = -1
        self.T1 = 2
        self.T2 = 3

    def restart(self, tau2, tau3):
        self.tau = tau2
        self.tau2 = tau2
        self.tau3 = tau3
        self.losses = [None, ]
        self.taus = [None, ]
        self.k = 1
        self.T1 = 2
        self.T2 = 3
        return tau2

    def step(self, loss):
        self.k += 1
        if self.start_point_update_step > 3 and (self.k % self.start_point_update_step == 0):
            self.T1 = self.k - 3
        # T2 = self.k - 1
        self.losses.append(loss)
        self.taus.append(self.tau)

        if self.k > 3:
            alpha = (self.losses[self.k - 1] - self.losses[self.T1 - 1]) / (
                    self.losses[self.T2 - 1] - self.losses[self.T1 - 1])
            zero_val = 1 if loss < self.losses[self.T2 - 1] else -1
            tau = alpha * self.no_zero(self.taus[self.T2] - self.taus[self.T1], zero_val=zero_val) + self.taus[self.T1]
            # Normalize tau
            tau = max(tau, self.tau_min)
            tau = min(tau, self.tau_max) if self.tau_max > 0 else tau
            if loss < self.losses[self.T2 - 1]:
                self.tau = m.ceil(tau)
                self.T2 = self.k
            else:
                self.tau = m.floor(tau)
                self.taus[self.T2] = self.tau
                if self.taus[self.T2] < self.taus[self.T1]:
                    self.taus[self.T1] -= max(self.taus[self.T1] - 5, 1)
                    tau_2 = max(self.taus[self.T1] - 5, 1)
                    tau_3 = tau_2 + 1
                    return self.restart(tau_2, tau_3)
        elif self.k < 3:
            self.tau = self.tau2
        elif self.k == 3:
            self.tau = self.tau3

        return self.tau

    def no_zero(self, val, zero_val=1e-18):
        if val == 0:
            return zero_val
        return val


if __name__ == "__main__":
    from options import args_parser

    st = RPCStrategy(args_parser())
    st.reset()
    st.step(1.2557084888520866)
    st.step(1.064652668426334)
    st.step(0.7617901886252041)
    st.step(0.6531890752138915)
    st.step(0.05)
    st.step(0.009)
