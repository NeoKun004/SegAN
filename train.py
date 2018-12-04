from __future__ import print_function

import os
import json
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from collections import OrderedDict
from torch.autograd import Variable
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver

from model.critic import CriticNet
from model.segmentor import SegmentorNet
from LoadData import Dataset, loader, Dataset_val

base_path = os.path.dirname(os.path.abspath(__file__))

ex = Experiment()
fs_observer = FileStorageObserver.create("results")
ex.observers.append(fs_observer)


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def dice_loss(input_, target):
    num = input_ * target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)

    den1 = input_ * input_
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)

    den2 = target * target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)

    dice = 2 * (num / (den1 + den2))

    dice_total = 1 - 1 * torch.sum(dice) / dice.size(0)  # divide by batch size

    return dice_total


class SummaryWriter:
    def __init__(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.dir = dir
        self.dict = {}

    def add_scalar(self, tag, value, key):
        if tag not in self.dict:
            self.dict[tag] = {}

        self.dict[tag][key] = value

    def commit(self):
        with open(os.path.join(self.dir, "writer.json"), "w+") as f:
            json.dump(self.dict, f)


@ex.automain
def main(batch_size, n_epochs, lr, beta1, decay, _run):
    assert torch.cuda.is_available()

    writer = SummaryWriter(os.path.join(base_path, "runs", "experiment-{}".format(_run._id)))
    model_path = os.path.join(fs_observer.dir, "best_model.pth")

    outputs_path = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

    cudnn.benchmark = True
    s_model = SegmentorNet().cuda()
    c_model = CriticNet().cuda()

    s_optimizer = optim.Adam(s_model.parameters(), lr=lr, betas=(beta1, 0.999))
    c_optimizer = optim.Adam(c_model.parameters(), lr=lr, betas=(beta1, 0.999))

    dataloaders = {
        "train": loader(Dataset('./'), batch_size),
        "validation": loader(Dataset_val('./'), 36)
    }

    best_IoU = 0.0
    s_model.train()
    for epoch in range(n_epochs):
        progress_bar = tqdm(dataloaders["train"], desc="Epoch {} - train".format(epoch))

        s_losses = []
        s_losses_joint = []
        c_losses = []
        dices = []

        for i, (inputs, targets) in enumerate(progress_bar):
            c_model.zero_grad()

            inputs = Variable(inputs).cuda()
            targets = Variable(targets).cuda().type(torch.FloatTensor).cuda()

            outputs = s_model(inputs)
            outputs = F.sigmoid(outputs)
            outputs = outputs.detach()
            outputs_masked = inputs.clone()
            inputs_mask = inputs.clone()

            for d in range(3):
                outputs_masked[:, d, :, :] = inputs_mask[:, d, :, :].unsqueeze(1) * outputs
            outputs_masked = outputs_masked.cuda()

            results = c_model(outputs_masked)
            targets_masked = inputs.clone()
            for d in range(3):
                targets_masked[:, d, :, :] = inputs_mask[:, d, :, :].unsqueeze(1) * targets

            for d in range(3):
                targets_masked[:, d, :, :] = inputs_mask[:, d, :, :].unsqueeze(1) * targets

            targets_masked = targets_masked.cuda()
            targets_D = c_model(targets_masked)
            loss_D = - torch.mean(torch.abs(results - targets_D))
            loss_D.backward()
            c_optimizer.step()

            for p in c_model.parameters():
                p.data.clamp_(-0.05, 0.05)

            s_model.zero_grad()
            outputs = s_model(inputs)
            outputs = F.sigmoid(outputs)

            for d in range(3):
                outputs_masked[:, d, :, :] = inputs_mask[:, d, :, :].unsqueeze(1) * outputs
            outputs_masked = outputs_masked.cuda()

            results = c_model(outputs_masked)
            for d in range(3):
                targets_masked[:, d, :, :] = inputs_mask[:, d, :, :].unsqueeze(1) * targets
            targets_masked = targets_masked.cuda()

            targets_G = c_model(targets_masked)
            loss_dice = dice_loss(outputs, targets)
            loss_G = torch.mean(torch.abs(results - targets_G))
            loss_G_joint = loss_G + loss_dice
            loss_G_joint.backward()
            s_optimizer.step()

            c_losses.append(loss_D.data[0])
            s_losses.append(loss_G.data[0])
            s_losses_joint.append(loss_G_joint.data[0])
            dices.append(loss_dice.data[0])

            progress_bar.set_postfix(OrderedDict({
                "c_loss": np.mean(c_losses),
                "s_loss": np.mean(s_losses),
                "s_loss_joint": np.mean(s_losses_joint),
                "dice": np.mean(dices)
            }))

        mean_c_loss = np.mean(c_losses)
        mean_s_loss = np.mean(s_losses)
        mean_s_loss_joint = np.mean(s_losses_joint)
        mean_dice = np.mean(dices)

        c_loss_tag = "train.c_loss"
        s_loss_tag = "train.s_loss"
        s_losses_joint_tag = "train.s_loss_joint"
        dice_loss_tag = "train.loss_dice"

        writer.add_scalar(c_loss_tag, mean_c_loss, epoch)
        writer.add_scalar(s_loss_tag, mean_s_loss, epoch)
        writer.add_scalar(s_losses_joint_tag, mean_s_loss_joint, epoch)
        writer.add_scalar(dice_loss_tag, mean_dice, epoch)

        if epoch % 10 == 0:
            progress_bar = tqdm(dataloaders["validation"], desc="Epoch {} - validation".format(epoch))

            s_model.eval()
            IoUs, dices = [], []
            for i, (inputs, targets) in enumerate(progress_bar):
                inputs = Variable(inputs).cuda()
                targets = Variable(targets).cuda()

                pred = s_model(inputs)
                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1

                pred = pred.type(torch.LongTensor)
                pred_np = pred.data.cpu().numpy()

                targets = targets.data.cpu().numpy()
                for x in range(inputs.size()[0]):
                    IoU = np.sum(pred_np[x][targets[x] == 1]) / float(
                        np.sum(pred_np[x]) + np.sum(targets[x]) - np.sum(pred_np[x][targets[x] == 1])
                    )
                    dice = np.sum(pred_np[x][targets[x] == 1]) * 2 / float(np.sum(pred_np[x]) + np.sum(targets[x]))
                    IoUs.append(IoU)
                    dices.append(dice)

            s_model.train()
            IoUs = np.array(IoUs, dtype=np.float64)
            dices = np.array(dices, dtype=np.float64)
            mIoU = np.mean(IoUs, axis=0)
            mDice = np.mean(dices, axis=0)

            progress_bar.set_postfix(OrderedDict({
                "mIoU": np.mean(mIoU),
                "mDice": np.mean(mDice)
            }))

            miou_tag = "validation.miou"
            mdice_tag = "validation.mdice"

            writer.add_scalar(miou_tag, mIoU, epoch)
            writer.add_scalar(mdice_tag, mDice, epoch)
            writer.commit()

            if mIoU > best_IoU:
                best_IoU = mIoU
                torch.save(s_model, model_path)

        if epoch % 25 == 0:
            lr = max(lr * decay, 0.00000001)
            s_optimizer = optim.Adam(s_model.parameters(), lr=lr, betas=(beta1, 0.999))
            c_optimizer = optim.Adam(c_model.parameters(), lr=lr, betas=(beta1, 0.999))
