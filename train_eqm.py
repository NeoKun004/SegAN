import os
import torch
import torch.optim as optim

from torch.utils import data
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver

from dataset import SegANDataset
from model.critic import CriticNet
from model.segmentor import SegmentorNet
from summary_writer import SummaryWriter


base_path = os.path.dirname(os.path.abspath(__file__))

ex = Experiment()
fs_observer = FileStorageObserver.create("results")
ex.observers.append(fs_observer)


@ex.automain
def main(train_fpath, val_fpath, batch_size, n_epochs, lr, beta1, decay,
         transform_train, transform_val, postprocess, _run):
    assert torch.cuda.is_available()

    writer = SummaryWriter(os.path.join(base_path, "runs", "experiment-{}".format(_run._id)))
    model_path = os.path.join(fs_observer.dir, "best_model.pth")

    outputs_path = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

    s_model = SegmentorNet().cuda()
    c_model = CriticNet().cuda()

    s_optimizer = optim.Adam(s_model.parameters(), lr=lr, betas=(beta1, 0.999))
    c_optimizer = optim.Adam(c_model.parameters(), lr=lr, betas=(beta1, 0.999))

    augmentation = []
    train_input_preprocess = []
    train_target_preprocess = []
    val_input_preprocess = []
    val_target_preprocess = []

    train_dataset = SegANDataset(train_fpath, augmentation)
    val_dataset = SegANDataset(val_fpath, augmentation)

    dataloaders = {
        "train": data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8),
        "validation": data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8)
    }

    best_IoU = 0.0
    s_model.train()
    for epoch in range(n_epochs):
        progress_bar = tqdm(dataloaders["train"], desc="Epoch {} - train".format(epoch))

        s_losses = []
        s_losses_joint = []
        c_losses = []
        dices = []

        for i, (inputs, targets, fname) in enumerate(progress_bar):
            pass
