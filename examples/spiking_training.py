import logging
import sys

sys.path.append("..")
# if you encounter ModuleNotFoundError: No module named 'naslib'
# then you need to add the naslib directory to your PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/path/to/naslib or export PYTHONPATH=$PYTHONPATH:$pwd

from naslib.optimizers import DARTSOptimizer
from naslib.search_spaces import NasBenchSpikingASRSearchSpace #NasBenchASRSearchSpace #NasBench301SearchSpace#NasBenchNLPSearchSpace
from naslib.utils import set_seed, setup_logger, get_config_from_args, create_exp_dir

from naslib import utils
from naslib.defaults.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch


search_space = NasBenchSpikingASRSearchSpace()  #NasBenchNLPSearchSpace()

config = utils.get_config_from_args()
config.save_arch_weights = True
config.plot_arch_weights = True
config.save = "{}/{}/{}/{}/{}".format(
    config.out_dir, config.search_space, config.dataset, config.optimizer, config.seed
)
create_exp_dir(config.save)
create_exp_dir(config.save + "/search")  # required for the checkpoints
create_exp_dir(config.save + "/eval")

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose


optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)  
optimizer.before_training() # loads the graph to the gpu


trainer = Trainer(optimizer, config)
# trainer.search(summary_writer=SummaryWriter("runs/shd"))
with torch.autograd.set_detect_anomaly(True):
    trainer.search(summary_writer=SummaryWriter("runs/shd"))
trainer.evaluate()


# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import DataLoader
# from torch.utils.data import ConcatDataset
# from spiking_data import get_numpy_datasets
# train_datasets, test_datasets = get_numpy_datasets("shd", n_inp=100)
# train_dataset = ConcatDataset(train_datasets)
# test_dataset = ConcatDataset(test_datasets)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# search_space.to(device)

# config = utils.AttrDict()
# config.dataset = 'shd'
# config.resume = False
# config.search = utils.AttrDict()
# config.search.grad_clip = None
# config.search.checkpoint_freq = 10
# config.search.learning_rate = 0.01
# config.search.momentum = 0.1
# config.search.weight_decay = 0.1
# config.search.arch_learning_rate = 0.01
# config.search.arch_weight_decay = 0.1
# config.search.tau_max = 10
# config.search.tau_min = 1
# config.search.epochs = 2
# config.search.seed = 42

# epochs = 50
# lr = 0.025
# momentum = 0.9
# weight_decay = 3e-4

# architect = DARTSOptimizer(config)
# optimizer = optim.SGD(search_space.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
# scheduler = CosineAnnealingLR(optimizer, epochs)

# for epoch in range(1, epochs + 1):
#     train_loss = train(train_loader, optimizer, device)
#     valid_acc = test(test_loader, search_space, device)
#     scheduler.step()
#     print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Validation accuracy = {valid_acc:.4f}")
