from typing import Dict, Tuple, Optional
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import visdom

import utils.log_util as lutil
import utils.exp_util as eutil


def tree_to(device, tree:dict) -> dict:
    '''
    Move values of given tree to given device.
    '''
    new_tree = {}
    for k, v in tree.items():
        new_tree.update({k: v.to(device)})
    return new_tree


def get_writer(method, work_dir: Optional[str]=None, port: Optional[int]=None):
    if method == 'tensorboard':
        return SummaryWriter(work_dir)
    elif method == 'visdom':
        return visdom.Visdom(port=port)
    else:
        raise NotImplementedError

class Method():
    """
    General class for training and testing
    """

    def __init__(self,
                 network: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 config: Dict,
                 ) -> None:
        """__init__.

        Args:
            network (nn.Module): neural network to trian and test
            dataloader (DataLoader): torch-based dataloader
            device (torch.device): torch device
            config (Dict): configurations for training, testing, and logging

        Returns:
            None:
        """
        self.network = network.to(device)
        self.dataloader = dataloader
        self.device = device
        self.config = config
        
        # setting for logging and monitoring
        self.monitor = 'tensorboard'
        self.exp_dir = eutil.create_exp_dir(work_dir='./work_dir')
        self.writer = get_writer(self.monitor, self.exp_dir)
        self.logger = lutil.get_logger(name="Orangingk")
        eutil.save_exp_condition(self.exp_dir, config)

        self.configure_optimizer()


    def _next_batch(self) -> Dict[str, torch.Tensor]:
        try:
            data = next(self.iter_loader)
            if isinstance(data, list):
                data = data[0]  # assume that dataset return image as first value.
        except (StopIteration, AttributeError):
            self.iter_loader = iter(self.dataloader)
            return self._next_batch()
        return {'img': data}


    def configure_optimizer(self):
        """Create optimizer."""
        self.optim = torch.optim.AdamW(self.network.parameters(), lr=1e-4)


    def save_checkpoint(self, current_iteration: int) -> None:
        """Save checkpoint. 

        Args:
            current_iteration (int): current_iteration
        """
        try:
            save_name = f'checkpoint_{current_iteration}.pt'
            save_path = self.exp_dir.joinpath(save_name)
            
            states = {
                "net": self.network.state_dict(),
                "optim":self.optim.state_dict()
            }
            torch.save(states, str(save_path))
            self.logger.info(f"Checkpoint is saved at {save_path}")
        except Exception as exc:
            self.logger.warning(f"Checkpoint could not be saved, since {exc}")


    def load_checkpoint(self, ckpt_path: Path) -> None:
        """Load checkpoint.

        Args:
            ckpt_path (Path): ckpt_path
        """
        try:
            state_dict = torch.load(str(ckpt_path))
            self.network.load_state_dict(state_dict['net'], strict=True)
            self.optim.load_state_dict(state_dict['optim'], strict=True)
            self.logger.info(f"Checkpoint is loaded from {ckpt_path}")
        except Exception as exc:
            self.logger.warning(f"Problem in loading checkpoint: {exc}")

    @lutil.record
    def train_step(self, batch: Dict[str, torch.Tensor], num_iter: int) -> Tuple[Dict[str, torch.Tensor], float]:
        return {'img': torch.ones((1,1,32,32), device=self.device)}, torch.Tensor([0.0])

    def train(self, num_iterations: int):
        self.logger.info(f'Training for {num_iterations} iterations start.')
        self.logger.info(f'Working directory: {self.exp_dir}')
        
        pbar = tqdm(range(num_iterations))
        for i in pbar:
            batch = self._next_batch()
            batch = tree_to(self.device, batch)
            _, loss = self.train_step(batch, num_iter=i)

            pbar.set_postfix({'loss': loss.item()}, refresh=True)

            if (i+1)%self.save_freq == 0:
                self.save_checkpoint(i+1, snapshot=False)
        
        self.save_checkpoint(num_iterations, snapshot=False)

