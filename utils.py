import os
from datetime import datetime

import torch
from torch.utils import tensorboard


class IO:
    def __init__(self, base_dir, checkpoint_dir, logdir):
        self.base_dir = base_dir
        self.checkpoint_dir = os.path.join(self.base_dir, checkpoint_dir)
        self.logdir = os.path.join(self.base_dir, logdir, datetime.now().strftime("%Y%m%d%H%M%S"))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)

        self.writer = tensorboard.SummaryWriter(self.logdir, flush_secs=60)

    def plot(self, tag, values, step):
        self.writer.add_scalars(tag, values, step)

    def __del__(self):
        self.writer.close()

    def load(self, model, optimizer):
        if len(os.listdir(self.checkpoint_dir)) > 0:
            latest = sorted(os.listdir(self.checkpoint_dir))[-1]
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, latest))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            return checkpoint['steps'], checkpoint['epoch']
        return 0, 0

    def checkpoint(self, steps, epoch, model, optimizer):
        latest = f'%s.pt' % datetime.now().strftime("%Y%m%d%H%M%S")
        torch.save({
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, latest))

    def save(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.base_dir, name))
