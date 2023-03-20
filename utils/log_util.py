"Module for logging and monitoring. Support tensorboard and visdom"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch


# ============ Logging on terminal =============

# From https://github.com/tqdm/tqdm/issues/313#issuecomment-347960988.
# To pass tqdm bar to logger. Awesome.
class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # , file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def get_logger(name: str):
    logger = logging.getLogger(name=name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    tqdm_handler = TqdmHandler()
    tqdm_handler.setFormatter(formatter)
    logger.addHandler(tqdm_handler)

    return logger


# ============= Monitoring methods =================

MONITOR = {}

def register_monitor(name: str):
    def wrapper(func):
        if MONITOR.get(name) is not None:
            raise NameError(f'Monitor named {name} is already registered!')
        MONITOR[name] = func
        return func
    return wrapper

def get_monitor(name: str):
    if MONITOR.get(name) is None:
        raise NameError(f'Monitor called {name} does not exist.')
    return MONITOR[name]

# use this wrapper to train_step
def record(func):
    def wrapper(*args, **kwargs):
        monitor = get_monitor(args[0].monitor)
        return monitor(func)(*args, **kwargs)
    return wrapper

# ============= Visdom ===============

# global dictionary to manage visdom windows
windows = {}

@register_monitor(name='visdom')
def record_visdom(func):
    def wrapper(*args, **kwargs):
        result, loss = func(*args, **kwargs)
        
        try:
            x = torch.tensor([kwargs['num_iter']])

            # window initialize
            if kwargs['num_iter'] == 0:
                line_win = args[0].writer.line(Y=loss.unsqueeze(0), 
                                               X=x, 
                                               opts=dict(title=f'{func.__name__}/loss'))
                windows['line'] = line_win
                
                images = list(result.values())
                images = list(map(lambda x: x[0], images))
                img_win = args[0].writer.images(torch.stack(images, dim=0))
                windows['img'] = img_win
                
            if kwargs['num_iter'] % args[0].scalar_freq == 0:
                args[0].writer.line(Y=loss.unsqueeze(0), X=x, win=windows['line'], update='append')
            
            if kwargs['num_iter'] % args[0].image_freq == 0:
                images = list(result.values())
                images = list(map(lambda x: x[0], images))
                titles = list(result.keys())
                args[0].writer.images(torch.stack(images, dim=0), win=windows['img'])
            
        except Exception as e:
            raise e

        return result, loss
    return wrapper



# ============= Tensorboard ============== 

def shape_check(images: list):
    '''
    Check shape of images in given list.
    We assume that shape of all images are the same.
    Given image's shape will be converted to (H, W, C).
    '''

    if images[0].ndim == 4:
        if images[0].shape[1] in [1, 3]:  # (B, C, H, W) cases
            images = list(map(lambda x: x[0].permute(1,2,0), images))
        elif images[0].shape[-1] in [1, 3]:  # (B, H, W, C) cases
            images = list(map(lambda x: x[0], images))
        else:
            raise NotImplementedError
    elif images[0].ndim == 3:
        if images[0].shape[0] in [1, 3]:  # (C, H, W) cases
            images = list(map(lambda x: x.permute(1,2,0), images))
        elif images[0].shape[-1] in [1, 3]:  # (H, W, C) cases
            pass
        else:
            raise NotImplementedError
    return images
            
    
def plot_multiple_figures(images, titles:list=None, nrows:int=1):
    matplotlib.use('Agg')
    assert len(images) % nrows == 0, "currently, only support square plots."
    n_cols = len(images) // nrows

    images = shape_check(images)
    fig = plt.figure(figsize=(4*n_cols, 4*nrows))
     
    for idx in range(1, len(images)+1):
        ax = fig.add_subplot(nrows, n_cols, idx, xticks=[], yticks=[])
        plt.imshow(images[idx-1].detach().cpu())
        if titles is not None:
            ax.set_title(titles[idx-1])

    return fig

def plot_compare_figure(input, output, label):
    fig = plt.figure(figsize=(12, 4))
    
    images = [input, output, label]
    images = list(map(lambda x: x[0].permute(1,2,0), images))
    titles = ['input', 'output', 'label']
    
    for idx in np.arange(3):
        ax = fig.add_subplot(1, 3, idx+1, xticks=[], yticks=[])
        plt.imshow(images[idx])
        ax.set_title(titles[idx])
    
    return fig

@register_monitor(name='tensorboard')
def record_tensorboard(func):
    def wrapper(*args, **kwargs):
        result, loss = func(*args, **kwargs)
        
        try: 
            if kwargs['num_iter'] % args[0].scalar_freq == 0:
                # record loss
                args[0].writer.add_scalar(
                        f'{func.__name__}/loss',
                        loss.item(),
                        kwargs['num_iter']
                        )

            if kwargs['num_iter'] % args[0].image_freq == 0:
                # record output image
                images = list(result.values())
                titles = list(result.keys())
                args[0].writer.add_figure(
                        f'{func.__name__}/imgs',
                        plot_multiple_figures(images, titles),
                        kwargs['num_iter']
                        )
        except Exception as e:
            args[0].logger.warning(f"Cannot write tensorboard, since {e}.")
            
        return result, loss
    return wrapper
