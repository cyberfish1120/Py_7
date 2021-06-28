# --- coding:utf-8 ---
# author: Cyberfish time:2021/4/25
import random
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Todo 是否有必要


def load_model_and_parallel(model, gpu_ids):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """
    gpu_ids = gpu_ids.split(',')

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    model.to(device)

    if len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device


def get_batch_num(data, batch_size):
    if len(data) % batch_size == 0:
        BATCH_NUM = len(data) // batch_size
    else:
        BATCH_NUM = len(data) // batch_size + 1
    return BATCH_NUM
