# --- coding:utf-8 ---
# author: Cyberfish time:2021/4/25
import logging
import numpy as np
import pandas as pd
from utils.configs import args_parse
from utils.functions_utils import set_seed, load_model_and_parallel, get_batch_num
from utils.preprocess import Dataloader, KETEDataLoader, MyDataSet
from utils.model import MyModel
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
import time
from tqdm import tqdm

gpu_ids = '0'
label_class = ['FALSE', 'TRUE']
localtime = time.localtime(time.time())


def train(model, train_data, optimizer, scheduler):
    model.train()
    for batch_roberta_inputs in tqdm(train_data):
        input_ids = batch_roberta_inputs['input_ids'].squeeze().to(device)
        attention_mask = batch_roberta_inputs['attention_mask'].squeeze().to(device)

        loss = model(input_ids,
                     device,
                     attention_mask=attention_mask,
                     # token_type_ids=batch_token_type_ids 
                     )

        model.zero_grad()
        loss.backward()
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()


def evaluate(model, data_iterator, BATCH_NUM):
    model.eval()
    S, P, R = 1e-10, 1e-10, 1e-10
    G_dis, P_dis, S_dis = 1e-10, 1e-10, 1e-10
    G_che, P_che, S_che = 1e-10, 1e-10, 1e-10

    one_epoch = trange(BATCH_NUM)
    with torch.no_grad():
        for _ in one_epoch:
            batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_labels, y_type = next(data_iterator)

            y_pred = model(batch_input_ids,
                           # attention_mask=batch_attention_mask,
                           # token_type_ids=batch_token_type_ids
                           )
            _, y_pred = torch.max(y_pred, 1)
            y_pred = y_pred.cpu().numpy()
            y_true = batch_labels.cpu().numpy()
            P += (y_pred == 1).sum()
            R += (y_true == 1).sum()
            S += (y_pred & y_true).sum()

            # G_dis += (y_true & (y_type == 0)).sum()
            # G_che += (y_true & (y_type == 1)).sum()
            # P_dis += (y_pred & (y_type == 0)).sum()
            # P_che += (y_pred & (y_type == 1)).sum()
            # S_dis += (y_true & y_pred & (y_type == 0)).sum()
            # S_che += (y_true & y_pred & (y_type == 1)).sum()

    f1, pre, rec = 2 * S / (P + R), S / P, S / R
    # f1_dis, precision_dis, recall_dis = 2 * S_dis / (G_dis + P_dis), S_dis / P_dis, S_dis / G_dis
    # f1_che, precision_che, recall_che = 2 * S_che / (G_che + P_che), S_che / P_che, S_che / G_che

    result = {
        'f1': f1, 'precision': pre, 'recall': rec,
        # 'f1_dis': f1_dis, 'pre_dis': precision_dis, 'rec_dis': recall_dis,
        # 'f1_che': f1_che, 'pre_che': precision_che, 'rec_che': recall_che,
        # 'f1_in': f1_in, 'pre_in': precision_in, 'rec_in': recall_in,
        # 'f1_not': f1_not, 'pre_not': precision_not, 'rec_not': recall_not,
        # 'f1_in_dis': f1_in_dis, 'pre_in_dis': precision_in_dis, 'rec_in_dis': recall_in_dis,
        # 'f1_not_dis': f1_not_dis, 'pre_not_dis': precision_not_dis, 'rec_not_dis': recall_not_dis,
        # 'f1_in_che': f1_in_che, 'pre_in_che': precision_in_che, 'rec_in_che': recall_in_che,
        # 'f1_not_che': f1_not_che, 'pre_not_che': precision_not_che, 'rec_not_che': recall_not_che,
    }
    return result


def fit(model, device, train_data, optimizer, scheduler, args, val_data=None, test_data=None):
    with open('result/best_7月_f1.txt') as f:
        best_f1 = float(f.read())
    for epoch in range(1, args.epochs + 1):
        logger.info(f'Epoch {epoch}/{args.epochs}')

        # 模型训练
        logger.info(f'---------正在训练----------')
        train(model, train_data, optimizer, scheduler)

        logger.info(f'---------正在验证----------')
        BATCH_NUM = get_batch_num(val_data[0], args.batch_size)
        val_data_iterator = data_loader.data_iterator(val_data, device, BATCH_NUM, shuffle=False)
        val_result = evaluate(model, val_data_iterator, BATCH_NUM)
        logger.info('\t'.join('{}: {:.4%}'.format(k, v) for k, v in val_result.items()))

        logger.info(f'---------正在测试----------')
        BATCH_NUM = get_batch_num(test_data[0], args.batch_size)
        test_data_iterator = data_loader.data_iterator(test_data, device, BATCH_NUM, shuffle=False)
        test_result = evaluate(model, test_data_iterator, BATCH_NUM)
        logger.info('\t'.join('{}: {:.4%}'.format(k, v) for k, v in test_result.items()))
        if test_result['f1'] > best_f1:
            best_f1 = test_result['f1']
            with open('result/best_7月_f1.txt', 'w') as f:
                f.write(str(best_f1))
            torch.save(model.state_dict(), 'result/best_7月.state')

        result = [v for v in test_result.values()]
        result.extend([val_result['f1'], val_result['precision'], val_result['recall']])
        his.append(result)
        header = list(test_result.keys())
        header.extend(['val_f1', 'val_pre', 'val_rec'])
        pd.DataFrame(his).to_excel('result/7_%d日%d点%d分his_log.xlsx'
                                   % (
                                       localtime.tm_mday,
                                       localtime.tm_hour,
                                       localtime.tm_min),
                                   header=header
                                   )


if __name__ == '__main__':
    his = []
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    args = args_parse()

    set_seed(args.seed)

    logger.info(f'----------加载数据---------')
    train_set = MyDataSet(args.train_file, args)
    train_data = DataLoader(train_set, batch_size=21)

    data_loader = Dataloader(args, label_class)
    val_data = data_loader.load_data(args.dev_file)
    test_data = data_loader.load_data(args.test_file)

    logger.info(f'----------加载模型---------')
    model = MyModel(args)

    model, device = load_model_and_parallel(model, gpu_ids)

    # TODO: 没有进行weight_decay

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
    warmup_steps = len(train_set) // 21
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.epochs * warmup_steps)

    logger.info(f'----------开始训练---------')
    fit(model, device, train_data, optimizer, scheduler, args,
        val_data, test_data
        )
