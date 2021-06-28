# --- coding:utf-8 ---
# author: Cyberfish time:2021/4/25
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='data/NCBI_train_20.csv')
    parser.add_argument('--dev_file', default='../../sources/NCBI/my_develop.txt')
    parser.add_argument('--test_file', default='../../sources/NCBI/my_test.txt')

    # parser.add_argument('--train_file', default='../../sources/BC5CDR/my_Train_20.txt')
    # parser.add_argument('--dev_file', default='../../sources/BC5CDR/my_Development.txt')
    # parser.add_argument('--test_file', default='../../sources/BC5CDR/my_Test.txt')

    parser.add_argument('--seed', default=2021)
    parser.add_argument('--max_lenth', default=512)
    parser.add_argument('--bert_class', default='roberta-base')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--dropout_rate', default=0.1)
    parser.add_argument('--label_class', default=2)
    parser.add_argument('--learning_rate', default=1e-5)
    parser.add_argument('--epochs', default=15)

    return parser.parse_args()