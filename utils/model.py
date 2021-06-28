# --- coding:utf-8 ---
# author: Cyberfish time:2021/4/25
from torch import nn
import torch
from transformers import RobertaConfig, RobertaModel
from torch.nn.utils.rnn import pad_sequence

class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(args.bert_class)
        self.main_dropout = nn.Dropout(args.dropout_rate)
        self.sim_dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, args.label_class)
        self.simmlp = nn.Linear(self.roberta.config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, device, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids
                               )
        sequence_output = outputs[1]
        """二分类loss"""
        main_out = self.main_dropout(sequence_output)
        main_out = self.classifier(main_out)
        main_loss_fn = nn.CrossEntropyLoss()
        main_loss = main_loss_fn(main_out, torch.tensor([1]+[0]*20).to(device))
        """对比loss"""
        sim_out = self.sim_dropout(sequence_output)
        sim_out = self.simmlp(sim_out)
        sim_out = sim_out.reshape(1, 21)
        sim_loss_fn = nn.CrossEntropyLoss()
        sim_loss = sim_loss_fn(sim_out, torch.tensor([0]).to(device))

        loss = main_loss + sim_loss
        return main_loss



        return outputs