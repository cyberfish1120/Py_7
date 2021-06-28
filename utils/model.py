# --- coding:utf-8 ---
# author: Cyberfish time:2021/4/25
from torch import nn
from transformers import RobertaConfig, RobertaModel
from torch.nn.utils.rnn import pad_sequence

class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(args.bert_class)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, args.label_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        sequence_output = outputs[1]
        outputs = self.dropout(sequence_output)
        outputs = self.classifier(outputs)
        return outputs