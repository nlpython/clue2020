import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
from torch.utils.data import DataLoader
from processer.utils import read_json, build_corpus
from models.crf import CRF

class BertCrfForNer(BertPreTrainedModel):

    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, bidirectional=True,
                            batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores



# BERT + BiLSTM + CRF in clue
class BERTNER(BertPreTrainedModel):

    def __init__(self, bert_path, config):
        super(BERTNER, self).__init__(bert_path)
        self.bert = BertModel(bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.lstm = nn.LSTM(config.bert_hidden_size, config.lstm_hidden_size // 2, bidirectional=True, 
                            batch_first=True, num_layers=config.lstm_layer_num, dropout=config.lstm_dropout)
        self.classifier = nn.Linear(config.lstm_hidden_size, config.tag_num)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.dropout)

        # self.init_weights()

        self.crf = CRF(config.tag_num, batch_first=True)


    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids, attention_mask)[0]

        out, _ = self.lstm(outputs)
        out = self.act(out)
        out = self.classifier(out)
        
#         out = self.drop(out)

        return out

    def compute_loss(self, input_ids, attention_mask, y):

        out = self.forward(input_ids, attention_mask)
        # 最小化损失
        loss = -self.crf(out, y)
        return loss

    def decode(self, input_ids, attention_mask):
        out = self.forward(input_ids, attention_mask)
        predicted_index = self.crf.decode(out)
        return predicted_index

# loss = -models.crf(out,targets.long().to(parameter['device']))



if __name__ == '__main__':
    text_lists, tag_lists = read_json()
    tag_ids, tag2id, id2tag = build_corpus(tag_lists)
    # print(tag_ids[:10])

    tokenizer = BertTokenizer.from_pretrained('../../../transformers_model/bert-base-chinese')
    dataset = NERDataset(text_lists, tag_ids, tokenizer)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)
    # batch = next(iter(dataloader))
    # print(batch)

    # init models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BERTNER.from_pretrained('./bert-base-chinese', lstm_hidden_size=1024, tag_num=32, dropout=0.1)
    batch = next(iter(dataloader))

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']


    # init optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    # print(param_optimizer)
    for n, p in param_optimizer:
        print(n)

    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': 5e-5},
                                    {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr' : 2e-3}]

    # [ 0.0262,  0.0109, -0.0187,  ...,  0.0903,  0.0028,  0.0064]








