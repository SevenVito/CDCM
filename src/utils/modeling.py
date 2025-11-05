import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, BertModel

import model_utils, model_calib, modeling


# BERT
class bert_classifier(nn.Module):

    def __init__(self, num_labels, model_select, dropout):
        super(bert_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']


        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        cls_hidden = last_hidden[0][:, 0]

        query = self.dropout(cls_hidden)

        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out


# BERTweet
class roberta_large_classifier(nn.Module):

    def __init__(self, num_labels, model_select, dropout):
        super(roberta_large_classifier, self).__init__()

        self.config = AutoConfig.from_pretrained('vinai/bertweet-base', local_files_only=True)
        self.roberta = AutoModel.from_pretrained('vinai/bertweet-base', local_files_only=True)
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        # CLS token
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)
        cls_hidden = last_hidden[0][:, 0]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out



class ban_updater(object):

    def __init__(self, **kwargs):

        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.s_gen = kwargs.pop("s_gen")
        self.last_model = None
        self.alpha = kwargs.pop("alpha")
        self.beta = kwargs.pop("beta")

    def update(self, inputs_raw, inputs_rev, inputs_rev_1,criterion, percent, T, args):

        self.optimizer.zero_grad()
        outputs_1 = self.model(**inputs_raw)
        outputs_2 = self.model(**inputs_rev)
        outputs_3 = self.model(**inputs_rev_1)
        if self.s_gen > 0:
            self.last_model.eval()
            with torch.no_grad():
                teacher_outputs = self.last_model(**inputs_raw).detach()
            loss_1 = self.kd_loss(outputs_1, inputs_raw['gt_label'], teacher_outputs, percent, T)
            loss_2 = self.kd_loss(outputs_2, inputs_rev['gt_label'], teacher_outputs, percent, T)
            loss_3 = self.kd_loss(outputs_3, inputs_rev['gt_label'], teacher_outputs, percent, T)
        else:
            loss_1 = criterion(outputs_1, inputs_raw['gt_label'])
            loss_2 = criterion(outputs_2, inputs_rev['gt_label'])
            loss_3 = criterion(outputs_3, inputs_rev['gt_label'])

        loss =loss_1 + self.alpha * loss_2 +self.beta *loss_3

        loss.backward()
        if args['clipgradient']:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        return loss.item()

    def register_last_model(self, weight, num_labels, model_select, device, dropout):

        if model_select == 'Bert':
            self.last_model = modeling.bert_classifier(num_labels, model_select, dropout).to(device)
        elif model_select == 'Bertweet':
            self.last_model = modeling.roberta_large_classifier(num_labels, model_select, dropout).to(device)
        self.last_model.load_state_dict(torch.load(weight))

    def get_calib_temp(self, valloader, rev_val,rev_val_1,y_val, device, criterion, dataset,alpha,beta,k):

        with torch.no_grad():
            preds, _ , _ , _ = model_utils.model_preds(valloader,rev_val ,rev_val_1,self.last_model, device, criterion,alpha,beta)
            T = model_calib.get_best_temp(preds, y_val, dataset,k)

        return T

    def kd_loss(self, outputs, labels, teacher_outputs, percent, T=1):

        KD_loss = T * T * nn.KLDivLoss(reduction='sum')(F.log_softmax(outputs / T, dim=1),
                                                        F.softmax(teacher_outputs / T, dim=1)) * \
                  (1. - percent) + nn.CrossEntropyLoss(reduction='sum')(outputs, labels) * percent

        return KD_loss
