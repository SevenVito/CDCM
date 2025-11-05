import torch
import os
import transformers
import nltk
import preprocessing as dp
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, AlbertTokenizer, BartTokenizer, RobertaTokenizer, \
    BertweetTokenizer


def convert_data_to_ids(tokenizer, target, text, label, config, rev_flag, idx,fre_dict):

    if rev_flag=="fan":
        encoded_dict={
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": []
        }                                           #

        for tar, sent, id_list in zip(target, text, idx):
            flag=True
            for idx in range(len(sent)):

                if flag :
                    token = [0]
                    flag = False
                if idx  not in id_list and idx<len(sent):
                    token.extend([64000] * len(tokenizer.encode(sent[idx], add_special_tokens=False)))

                else:
                    token_start = tokenizer.encode(sent[idx], add_special_tokens=False)
                    token.extend(token_start)

            token.append(2)
            token.append(2)

            target = ' '.join(tar)

            tarr = tokenizer.encode(target, add_special_tokens=False)
            for i in tarr:
                token.append(i)
            token.append(2)
            len_pad=int(config['max_tok_len'])-len(token)
            token.extend([1] * len_pad )


            if len(token)>128:
                token=token[:128]
            encoded_dict["input_ids"].append(token)

            attention_mask = [1] * len(token)
            attention_mask.extend([0]* len_pad )

            if len(attention_mask)>128:
                attention_mask=attention_mask[:128]
            encoded_dict["attention_mask"].append(attention_mask)
            encoded_dict["token_type_ids"].append([0] * int(config['max_tok_len']))

        encoded_dict['gt_label'] = label

    elif rev_flag=="yuan":
        concat_sent = []
        for tar, sent in zip(target, text):
            concat_sent.append([' '.join(sent), ' '.join(tar)])

        encoded_dict = tokenizer.batch_encode_plus(
            concat_sent,
            add_special_tokens=True,
            max_length=int(config['max_tok_len']),
            padding='max_length',
            return_attention_mask=True,
            truncation='longest_first',
        )
        encoded_dict['gt_label'] = label

    elif rev_flag=="fan_2":
        concat_sent = []
        for tar, sent in zip(target, text):
            masked_train = ['<mask>' if word not in fre_dict else word for word in sent]
            concat_sent.append([' '.join(masked_train), ' '.join(tar)])

        encoded_dict = tokenizer.batch_encode_plus(
            concat_sent,
            add_special_tokens=True,
            max_length=int(config['max_tok_len']),
            padding='max_length',
            return_attention_mask=True,
            truncation='longest_first',
        )
        encoded_dict['gt_label'] = label

    return encoded_dict

def data_helper_bert(x_train_all, x_val_all, x_test_all, model_select, config, rev_flag,fre_dict):
    print('Loading data')

    x_train, y_train, x_train_target, rev_idx_train = x_train_all[0], x_train_all[1], x_train_all[2], x_train_all[3]
    x_val, y_val, x_val_target, rev_idx_val = x_val_all[0], x_val_all[1], x_val_all[2], x_val_all[3]
    x_test, y_test, x_test_target, rev_idx_test = x_test_all[0], x_test_all[1], x_test_all[2], x_test_all[3]


    if model_select == 'Bertweet':
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",
                                                  local_files_only=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

    train_encoded_dict = convert_data_to_ids(tokenizer, x_train_target, x_train, y_train, config, rev_flag,
                                             rev_idx_train,fre_dict)
    val_encoded_dict = convert_data_to_ids(tokenizer, x_val_target, x_val, y_val, config, rev_flag, rev_idx_val,fre_dict)
    test_encoded_dict = convert_data_to_ids(tokenizer, x_test_target, x_test, y_test, config, rev_flag, rev_idx_test,fre_dict)

    trainloader, y_train = data_loader(train_encoded_dict, int(config['batch_size']), model_select, 'train')
    valloader, y_val = data_loader(val_encoded_dict, int(config['batch_size']), model_select, 'val')
    testloader, y_test = data_loader(test_encoded_dict, int(config['batch_size']), model_select, 'test')
    #trainloader2, y_train2 = data_loader(train_encoded_dict, int(config['batch_size']), model_select, 'train2')

    return (trainloader, valloader, testloader," trainloader2"), (y_train, y_val, y_test, "y_train2")


def data_loader(x_all, batch_size, model_select, mode):
    x_input_ids = torch.tensor(x_all['input_ids'], dtype=torch.long)
    x_atten_masks = torch.tensor(x_all['attention_mask'], dtype=torch.long)
    y = torch.tensor(x_all['gt_label'], dtype=torch.long)
    if model_select == 'Bert':
        x_seg_ids = torch.tensor(x_all['token_type_ids'], dtype=torch.long)
        tensor_loader = TensorDataset(x_input_ids, x_atten_masks, x_seg_ids, y)
    else:
        tensor_loader = TensorDataset(x_input_ids, x_atten_masks, y)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

    return data_loader, y


def sep_test_set(input_data,p_v, pv,dataset):

    if dataset == 'pstance':
        data_list = [input_data[:777], input_data[777:1522], input_data[1522:2157]]
        data_rev_list = [p_v[:777],p_v[777:1522],p_v[1522:2157]]

        data_rev_1 =[pv[:777],pv[777:1522],pv[1522:2157]]
    elif dataset == 'covid19':
        data_list = [input_data[:200], input_data[200:400], input_data[400:600], input_data[600:800]]
        data_rev_list = [p_v[:200],p_v[200:400],p_v[400:600],p_v[600:800]]
        data_rev_1 = [pv[:200],pv[200:400],pv[400:600],pv[600:800]]

    return data_list ,data_rev_list,data_rev_1
