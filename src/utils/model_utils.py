import torch
from transformers import AdamW
import modeling


def model_setup(num_labels, model_select, device, config, dropout):
    
    if model_select == 'Bert':
        print("BERT is used as the stance classifier.")

        model = modeling.bert_classifier(num_labels, model_select, dropout).to(device)

        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False


        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
        
    elif model_select == 'Bertweet':
        print("BERTweet is used as the stance classifier.")
        model = modeling.roberta_large_classifier(num_labels, model_select, dropout).to(device)      
        for n, p in model.named_parameters():
            if "roberta.embeddings" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('roberta')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]     

    optimizer = AdamW(optimizer_grouped_parameters)
    
    return model, optimizer


def model_preds(raw_loaders,rev_loaders, rev_lod_1,model, device, loss_function,kwargs_1,beta):
    alpha=kwargs_1
    beta=beta
    preds_raw,preds_rev,preds_rev_1 = [],[],[]
    valtest_loss = []
    for (b_id, sample_batch_raw),(b_id_rev,sample_batch_rev),(b_id,sample_batch_rev_1) in zip(enumerate(raw_loaders),enumerate(rev_loaders),
                                                                                              enumerate(rev_lod_1)):
        dict_batch_raw = batch_fn(sample_batch_raw)
        dict_batch_rev = batch_fn(sample_batch_rev)
        dict_batch_rev_1 = batch_fn(sample_batch_rev_1)

        inputs_raw = {k: v.to(device) for k, v in dict_batch_raw.items()}
        inputs_rev = {k: v.to(device) for k, v in dict_batch_rev.items()}
        inputs_rev_1 ={k: v.to(device) for k, v in dict_batch_rev_1.items()}

        outputs_raw = model(**inputs_raw)
        outputs_rev = model(**inputs_rev)
        outputs_rev_1 = model(**inputs_rev_1)

        preds_raw.append(outputs_raw)
        preds_rev.append(outputs_rev)
        preds_rev_1.append(outputs_rev_1)

        loss = loss_function(outputs_raw, inputs_raw['gt_label'])

        valtest_loss.append(loss.item())

    return torch.cat(preds_raw,0),torch.cat(preds_rev,0),torch.cat(preds_rev_1,0),valtest_loss



def batch_fn(sample_batch):
    
    dict_batch = {}
    dict_batch['input_ids'] = sample_batch[0]
    dict_batch['attention_mask'] = sample_batch[1]
    dict_batch['gt_label'] = sample_batch[-1]
    if len(sample_batch) > 3:
        dict_batch['token_type_ids'] = sample_batch[-2]
    
    return dict_batch