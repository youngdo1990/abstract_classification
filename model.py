# -*- coding=utf-8 -*-
import torch
import pandas as pd
from tqdm import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
import numpy as np

import os
from sklearn.model_selection import train_test_split # 데이터셋 나누기 위한 라이브러리

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import random

import matplotlib.pyplot as plt
import argparse


# BATCH_SIZE = 1 # 3
# EPOCHS = 5
# TOKENIZER = 'allenai-specter2' # bert-base-uncased / biobert-v1.1 / allenai-specter / allenai/specter2
# PRETRAINED_MODEL = 'allenai-specter2' # bert-base-uncased / biobert-v1.1 / allenai-specter / allenai/specter2
# DATASET_PATH = 'data/dataset.csv'


def prepare_dataset(dataset_path, tokenaizer):
    df = pd.read_csv(dataset_path)
    print(df.head())

    # df = df.iloc[:50]

    # print(len(df))
    categories = {}
    for i in tqdm(df.index):
        if df["Conference"][i] in categories:
            categories[df["Conference"][i]] += 1
        else:
            categories[df["Conference"][i]] = 1
    # print(categories)

    # print(df['Conference'].value_counts())
    possible_labels = df.Conference.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    # print("="*10)
    # print(label_dict)
    with open("label_dict", "w", encoding="utf-8", errors="ignore") as dict_file:
        dict_file.write(str(label_dict))
    df['label'] = df.Conference.replace(label_dict)


    X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                    df.label.values, 
                                                    test_size=0.15, 
                                                    random_state=42, 
                                                    stratify=df.label.values)


    df['data_type'] = ['not_set']*df.shape[0]

    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'

    df.groupby(['Conference', 'label', 'data_type']).count()


    tokenizer = BertTokenizer.from_pretrained(tokenaizer,
                                            do_lower_case=True)
                                            
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].Title.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'].Title.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )


    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)
    

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    return dataset_train, dataset_val, label_dict


# 모댈 서택
def model_load(pretrained_model, label_dict):
    model = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                        num_labels=len(label_dict),  # 레이블 수
                                                        output_attentions=False,
                                                        output_hidden_states=False)
    return model

    

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def evaluate(model, dataloader_val, predict=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':  batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2],
                }


        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val)
    true_vals = np.concatenate(true_vals, axis=0)
    
    predictions = np.concatenate(predictions, axis=0)
    
    
    if not predict:
        return loss_val_avg, predictions, true_vals
    else:
        return np.argmax(predictions, axis=1).flatten()



def train_model(model, dataset_train, dataset_val, epochs, batch_size, output):
    log_file = open("training logs.txt", "w", encoding="utf-8")
    train_metrics = []
    val = []

    dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(),
                    lr=1e-5, 
                    eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in tqdm(range(1, epochs+1)):
        
        model.train()
        
        loss_train_total = 0
        train_predictions, train_true_vals = [], []

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:

            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
            # Train metrics
            train_logits = outputs[1]
            train_logits = train_logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            train_predictions.append(train_logits)
            
            train_true_vals.append(label_ids)
        

        loss_train_avg = loss_train_total/len(dataloader_train) 
    
        train_predictions = np.concatenate(train_predictions, axis=0)
        train_true_vals = np.concatenate(train_true_vals, axis=0)
        train_f1 = f1_score_func(train_predictions, train_true_vals)
        # Train metrics
        
        if not os.path.exists(output):
            os.makedirs(output)

        torch.save(model.state_dict(), os.path.join(output, f'finetuned_BERT_epoch_{epoch}.model'))
            
        tqdm.write(f'\nEpoch {epoch}')
        log_file.write(f'Epoch {epoch}\n')
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        tqdm.write(f'F1 Score[Train] (Weighted): {train_f1}')
        log_file.write(f'Training loss: {loss_train_avg}\n')
        log_file.write(f'F1 Score[Train] (Weighted): {train_f1}\n')
        val_loss, predictions, true_vals = evaluate(model=model, dataloader_val=dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}\n')
        tqdm.write(f'F1 Score[Val] (Weighted): {val_f1}\n')
        log_file.write(f'Validation loss: {val_loss}\n')
        log_file.write(f'F1 Score[Val] (Weighted): {val_f1}\n')
        log_file.write(f'='*100)
        log_file.write("\n")  
        train_metrics.append(train_f1)
        val.append(val_f1)

    plt.plot(train_metrics, color="red", label="Train")
    plt.plot(val, color="blue", label="Validation")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.grid(True)
    plt.savefig("validation.png")
    # plt.show()
    log_file.close()
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="사전 학습 모델 경로", default="./allenai-specter")
    parser.add_argument("--tokenaizer", help="토크나이저 경로", default="./allenai-specter")
    parser.add_argument("--dataset", help="데이터셋 경로", default="./data/dataset.csv")
    parser.add_argument("--epochs", help="에포크 수", default=5)
    parser.add_argument("--batch_size", help="에포크 수", default=3)
    parser.add_argument("--output", help="학습시킨 모델 저장할 경로", default="./data_volume")
    
    args = parser.parse_args()
    
    print(args.model)
    print(args.tokenaizer)
    print(args.dataset)
    print(args.epochs)
    print(args.output)
    dataset_train, dataset_val, label_dict = prepare_dataset(dataset_path=args.dataset, tokenaizer=args.tokenaizer)
    model = model_load(pretrained_model=args.model, label_dict=label_dict)
    train_model(model=model, dataset_train=dataset_train, dataset_val=dataset_val, epochs=int(args.epochs), batch_size=int(args.batch_size), output=args.output)
    return


if __name__ == "__main__":
    main()