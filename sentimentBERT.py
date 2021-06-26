import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

!pip install -q transformers
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/gdrive')
from pickle import load
np.random.seed(0)

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


df = pd.read_csv("datasets/IMDBDataset.csv")

# Select only the first 1500 records
df = df[:10000]

print(df.head)
print(df.isnull().sum())
print(df['sentiment'].describe())
print(df.columns)

texts = df['review'].values
print(type(texts))

print('>>', df[['sentiment']].groupby('sentiment').sum())

convert_to_int = lambda s: int(s=='positive')
sentiment = df['sentiment'].apply(convert_to_int).values

print('\n', sentiment)
print(type(sentiment))

sentences = ["[CLS] " + s + " [SEP]" for s in texts]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(s) for s in sentences]
inputs_ids = [tokenizer.convert_tokens_to_ids(tt) for tt in tokenized_texts]
print(tokenized_texts[0])
print(inputs_ids[0])

MAX_LEN = 128

input_ids = pad_sequences(inputs_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, sentiment,
                                                                                    random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                       random_state=2018, test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)


batch_size = 64

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


## DEFINE MODELS

models = {1: {"name": "bert-base-uncased",
              "param_no_decay": [],
              "learning_rate": 3e-5,
              "eps": 5e-8,
              "epochs": 6,
              "num_warmup_steps": 30}
          }

##  ITERATE ON MODEL

eval_by_model = []

for model in models.items():
    print(model)
    num, model_def = model

    model = BertForSequenceClassification.from_pretrained(model_def.get('name'), num_labels=2)
    model.cuda()

    param_optimizer = list(model.named_parameters())
    #  no_decay = ['bias', 'LayerNorm.weight']
    no_decay = model_def.get('param_no_decay')
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.1},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    epochs = model_def.get('epochs')
    # epochs = 32
    optimizer = AdamW(optimizer_grouped_parameters,
                      # lr = 5e-5,
                      lr=model_def.get('learning_rate'),
                      # eps = 1e-8
                      eps=model_def.get('eps')
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                # num_warmup_steps = 0,
                                                num_warmup_steps=model_def.get('num_warmup_steps'),
                                                num_training_steps=total_steps)

    ##  TRAIN MODEL

    t = []

    train_loss_set = []

    for _ in trange(epochs, desc="Epoch"):
        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs['loss']
            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            scheduler.step()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = logits['logits'].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    ##  PLOT LOSS

    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.show()

