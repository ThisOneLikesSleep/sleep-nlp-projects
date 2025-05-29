import numpy as np
import pandas as pd
import torch
import json
import transformers

import random
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast
from imblearn.over_sampling import SVMSMOTE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, BertTokenizerFast

print("CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('data/netflix_reviews.csv', usecols=['content', 'score'])

df = df.dropna(subset=['content', 'score'])

# Ensure all content are strings and scores are numeric
df = df[df['content'].apply(lambda x: isinstance(x, str))]
df = df[df['score'].apply(lambda x: isinstance(x, (int, float)))]

# Remove rows with empty strings in content
df = df[df['content'].str.strip() != '']

# check class distribution
print(df['score'].value_counts(normalize = True))

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 5)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x



def apply_BERT(df, test_ratio, val_ratio, batch_size):
    '''
    Splits dataset into training, testing, and
    '''

    train_text, remaining_text, train_labels, remaining_labels = train_test_split(df['content'],
                                                                                  df['score'],
                                                                                  random_state=random.randint(1, 10000),
                                                                                  test_size=(1 - test_ratio - val_ratio),
                                                                                  stratify=df['score'])
    val_text, test_text, val_labels, test_labels = train_test_split(remaining_text,
                                                                    remaining_labels,
                                                                    random_state=random.randint(1, 10000),
                                                                    test_size=(test_ratio / (test_ratio + val_ratio)),
                                                                    stratify=remaining_labels)

    # Ensure labels are continuous and start from 0
    label_mapping = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    train_labels = train_labels.map(label_mapping)
    val_labels = val_labels.map(label_mapping)
    test_labels = test_labels.map(label_mapping)

    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # get length of all the messages in the train set
    seq_len = [len(i.split()) for i in train_text]

    pd.Series(seq_len).hist(bins=30)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(seq_len, bins=30)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sequence Lengths in Training Set')
    plt.show()

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=35,
        padding='max_length',
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=35,
        padding='max_length',
        truncation=True
    )

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=35,
        padding='max_length',
        truncation=True
    )

    # convert lists to tensors

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False

    # pass the pre-trained BERT to our define architecture
    model = BERT_Arch(bert)

    # push the model to GPU
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # compute the class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels),
                                         y=train_labels)

    print('Class Weights: ', class_weights)

    # converting list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # number of training epochs
    epochs = 10

    def train():

        model.train()
        total_loss, total_accuracy = 0, 0

        # empty list to save model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(train_dataloader):

            # progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            sent_id, mask, labels = batch

            # clear previously calculated gradients
            model.zero_grad()

            # get model predictions for the current batch
            preds = model(sent_id, mask)

            # compute the loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            # add on to the total loss
            total_loss = total_loss + loss.item()

            # backward pass to calculate the gradients
            loss.backward()

            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters
            optimizer.step()

            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)

        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        # returns the loss and predictions
        return avg_loss, total_preds

    def evaluate():

        print("\nEvaluating...")

        # deactivate dropout layers
        model.eval()

        total_loss, total_accuracy = 0, 0

        # empty list to save the model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(val_dataloader):

            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            # push the batch to gpu
            batch = [t.to(device) for t in batch]

            sent_id, mask, labels = batch

            # deactivate autograd
            with torch.no_grad():

                # model predictions
                preds = model(sent_id, mask)

                # compute the validation loss between actual and predicted values
                loss = cross_entropy(preds, labels)

                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

        # compute the validation loss of the epoch
        avg_loss = total_loss / len(val_dataloader)

        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def test():
        print("\nTesting...")

        # Deactivate dropout layers
        model.eval()

        total_preds = []

        # Iterate over batches
        for step, batch in enumerate(test_dataloader):
            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

            # Push the batch to gpu
            batch = [t.to(device) for t in batch]

            sent_id, mask, labels = batch

            # Deactivate autograd
            with torch.no_grad():
                # Model predictions
                preds = model(sent_id, mask)
                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

        # Concatenate predictions
        total_preds = np.concatenate(total_preds, axis=0)

        return total_preds

    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train()

        # evaluate model
        valid_loss, _ = evaluate()

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'data/saved_weights.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    path = 'data/saved_weights.pt'
    model.load_state_dict(torch.load(path))

    # Wrap tensors
    test_data = TensorDataset(test_seq, test_mask, test_y)

    # DataLoader for test set without explicit sampler
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # get predictions for test data
    preds = test()
    preds = np.argmax(preds, axis=1)
    print(classification_report(test_y, preds))



apply_BERT(df, .2, .2, 128)