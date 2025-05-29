import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import pandas as pd
import math
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

print("CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
dataset = []
with open('News_Category_Dataset_v3.json', 'r') as file:
    for line in file:
        line_dict = json.loads(line)
        dataset.append([line_dict['headline'], line_dict['category']])
'''

file_path = 'data/netflix_reviews.csv'
df = pd.read_csv(file_path)

# Ensure the DataFrame has the required columns
if 'content' in df.columns and 'score' in df.columns:
    # Extract the 'content' and 'score' columns
    content_score_df = df[['content', 'score']]

    # Drop rows with missing 'content'
    content_score_df = content_score_df.dropna(subset=['content'])

    # Ensure 'content' column is of type string
    content_score_df['content'] = content_score_df['content'].astype(str)

    # Convert the DataFrame to a list of lists
    dataset = content_score_df.values.tolist()
else:
    print("The required columns 'content' and 'score' are not present in the dataset.")
    dataset = []


def preprocess_data(l):
    '''
    Accepts a list of lists with two strings, removes non alphabetic characters
    and converts uppercase into lowercase letters.
    '''

    l_length = len(l)  # length of dataset
    for i in range(l_length):
        s = l[i][0]
        s = s.lower()  # convert string to lowercase
        s_length = len(s)  # length of string to be modified
        new_s = ''  # new string to replace
        for j in range(s_length):
            if (97 <= ord(s[j]) <= 122 or ord(s[j]) == 32):  # if char is lowercase or a space
                new_s += s[j]  # append char to new_s

        l[i][0] = new_s  # replace string with new one


def select_rows(l, n, cat_indices):
    '''
    Takes in list, number of rows and set of categories to select, shuffles, then selects
    n rows per category
    '''

    class_num = len(cat_indices)
    list_to_shuffle = l.copy()  # assign l so the list doesn't get shuffled
    random.shuffle(list_to_shuffle)  # shuffle
    sorted_records = [[] for _ in range(class_num)]  # create a list of empty lists for each category
    remaining_list = []  # list without the selected data
    records = []  # final return record

    for row in list_to_shuffle:
        cat_index = cat_indices[row[1]]  # index of category
        if len(sorted_records[cat_index]) < n:
            sorted_records[cat_index].append(row)
        else:
            remaining_list.append(row)

    for cat in sorted_records:
        records += cat

    return records, remaining_list


def set_of_category(l):
    '''
    Takes a dataset, produces a set containing all unique categories of the dataset
    '''

    cat_of_set = set()
    for record in l:
        cat_of_set.add(record[1])

    return cat_of_set


def tf_idf(l, train_test_tuple):
    '''
    Takes in a dataset and shuffles. Depending on which value is not None,
    test_train_tuple has two values of number of test dataset per category
    and training set per category. Calculates tf-idf matrix for training and
    testing dataset, returns both matrices along with two lists of categories
    for each row in training and testing datasets.
    '''

    shuffled_dataset = l.copy()  # copy list so it doesn't get shuffled
    random.shuffle(shuffled_dataset)

    cat_set = set_of_category(shuffled_dataset)  # get a set of categories
    dict_of_cat_indices = {}  # dictionary of cat as key and index as value
    i = 0  # initialize index
    for cat in cat_set:
        dict_of_cat_indices[cat] = i
        i += 1

    train_num_per_cat, test_num_per_cat = train_test_tuple  # unpack the tuple
    train_set, remaining_set = select_rows(shuffled_dataset, train_num_per_cat,
                                           dict_of_cat_indices)  # get the train set and remaining set
    test_set, _ = select_rows(shuffled_dataset, test_num_per_cat, dict_of_cat_indices)  # get the test set

    train_words = [x[0] for x in train_set]  # words from train_set
    train_set_categories = [dict_of_cat_indices[x[1]] for x in train_set]  # cats from train_set
    test_words = [x[0] for x in test_set]  # words from train_set
    test_set_categories = [dict_of_cat_indices[x[1]] for x in test_set]  # cats from train_set

    vectorizer = TfidfVectorizer()  # initialize TfidfVectorizer
    vectorizer.fit(train_words)  # fit vectorizer to train_set
    vocab_dict = vectorizer.vocabulary_  # retrieve vocab dict
    vectorizer = TfidfVectorizer(vocabulary=vocab_dict)  # reinitialize with vocab_dict

    # calculate tfidf matrices
    train_tfidf = vectorizer.fit_transform(train_words)
    test_tfidf = vectorizer.transform(test_words)

    # convert matrices into dense
    train_tfidf_dense = train_tfidf.toarray()
    test_tfidf_dense = test_tfidf.toarray()

    # convert dense to PyTorch tensor and put it into GPU
    train_tfidf_tensor = torch.tensor(train_tfidf_dense, dtype=torch.float32, device=device)
    test_tfidf_tensor = torch.tensor(test_tfidf_dense, dtype=torch.float32, device=device)

    return train_tfidf_tensor, test_tfidf_tensor, train_set_categories, test_set_categories


class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TwoLayerNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 2048)
        self.activation = torch.nn.ReLU()
        self.linear2 = nn.Linear(2048, 1024)
        self.activation = torch.nn.ReLU()
        self.linear3 = nn.Linear(1024, 256)
        self.activation = torch.nn.ReLU()
        self.linear4 = nn.Linear(256, output_dim)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.softmax(x)
        return x


preprocess_data(dataset)
train_tensor, test_tensor, train_cats, test_cats = tf_idf(dataset, (500, 4000))

train_tensor_normalized = nn.functional.normalize(train_tensor)
test_tensor_normalized = nn.functional.normalize(test_tensor)

# PCA once for q value
U, S, V = torch.pca_lowrank(train_tensor_normalized, center=False)


# Calculate explained variance ratio
explained_variance_ratio = S ** 2 / torch.sum(S ** 2)

# Calculate cumulative explained variance ratio
cumulative_explained_variance = torch.cumsum(explained_variance_ratio, dim=0)

# Determine the number of components to retain 95% variance
q = torch.where(cumulative_explained_variance >= 0.95)[0][0].item() + 1

# Use the top q principal components to transform the training data
train_principal_components = torch.matmul(train_tensor_normalized, V[:, :q])

# Use the same top q principal components to transform the test data
test_principal_components = torch.matmul(test_tensor_normalized, V[:, :q])

print("Train Principal Components shape:", train_principal_components.shape)
print("Test Principal Components shape:", test_principal_components.shape)
print(f"Number of components to retain 95% variance: {q}")

# Check device of the principal components

# convert category lists to one hot vectors
num_cat = np.max(train_cats) + 1
y_train = np.eye(num_cat)[train_cats]
y_test = np.eye(num_cat)[test_cats]

y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

# Check device of the labels

# Note: Use train_principal_components and test_principal_components in the DataLoader
dataset = TensorDataset(train_principal_components, y_train)

train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

input_dim = train_principal_components.shape[1]  # Note the change here
model = TwoLayerNN(input_dim, output_dim=num_cat).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 250

for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, labels = batch

        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze()  # Adjust the output shape
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

model.eval()

with torch.no_grad():
    test_principal_components = test_principal_components.to(device)

    test_outputs = model(test_principal_components)
    _, predicted = torch.max(test_outputs, 1)

    test_cats_tensor = torch.tensor(test_cats, device=device)
    correct_predictions = (predicted == test_cats_tensor).sum().item()
    accuracy = correct_predictions / len(test_cats)

print(f'Accuracy: {accuracy:.4f}')
