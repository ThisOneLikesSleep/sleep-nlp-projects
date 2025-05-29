from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import torch
import json
import random
import math
import gc
import numpy as np
import pandas as pd

print("CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = []
with open('data/News_Category_Dataset_v3.json', 'r') as file:
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

'''

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
            if (97 <= ord(s[j]) <= 122
                    or ord(s[j]) == 32):  # if char is lowercase or a space
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
    remaining_list = [] # list without the selected data
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


def delete_all_tensors():
    '''
    Clears all memory, and I didn't write this function
    '''

    def is_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return True
        if isinstance(obj, (list, tuple, dict)):
            return any(is_tensor(o) for o in obj)
        return False

    # Delete all tensors in global and local scope
    global_vars = globals()
    local_vars = locals()

    # Check for tensors in global scope
    for name, var in list(global_vars.items()):
        if is_tensor(var):
            del global_vars[name]

    # Check for tensors in local scope
    for name, var in list(local_vars.items()):
        if is_tensor(var):
            del local_vars[name]

    # Clear cache and force garbage collection
    torch.cuda.empty_cache()
    gc.collect()

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
    test_set, _ = select_rows(shuffled_dataset, test_num_per_cat,
                                                        dict_of_cat_indices)  # get the test set

    train_words = [x[0] for x in train_set]  # words from train_set
    train_set_categories = [dict_of_cat_indices[x[1]] for x in train_set]  # cats from train_set
    test_words = [x[0] for x in test_set]  # words from train_set
    test_set_categories = [dict_of_cat_indices[x[1]] for x in test_set]  # cats from train_set

    vectorizer = TfidfVectorizer()  # initialize TfidfVectorizer
    vectorizer.fit(train_words)  # fit vectorizer to train_set
    vocab_dict = vectorizer.vocabulary_  # retrieve vocab dict
    print(len(vocab_dict))
    vectorizer = TfidfVectorizer(vocabulary=vocab_dict)  # reinitialize with vocab_dict

    # calculate tfidf matrices
    train_tfidf = vectorizer.fit_transform(train_words)
    test_tfidf = vectorizer.transform(test_words)

    # convert matrices into dense
    train_tfidf_dense = train_tfidf.toarray()
    test_tfidf_dense = test_tfidf.toarray()

    # convert dense to PyTorch tensor and put it into GPU
    train_tfidf_tensor = torch.tensor(train_tfidf_dense, dtype=torch.float16,
                                      device=device)
    test_tfidf_tensor = torch.tensor(test_tfidf_dense, dtype=torch.float16,
                                     device=device)

    return train_tfidf_tensor, test_tfidf_tensor, train_set_categories, test_set_categories

def knn(train_mat, test_mat, train_cats, test_cats, k):
    '''
    takes train and test tf-idf matrices and list of
     categories from train and test datasets and k to perform knn algorithm.
     returns correct percentage and auroc score
    '''

    test_mat = test_mat.unsqueeze(1)
    cosine_similarity = torch.nn.functional.cosine_similarity(train_mat,
                                                              test_mat, dim=2)  # calculate cosine similarity
    _, indices = torch.sort(cosine_similarity, dim=1, descending=True)  # sort cosine similarity by greatest
    doc_index = [x[:k] for x in indices]  # pick k nearest training sample's indices for each row
    correct_classification = 0  # count of correct classifications
    target = []  # list of correct classifications
    prediction = []  # list of predicted classifications

    for i in range(len(doc_index)):
        train_cat_indices = torch.tensor([train_cats[x] for x in doc_index[i]],
                                          device=device)  # retrieve categories of each knn
        unique_elements, counts = torch.unique(train_cat_indices, return_counts=True)  # get unique elements and their counts
        max_count_index = torch.argmax(counts)
        most_common_cat = unique_elements[max_count_index]
        target.append(test_cats[i])  # append target classification
        prediction.append(most_common_cat)  # append predicted classification
        if most_common_cat == test_cats[i]:
            correct_classification += 1  # increment if guess is correct

    # move tensors to CPU
    target_tensor = torch.tensor(target).cpu()
    prediction_tensor = torch.tensor(prediction).cpu()

    # convert list into np array
    target_array = np.array(target_tensor)
    prediction_array = np.array(prediction_tensor)

    # convert it into a one hot vector for each entries
    # NEED TO UNDERSTAND THE SYNTAX OF THESE FUNCTIONS
    num_class = max(train_cats) + 1  # number of class is max value in train_cats + 1
    prediction_one_hot = np.eye(num_class)[prediction_array]  # convert predictions to one hot vector

    correct_percentage = float(correct_classification) / float(len(test_cats)) * 100.0  # calculate percentage
    auroc = roc_auc_score(target_array, prediction_one_hot, multi_class='ovr')  # calculate roc_auc_score

    return correct_percentage, auroc

def evaluate_parameters(l, train_num_start, train_num_end, increment, test_num):
    '''
    Calculates accuracy percentage and auroc scores for different settings for tf_idf.
    '''

    for i in range(train_num_start, train_num_end, increment):
        train_tensor, test_tensor, train_cats, test_cats = tf_idf(dataset, (i, test_num))
        k = int(math.sqrt(max(train_cats) + 1) * i)  # k is total number of training documents sqrt
        print(k)
        percentage, auroc = knn(train_tensor, test_tensor, train_cats, test_cats, k)
        print(f'For {i} documents per category and {test_num} training set,'
            f'accuracy is {percentage}% with auroc score of'
            f' {auroc}\n')
        delete_all_tensors()  # free up all memory
        print(f'Memory allocated: {torch.cuda.memory_allocated()} bytes')
        print(f'Memory reserved: {torch.cuda.memory_reserved()} bytes')


preprocess_data(dataset)
# train_tensor, test_tensor, train_cats, test_cats = tf_idf(dataset,(20, 400),None)
# knn(train_tensor, test_tensor, train_cats, test_cats, 39)
evaluate_parameters(dataset, 20, 40, 10, 50)
