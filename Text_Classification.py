import json
import math
import random

dataset = []
with open('data/News_Category_Dataset_v3.json', 'r') as file:
    for line in file:
        line_dict = json.loads(line)
        dataset.append([line_dict['headline'], line_dict['category']])

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

def tokenize_and_list(l):
    '''
    Takes in a list, tokenize first string of each element and make a set of all
    unique words in the corpus. Returns a set
    '''
    l_length = len(l)  # length of dataset
    set_of_words = set()  # dictionary of unique words
    for i in range(l_length):
        words = l[i][0]  # string to be tokenized
        for word in words:
            set_of_words.add(word)

    return set_of_words

def really_really_tokenize(l):
    '''
    Takes in a list, tokenizes the string of words then replaces it in l. This
    exists because like an idiot the coder forgot to do that in tokenize_and_list
    '''

    for record in l:
        record[0] = record[0].split()

def set_of_category(l):
    '''
    Takes a dataset, produces a set containing all unique categories of the dataset
    '''

    cat_of_set = set()
    for record in l:
        cat_of_set.add(record[1])

    return cat_of_set

def pick_test_set(l, number):
    '''
    Takes in a list and number of test rows and returns the
    '''

    copied_l = l.copy()  # copy l
    random.shuffle(copied_l)  # shuffle the copied l

    test_set = copied_l[:number][:]
    return test_set

def select_training_rows(l, n, cat_indices):
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

def tf(l, word_indices):
    '''
    Takes dataset, dict of word and cat indices, dict containing total
    number of words per category and returns a tf matrix.
    '''

    tf_mat = []  # initialize tf matrix
    for i in range(len(l)):
        tf_row = []  # row of tf matrix
        word_num = 0  # total number of words in the row
        for j in range(len(word_indices)):
            tf_row.append(0.0)  # add word_indices amount of 0s
        for word in l[i][0]:  # bring words to iterate
            if word not in word_indices:  # ignore word not in indices
                continue
            tf_row[word_indices[word]] += 1.0  # increment by word count
            word_num += 1  # increment total word count
        if word_num > (abs(0 - 1e6)):
            tf_row = [x / word_num for x in tf_row]  # divide entire row by total word count
        tf_mat.append(tf_row)

    return tf_mat

def idf(l, word_indices):
    '''
    Given a dataset, dict of word and category indices and tf matrix, calculates idf matrix and
    returns it.
    '''
    n = len(l)  # total number of records

    idf_mat = []  # initialize idf matrix
    for i in range(len(word_indices)):
        idf_mat.append(0.0)  # append 0 for every word in word_indices
    for row in l:
        set_of_words = set()  # set of words that occurs per row
        for word in row[0]:
            if word not in word_indices:
                continue
            set_of_words.add(word)  # add words
        for word in set_of_words:
            idf_mat[word_indices[word]] += 1.0  # increment word frequency

    for i in range(len(idf_mat)):
        idf_mat[i] = math.log(n / (1.0 + idf_mat[i]))  # calculate idf

    return idf_mat

def mat_mul(a, b):
    '''
    Given two matrices of shape (m,n) and (n), performs multiplication
    '''

    output = a.copy()

    for i in range(len(output)):
        for j in range(len(b)):
            output[i][j] = output[i][j] * b[j]

    return output

def tf_idf(l, n_cat, n_test, cat_set):
    '''
    Given a dataset with a set of unique words, number of records per category and test records,
    set of categories, initializes and calculates a
    tf-idf matrix with categories and returns it along with the dict that denotes the index of
    each word and category      FIX THIS DESCRIPTION
    '''

    dict_of_cat_indices = {}  # dictionary of cat as key and index as value
    i = 0  # initialize index
    for cat in cat_set:
        dict_of_cat_indices[cat] = i
        i += 1

    class_num = len(cat_set)

    train_set, remaining_set = select_training_rows(l, n_cat, dict_of_cat_indices)  # get the training set picked, with n rows per cat
    test_set = pick_test_set(remaining_set, n_test)  # test set with n_test records

    word_set = tokenize_and_list(train_set)
    print(len(word_set))
    dict_of_word_indices = {}  # dictionary of word as key and index as value
    i = 0  # index number for words
    for word in word_set:
        dict_of_word_indices[word] = i  # key of i, value of word
        i += 1

    train_tf_matrix = tf(train_set, dict_of_word_indices)
    train_idf_matrix = idf(train_set, dict_of_word_indices)

    test_tf_matrix = tf(test_set, dict_of_word_indices)
    test_idf_matrix = idf(test_set, dict_of_word_indices)

    train_tf_idf_mat = mat_mul(train_tf_matrix, train_idf_matrix)  # multiply two matrices
    test_tf_idf_mat = mat_mul(test_tf_matrix, test_idf_matrix)

    for i in range(len(train_set)):
        cat_index = dict_of_cat_indices[train_set[i][1]]  # convert category into an index
        train_tf_idf_mat[i] = [train_tf_idf_mat[i], cat_index]

    for i in range(len(test_set)):
        cat_index = dict_of_cat_indices[test_set[i][1]]  # convert category into an index
        test_tf_idf_mat[i] = [test_tf_idf_mat[i], cat_index]

    return train_tf_idf_mat, test_tf_idf_mat, dict_of_word_indices, dict_of_cat_indices

def euclidian_distance(a, b):
    '''
    takes 2 matrices of size (n) and calculates the euclidian distance, and returns it
    '''

    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2

    return math.sqrt(sum)

def mat_dot(a, b):
    '''
    calculates a dot product between two matrices
    '''

    result = 0
    for a_data, b_data in zip(a, b):
        result += a_data * b_data

    return result

def euc_norm(mat):
    '''
    calculates Euclidian norm of a vector
    '''

    sum_of_sq = 0  # sum of squared elements of mat
    for data in mat:
        sum_of_sq += data ** 2  # add squared data
    return math.sqrt(sum_of_sq)

def cosine_similarity(a, b):
    '''
    calculates cosine similarity between two matrices
    '''

    numerator = mat_dot(a, b)
    denominator = euc_norm(a) * euc_norm(b)

    return numerator / denominator

def knn(train_mat, test_mat, k):
    '''
    Takes train and test tf-idf matrices and performs k nearest neighbor classification
    '''
    accurate_predictions = 0  # number of accurate predictions

    for test_row in test_mat:
        list_of_dist_and_categories = []  # list of distances and categories to train set from row
        test_category = test_row[1]  # index of category of the test row
        for train_row in train_mat:
            dist = cosine_similarity(test_row[0], train_row[0])  # calculate distance
            list_of_dist_and_categories.append([dist, train_row[1]])  # append distance and category of train row
        list_of_dist_and_categories.sort(key=lambda x: x[0], reverse=True)  # sort by distance
        list_of_categories = [x[1] for x in list_of_dist_and_categories[:k]]  # fetch k nearest neighbors' categories

        train_category = 0  # index of classified category
        knn_category_count = 0  # count of category in list of k nearest neighbors
        for cat in list_of_categories:
            current_freq = list_of_categories.count(cat)
            if current_freq > knn_category_count:
                knn_category_count = current_freq  # knn cat count replaces current freq if greater
                train_category = cat  # and becomes most frequent category

        if train_category == test_category:
            accurate_predictions += 1  # increment 1 for accurate predictions if made correctly

    print(f'Tested for {len(train_mat)} training records and {len(test_mat)} testing records, accuracy is'
          f' {accurate_predictions / float(len(test_mat))}')




preprocess_data(dataset)
cat_set = set_of_category(dataset)
really_really_tokenize(dataset)
train_mat, test_mat, dict_words, dict_cats = tf_idf(dataset, 20, 400,cat_set)
knn(train_mat, test_mat, 40)