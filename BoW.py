import math
import json

dataset = []
with open('data/News_Category_Dataset_v3.json', 'r') as file:
    for line in file:
        line_dict = json.loads(line)
        dataset.append([line_dict['headline'] + ' ' + line_dict['short_description'], line_dict['category']])

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

def bag_of_words(l, test_ratio):
    '''
    takes in a dataset and ratio of test dataset, splits the dataset
    up and computes BoW matrix.
    returns BoW matrix for train and test set along with categories
    for each row
    '''
    test_num = int(len(l) // (1 / test_ratio))  # pick number of test row
    test_set = l[:test_num]
    train_set = l[test_num:]

    word_set = tokenize_and_list(train_set)
    cat_set = set_of_category(l)

    dict_of_word_indices = {}  # dictionary of word as key and index as value
    i = 0  # index number for words
    for word in word_set:
        dict_of_word_indices[word] = i  # key of i, value of word
        i += 1

    dict_of_cat_indices = {}  # dictionary of cat as key and index as value
    i = 0  # initialize index
    for cat in cat_set:
        dict_of_cat_indices[cat] = i
        i += 1

    train_mat = []  # BoW list for train set
    for row in train_set:
        mat_row = [[], []]  # row of list to insert, matrix and cat index
        for i in range(len(word_set)):
            mat_row[0].append(0)  # insert 0 for each word
        for word in row[0]:  # for each word in the row
            mat_row[0][dict_of_word_indices[word]] += 1  # increment word count
        mat_row[1] = dict_of_cat_indices[row[1]]  # convert category into index
        train_mat.append(mat_row)

    test_mat = []  # BoW list for test set
    for row in test_set:
        mat_row = [[], []]  # row of list to insert, matrix and cat index
        for i in range(len(word_set)):
            mat_row[0].append(0)  # insert 0 for each word
        for word in row[0]:  # for each word in the row
            if word in dict_of_word_indices:
                mat_row[0][dict_of_word_indices[word]] += 1  # increment word count
        mat_row[1] = dict_of_cat_indices[row[1]]  # convert category into index
        test_mat.append(mat_row)

    return train_mat, test_mat

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
    Takes train and test BoW
     matrices and performs k nearest neighbor classification
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
really_really_tokenize(dataset)
train_mat, test_mat = bag_of_words(dataset, 0.2)
knn(train_mat, test_mat, 50)