import numpy as np
import random
import math

data = np.genfromtxt('data/Mall_Customers.csv', delimiter=',')
data = data[1:,2:]  # removes header and uses three categories; age, annual income and spending score

def pick_centroid(array, k):
    '''
    picks k data points as centroid and returns them as an array
    '''

    num_records = array.shape[0]
    num_columns = array.shape[1]
    index = random.sample(range(0, num_records), k)
    centroid = np.zeros([k, num_columns])


    for i in range(k):
        centroid[i] = array[index[i]]
    return centroid

def euclidian_distance(a, b):
    '''
    takes two arrays of same length, sums the squared difference of all the terms then square roots it
    '''
    differences = 0

    for i in range(a.shape[0]):
        differences += (a[i] - b[i]) ** 2

    return math.sqrt(differences)

def new_centroid(clusters):
    k = len(clusters)
    num_columns = len(clusters[0][0])
    centroids = np.zeros([k, num_columns])
    for i in range(k):
        cluster = clusters[i]
        centroids[i] = np.mean(cluster, axis=0)

    return centroids



def k_means(array, centroids, iter):
    '''
    takes array and centroids, performs iter amount of iterations of the k-means algorithm
    '''

    init_cluster = []
    num_records = array.shape[0]
    classes = centroids.shape[0]
    for i in range(classes):
        init_cluster.append([])

    for i in range(iter):
        cluster = init_cluster
        for j in range(num_records):
            index = 0
            a_dist = euclidian_distance(array[j], centroids[0])
            for k in range(1, classes):
                b_dist = euclidian_distance(array[j], centroids[k])
                if b_dist < a_dist:
                    index = k
            cluster[index].append(array[j])
        centroids = new_centroid(init_cluster)
        print(f'{i+1}th iteration, new centroids are:\n{centroids}\n')





centroids = pick_centroid(data, 3)
k_means(data, centroids, 20)
