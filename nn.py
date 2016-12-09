#############################################################################
#nearest neighbor module for cancer data
#The validator determines the accuracy of a specific classifier on a set of 
#data partitioned a certain number of times.
#
#Date: April 16th, 2016
#
#@author: Vamsi Gadiraju
#File Name: nn.py
#############################################################################

import numpy as np
import scipy.spatial.distance as ssd

def KNNclassifier(training, test, k, d, *args):
    training_numpy = np.array(training)
    test_numpy = np.array(test)

    test_rows = test_numpy.shape[0]
    label_list = []

    train_size = training_numpy.shape
    #remove labels from training data to form test data
    temp = np.delete(training_numpy, train_size[1]-1, axis=1)
        
    for y in range (0, test_rows):
        check_test = test[y]

        #determine distance between test and training data
        val = get_difference(temp, check_test, d, *args)
        least_indeces = np.argsort(val)
        
        #form a new matrix containing the labels of the 
        #k closest vectors
        label_compare = []
        for x in range(0,k):
            check_vector = training[least_indeces[x]]
            new_label = check_vector[-1]
            label_compare.append(new_label)
        
        #determine how much of each label is in the list
        label1 = 0
        label2 = 0
        #set label 1 to be whatever label is on the 0th 
        #element
        comparison = label_compare[0]
        #store the other label
        other_label = ""
        for x in label_compare:
            if x == comparison:
                label1 += 1
            else:
                label2 += 1
                other_label = label2

        #determine which label is more common and append it
        #to the final label list that is to be returned
        if label1 > label2:
            label_list.append(label_compare[0])
        else:
            label_list.append(other_label)

    return label_list

def NNclassifier(training, test):
    """This function takes as input an q x (n+1) array, training, that 
    consists of q rows of observation-label pairs. That is, each row is 
    an n-dimensional observation concatenated with an extra dimension for 
    the class label. The other parameter is a j x n array consisting of j 
    unlabeled, n-dimensional observations. This function will output a 
    1-dimensional array consisting of j labels for the test array 
    observations
    """
    training_numpy = np.array(training)
    test_numpy = np.array(test)

    test_rows = test_numpy.shape[0]
    label_list = []

    train_size = training_numpy.shape
    #remove labels from training data to form test data
    temp = np.delete(training_numpy, train_size[1]-1, axis=1)
        
    for y in range (0, test_rows):
        check_test = test[y]

        #determine distance between test and training data
        val = get_difference(temp, check_test)
        
        # print(val)
        x = np.argsort(val)

        # print(x)

        min_vector = training[x[0]]

        # print(min_vector)
        
        #determine new label and append it
        new_label = min_vector[-1]
        label_list.append(new_label)

    return label_list


def get_difference(training, test_line, d, *args):
    """This function takes in two vectors and determines the euclidean
     distance between the two. 
    """

    #ensure numpy are floats and 2D matrices
    checker = []
    checker.append(test_line)
    training = np.array(training, dtype='f')
    test_line = np.array(checker, dtype='f')

    #determine difference as multidimensional array
    distances = ssd.cdist(training, test_line, d)

    difference = []

    #create a single dimensional array of the distances 
    #to allow for argsort in classifier
    for x in distances:
        difference.extend(x)

    return difference


def n_validator(data, p, classifier, *args):
    """The purpose of this function is to estimate the performance of a 
    classifier in a particular setting. This function takes as input an 
    m x (n+1) array of data, an integer p, the classifier it is checking, 
    and any remaining parameters that the classifier requires will be stored 
    in args .  Here's how it works: It will first randomly mix-up the rows 
    of data. Then it will partition data into p equal parts. It will test 
    all partitions against the training data and return the overall accuracy 
    as a value between 0-1.
    """
    #shuffle data 
    np.random.shuffle(data)

    #partition data correctly
    partitions = np.array_split(data, p)
    #initialize test and training
    test = []
    training = []

    #success counter
    counter = 0
    for i in range(p):
        test_initial = partitions[i]
        #set training_initial to all other partitions
        training_initial = partitions[:i] + partitions[i+1:]
        training = np.vstack(training_initial)
        #delete real label on test data 
        test = np.delete(test_initial, test_initial.shape[1]-1, 1)
        #get labels from classifier
        labels = classifier(training, test, *args)

        for i in range(len(labels)):
            #if generated label and real label are the same,
            #count a success
            if labels[i] == test_initial[i][test_initial.shape[1]-1]:
                counter += 1
    #return decimal indicating accuracy
    return float(counter / len(data))


def analyze_tumor_data(lines):
    """Take in the lines of the tumor data and return in data form.
    The label will be placed at the end of the measurement values (
    last element of array)
    """
    data = []
    
    for line in lines:
        #remove line differentiator and form array
        new_line = line.rstrip('\n').split(",")
        #get label from the front
        label = new_line[1]
        #start at second element to ensure avoidance of id and label
        new_line = new_line[2:]
        #add label at end
        new_line.append(label)
        data.append(new_line)
    
    return data


def form_synthetic_data():
    """Form synthetic data with labels "a" and "b" at the end of 
    arrays.
    """
    #mean values
    x1=2.5
    x2=0.5

    y1=3.5
    y2=1

    mean_1 = [x1,y1]
    mean_2 = [x2,y2]

    #covariance values
    cov_1=[[1,1],[1,4.5]]
    cov_2=[[2,0],[0,1]]

    #form numpy arrays
    c1= np.random.multivariate_normal(mean_1,cov_1,300)
    c2= np.random.multivariate_normal(mean_2,cov_2,300)

    rows1 = c1.shape[0]
    rows2 = c2.shape[0]

    c1 = c1.tolist()
    c2 = c2.tolist()
    
    #append label 'a' to end of first class
    for x in range(0,rows1):
        c1[x].append('a')

    #append label 'b' to end of second class
    for y in range(0,rows2):
        c2[y].append('b')
    
    #combine both classes
    c1.extend(c2)

    return c1

def get_euclidean(data, p, classifier):
    """Print accuracies when distance is calculated using Euclidean
    distance equation. Accuracies are averaged over five calculations
    of each value. K values are all on the range of odd numbers
    between 1-15.
    """
    print("Euclidean Accuracies are: ")
    for y in range(1, 17, 2):
        check = []
        for x in range(0,5):
            result = n_validator(data, p, classifier, y, 'euclidean')
            check.append(result)
        #print value of k along with mean of five values generated
        print("{}: {}".format(y, np.mean(check)))
    print("========================================================")

def get_citydistance(data, p, classifier):
    """Print accuracies when distance is calculated using City Block
    distance equation. Accuracies are averaged over five calculations
    of each value. K values are all on the range of odd numbers
    between 1-15.
    """
    print("City Block Accuracies are: ")
    for y in range(1, 17, 2):
        check = []
        for x in range(0,5):
            result = n_validator(data, p, classifier, y, 'cityblock')
            check.append(result)
        #print value of k along with mean of five values generated
        print("{}: {}".format(y, np.mean(check)))
    print("========================================================")

def get_minkowski(data, p, classifier, p_norm):
    """Print accuracies when distance is calculated using Minkowski
    distance equation. Accuracies are averaged over five calculations
    of each value. K values are all on the range of odd numbers
    between 1-15.
    """
    print("Minkowski Accuracies are: ")
    for y in range(1, 17, 2):
        check = []
        for x in range(0,5):
            result = n_validator(data, p, classifier, y, 'minkowski', p_norm)
            check.append(result)
        #print value of k along with mean of five values generated
        print("{}: {}".format(y, np.mean(check)))
    print("========================================================")

def main():
    # tumor data 
    in_file = open('wdbc.data.txt', 'r')
    lines= in_file.readlines()
    
    data = analyze_tumor_data(lines)

    get_euclidean(data, 5, KNNclassifier)
    get_citydistance(data, 5, KNNclassifier)
    #get minkowski accuracies with varying p-norm values to ensure *args
    #functionality
    get_minkowski(data, 5, KNNclassifier, 1)
    get_minkowski(data, 5, KNNclassifier, 2)
    get_minkowski(data, 5, KNNclassifier, 3)


    #synthetic data 
    data = form_synthetic_data()

    get_euclidean(data, 5, KNNclassifier)
    get_citydistance(data, 5, KNNclassifier)
    #get minkowski accuracies with varying p-norm values to ensure *args
    #functionality
    get_minkowski(data, 5, KNNclassifier, 1)
    get_minkowski(data, 5, KNNclassifier, 2)
    get_minkowski(data, 5, KNNclassifier, 3)

    in_file.close()

main()






    