# Import Modules
from keras.models  import load_model, Model
from keras import optimizers
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.embedding import OpenNetworkEmbedder, EmbeddingClassifier
from skmultilearn.adapt import MLkNN
from skmultilearn.cluster import NetworkXLabelGraphClusterer

import dill
import networkx as nx
import matplotlib.pyplot as plt

from skmultilearn.embedding import CLEMS
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.externals import joblib

#################### Input Parameters ####################
learning_rate = 10**(-5)
img_shape = (100,100,3)
des_layer = -8
only_drama_thresh = 1000
comedy_and_drama_thresh = 1000
train_perc = 0.90
learning_rate = 10**(-6)

# Choose whether to perform gridsearch or create classifier
use_gridsearch = False

# Parameters for Random Forest Classifier
max_depth = None
max_features = 6
min_samples_split = 3
min_samples_leaf = 5
bootstrap = True
n_estimators = 50
num_neighbors = 5



genres = ['Action','Comedy','Drama','Horror','Romance','Thriller']
# Import the CSV File
full_csv_df = pd.read_csv('Data/full_poster_6.csv')
############################################################

#################### Define functions ####################
# Transform labels from ML to MC
def ML_to_MC(labels_ML):
    labels_MC = np.zeros((1,len(labels_ML)))
    
    # Find all unique row-vectors
    unique_label_vec, counts = np.unique(labels_ML, axis=0, return_counts=True)
    #unique_label_vec = [int(label) for label in unique_label_vec]
    class_num = 0
    for label_vec in unique_label_vec:
        for count,label_ML in enumerate(labels_ML):
            if np.array_equal(label_vec, label_ML):
                labels_MC[0,count] = int(class_num)
        class_num += 1
    return unique_label_vec, labels_MC[0], counts

def generate_MC_genres(unique_label_vec, labels_MC, genres):
    names = []
    for index,vect in enumerate(unique_label_vec):
        if index in labels_MC:
            out_vect = []
            # Go through each index of vector
            for label_index,name in enumerate(vect):

                try:
                    if int(name) == 1:
                        out_vect.append(genres[label_index])
                    else:
                        out_vect.append(str(0))
                except:
                    out_vect.append(str(0))

            names.append(out_vect)
    return names

# Function to calculate hamming distance
def hamming_dist(y_true,y_pred):
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    return K.mean(K.sum(K.abs(y_true-y_pred),axis=1))

def hamming_dist_classifier(y_true,y_pred):
    return np.mean(np.sum(np.abs(y_true-y_pred),axis=1))
############################################################

#################### Load data and preprocess it ####################
# Load model
model=load_model('Data/vgg16_1.h5', custom_objects={"hamming_dist": hamming_dist})
# Create new model, removing the last layer
model = Model(model.inputs, model.layers[des_layer].output)
# Compile model
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9), metrics=['mean_squared_error'])

num_imgs_6 = 68054
pickle_in = open("Data/labels_6_ML.pickle","rb")
labels = pickle.load(pickle_in)

data = np.memmap('Data/data_6.dat', dtype='float32', mode='r', shape=(num_imgs_6,img_shape[0],img_shape[1],img_shape[2]))

total_count = np.sum(labels,axis=0)
print('')
print('Below are the number of posters for each genre before thresholding:')
print([int(val) for val in total_count])

unique_label_vec, labels_MC, counts = ML_to_MC(labels)
names = generate_MC_genres(unique_label_vec, labels_MC, genres)
print('')
print('Below are the number of posters in each unique label vector before thresholding:')
print([int(num) for num in counts])
print([str(label) for label in names])

# Go through each label and threshold desired label vectors
print('')
print('Thresholding the data...')
num_after_thresh = 49260 # Manually found
drama_count = 0
comedy_drama_count = 0
other_count = 0
pos_index = 0
temp_data = np.memmap('Data/data_6_MLthresh.dat', dtype='float32', mode='w+', shape=(num_after_thresh,img_shape[0],img_shape[1],img_shape[2]))

temp_labels = np.zeros((num_after_thresh,labels.shape[1]))
cnt = 0
for label_index,label in enumerate(labels):
    #print('On ' + str(cnt) + ' out of ' + str(labels.shape[0]) )
    cnt += 1
    
    
    # Check current label to see if it is only nonzero for Drama (index 2)
    if (int(np.sum(label))==1) and (int(label[2])==1):
        if drama_count <= only_drama_thresh:
            temp_data[pos_index,:,:,:] = data[label_index,:,:,:]
            temp_labels[pos_index] = labels[label_index]
            pos_index+=1
            drama_count+=1

    # Also reduce number of comedy and dramas shared
    elif (int(np.sum(label))==2) and (int(label[1])==1) and (int(label[2])==1):
        if comedy_drama_count <= comedy_and_drama_thresh:
            temp_data[pos_index,:,:,:] = data[label_index,:,:,:]
            temp_labels[pos_index] = labels[label_index]
            pos_index+=1
            comedy_drama_count+=1

    else:
        temp_data[pos_index,:,:,:] = data[label_index,:,:,:]
        temp_labels[pos_index] = labels[label_index]
        pos_index += 1
        other_count += 1
            
print('')
print(str(drama_count) + ' movies removed tagged only with drama')
print(str(comedy_drama_count) + ' movies removed tagged specifically with comedy and drama')
print('There are now ' + str(pos_index) + ' unique movies')

total_count = np.sum(temp_labels,axis=0)
print('')
print('Below are the number of posters for each genre after thresholding:')
print([int(val) for val in total_count])

genre_hist = total_count

unique_label_vec, labels_MC, counts = ML_to_MC(temp_labels)
names = generate_MC_genres(unique_label_vec, labels_MC, genres)
print('')
print('Below are the number of posters in each unique label vector after thresholding:')
print([int(num) for num in counts])
print([str(label) for label in names])

# Split into test and train sets
num_in_test = int(len(temp_labels)*(1-train_perc))
num_in_train = int(len(temp_labels)*train_perc)+1
print('')
print('There are ' + str(num_in_train) + ' train posters and ' + str(num_in_test) + ' test posters') 

print('')
print('Creating the test-dataset...')
test_data = np.memmap('Data/test_data.dat', dtype='float32', mode='w+', shape=(num_in_test,img_shape[0], img_shape[1], img_shape[2]))
test_labels = np.zeros((num_in_test,temp_labels.shape[1]))
counter = 0
for row in list(range(num_in_test)):
    test_data[counter,:,:,:] = temp_data[row,:,:,:]
    test_labels[counter,:] = temp_labels[row,:]
    counter += 1
print('The shape of the test-dataset is: ' + str(test_data.shape))
print('The shape of the test-labels is: ' + str(test_labels.shape))

print('')
print('Creating the train dataset...')
train_data = np.memmap('Data/train_data.dat', dtype='float32', mode='w+', shape=(num_in_train, img_shape[0], img_shape[1], img_shape[2]))
train_labels = np.zeros((num_in_train,temp_labels.shape[1]))
counter = 0
for row in list(range(num_in_test,temp_labels.shape[0])):
    train_data[counter,:,:,:] = temp_data[row,:,:,:]
    train_labels[counter,:] = temp_labels[row,:]
    counter += 1
print('The shape of the train-dataset is: ' + str(train_data.shape))
print('The shape of the train-labels is: ' + str(train_labels.shape))
del temp_data
del temp_labels

print('')
print('Extracting feature vectors for the test and train data...')
print('')
feature_train = model.predict(train_data)
feature_test = model.predict(test_data)
del model

print('')
print('The feature vectors have been extracted')
print('The dimensions of the feature vector array for the test set is: ' + str(feature_test.shape) )
print('The dimensions of the feature vector array for the train set is: ' + str(feature_train.shape) )
############################################################

#################### Define basic architecture for the model ####################
# construct a graph builder that will include label relations weighted by how many times they co-occurred in the data, without self-edges
graph_builder = LabelCooccurrenceGraphBuilder(weighted = True, include_self_edges = False)

edge_map = graph_builder.transform(train_labels)
print('')
print("Our graph builder for {} labels has {} edges".format(6, len(edge_map)))
print('Below is the associated edge map for the train set:')
print(edge_map)

# setup the clusterer to use, we selected the modularity-based approach
clusterer = NetworkXLabelGraphClusterer(graph_builder=graph_builder, method='louvain')
partition = clusterer.fit_predict(train_data,train_labels)
print('')
print('The output of the NetworkXLabelGraphClusterer is below:')
print(partition)

def to_membership_vector(partition):
    return {
        member :  partition_id
        for partition_id, members in enumerate(partition)
        for member in members
    }
membership_vector = to_membership_vector(partition)

indices = list(range(len(genres)))
names_dict = dict(zip(indices,genres))

# Create and save map of graph-based relationships
f = plt.figure()
nx.draw(
    clusterer.graph_,
    pos=nx.circular_layout(clusterer.graph_),
    labels=names_dict,
    with_labels = True,
    width = [10*x/train_labels.shape[0] for x in clusterer.weights_['weight']],
    node_color = [membership_vector[i] for i in range(train_labels.shape[1])],
    cmap=plt.cm.Spectral,
    node_size=10,
    font_size=14
)
f.savefig("Data/label_graph.png")

# Set up the ensemble metaclassifier
#openne_line_params = dict(batch_size=1000, order=3)
#embedder = OpenNetworkEmbedder(
#    graph_builder,
#    'LINE',
#    dimension = 5*train_labels.shape[1],
#    aggregation_function = 'add',
#    normalize_weights=True,
#    param_dict = openne_line_params
#)



dimensional_scaler_params = {'n_jobs': -1}
embedder = CLEMS(metrics.jaccard_similarity_score, is_score=True, params=dimensional_scaler_params)
############################################################

#################### Run Grid-Search over this classifier ####################
if use_gridsearch==True:
    classifier = EmbeddingClassifier( embedder=embedder, regressor=RandomForestRegressor(), classifier=MLkNN(k=5), require_dense=[False, False])
    #classifier = EmbeddingClassifier( embedder, RandomForestRegressor(), MLkNN(k=1), regressor_per_dimension= True )

    parameters = {"regressor__max_depth": [3, None],
                  "regressor__max_features": [3, 6], # was [1,3,6]
                  "regressor__min_samples_split": [3, 10],
                  "regressor__min_samples_leaf": [1, 5],  # [1,3,10]
                  "regressor__bootstrap": [True, False],
                  "regressor__n_estimators": [20, 50]}

    scorer = make_scorer(hamming_dist_classifier, greater_is_better=False)
    grid_classifier = GridSearchCV(classifier, param_grid=parameters, scoring=scorer, verbose=0)

    grid_classifier.fit(feature_train, train_labels)
    print("Best Score (Hamming Distance): %f" % grid_classifier.best_score_)
    print("Optimal Hyperparameter Values: ", grid_classifier.best_params_)
############################################################

#################### Build the classifier for use on our test-set ####################
if use_gridsearch==False:
    # Define using found optimal parameters
    reg = RandomForestRegressor(max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, n_estimators=n_estimators)
#     reg = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    #reg = RandomForestRegressor()

    # Create classifier using those values
    
    #classifier = EmbeddingClassifier( embedder=embedder, regressor=reg, classifier=MLkNN(k=num_neighbors))
    #classifier = EmbeddingClassifier( embedder=embedder, regressor=reg, classifier=MLkNN(k=1), regressor_per_dimension= True, require_dense=[False, False] )
    classifier = EmbeddingClassifier( embedder=embedder, regressor=reg, classifier=MLkNN(k=5), regressor_per_dimension=True)



    print('')
    print('Fitting the train data to our graph-based classifier...')
    print('')
    # Fit to the train set
    classifier.fit(feature_train, train_labels)

    # Make the predictions on our test dataset
    print('Making predictions on our test data...')
    pred_test_labels = classifier.predict(feature_test).toarray()
    
    # Output the classification report
    print('')
    print(classification_report(test_labels,pred_test_labels))
    
    print('')
    print('These results use:')
    print('Max_depth: ' + str(max_depth))
    print('max_features: ' + str(max_features))
    print('min_samples_split: ' + str(min_samples_split))
    print('min_samples_leaf: ' + str(min_samples_leaf))
    print('bootstrap: ' + str(bootstrap))
    print('n_estimators: ' + str(n_estimators))
    print('num_neighbors: ' + str(num_neighbors))
    
    # Save the classifier model pieces
    #with open('Data/classifier.pickle', 'wb') as file:
    #    pickle.dump(classifier, file)
    with open("Data/classifier.pkd", "wb") as dill_file:
        dill.dump(classifier, dill_file)
    
############################################################
