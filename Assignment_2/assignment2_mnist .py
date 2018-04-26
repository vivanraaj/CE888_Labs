
# coding: utf-8


# import packages
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras import callbacks
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import random
import pickle


# # Concatenate original and permutated datasets


## load 1st data
X1_train = pickle.load(open("./cleaned_data/mnist_X1_train", "rb"))
X1_test = pickle.load(open("./cleaned_data/mnist_X1_test", "rb"))



## load 2nd data
X2_train = pickle.load(open("./cleaned_data/mnist_X2_train", "rb"))
X2_test = pickle.load(open("./cleaned_data/mnist_X2_test", "rb"))



## load 3rd data
X3_train = pickle.load(open("./cleaned_data/mnist_X3_train", "rb"))
X3_test = pickle.load(open("./cleaned_data/mnist_X3_test", "rb"))



## load 4th data
X4_train = pickle.load(open("./cleaned_data/mnist_X4_train", "rb"))
X4_test = pickle.load(open("./cleaned_data/mnist_X4_test", "rb"))



## concatenate all datasets
full_X_train = np.concatenate([X1_train,X2_train,X3_train,X4_train], axis=0)
full_X_test = np.concatenate([X1_test,X2_test,X3_test,X4_test], axis=0)



## load y train and test
Y_train = pickle.load(open("./cleaned_data/mnist_Y_train", "rb"))
Y_test = pickle.load(open("./cleaned_data/mnist_Y_test", "rb"))



full_Y_train = np.concatenate([Y_train,Y_train,Y_train,Y_train], axis=0)
full_Y_test = np.concatenate([Y_test,Y_test,Y_test,Y_test], axis=0)


# # split data to training and validation


from sklearn.model_selection import train_test_split
final_X_train, final_X_valid, final_Y_train, final_Y_valid = train_test_split(full_X_train,full_Y_train, test_size=0.33, random_state=42,shuffle=False)


final_X_valid.shape



final_Y_train.shape


# # Define autoencoder structure


input_shape = 784
num_classes = 10



def create_autoencoder_classifer(input_shape):
    input_img = Input(shape=(input_shape,))
    x = Dense(input_shape, activation='relu')(input_img)
    encoded1 = Dense(input_shape//2, activation='relu')(x)
    encoded2 = Dense(input_shape//8, activation='relu')(encoded1)
    #### y is the hidden layer below
    hidden_layer = Dense(input_shape//256, activation='relu')(encoded2)
    decoded2 = Dense(input_shape//8, activation='relu')(hidden_layer)
    decoded1 = Dense(input_shape//2, activation='relu')(decoded2)
    out = Dense(input_shape, activation='sigmoid')(decoded1)
    autoencoder = Model(input_img, out)
    encoder = Model(input_img, hidden_layer)
    # define new model encoder->Dense  10 neurons with soft max for classification 
    out2 = Dense(num_classes, activation='softmax')(encoder.output)
    classifier = Model(encoder.input,out2)
    autoencoder.compile(optimizer='adam', loss='mse') 
    classifier.compile(loss='categorical_crossentropy',optimizer='adam') 
    return autoencoder, classifier


# # Pretraining of newly created autoencoders


# number of autoencoder instances
n = 3

# number of training subsets
generate_size = 2000



# loop through all autoencoders
for i in range(0,n):
    autoencoder,classifier = create_autoencoder_classifer(input_shape)
    # to generate batches
    
    # take only 1st random 2000 samples to pretrain the autoencoders 
    gen_random = random.sample(range(0, len(final_X_train)), generate_size)
    x_train_pretrain = final_X_train[gen_random,:]
    y_train_pretrain = final_Y_train[gen_random,:]
    
    # callbacks for autoencoders
    # ModelCheckpoints is used to save the model after every epoch
    cbks_autoencoder = [callbacks.ModelCheckpoint(filepath='./mnist_models/autoencoder_model'+str(i)+'.h5')]
    
    # run training
    autoencoder.fit(x_train_pretrain,x_train_pretrain,epochs=1,batch_size=128,shuffle=True,callbacks =cbks_autoencoder)
    
    # visualize model architecture
    plot_model(autoencoder, to_file='autoencoder_pretrain_plot.png', show_shapes=True, show_layer_names=True)
    
    # save the classifier
    classifier.save('./mnist_models/newmodel_model'+str(i)+'.h5')


# # Train autoencoders and create new autoencoders on the fly


size_batch = 25000
epochs = 2
score_for_autoencoder = []
previous_F1 = 0
total_F1 = 0
number_of_validation_batches = 20
validation_batch_size = int(len(final_X_valid)/number_of_validation_batches)
threshold = 0.25
label = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
labels = list(label)



for k in range(0,epochs):   
    print(k)
    for i in range(0,int(len(final_X_train)/size_batch)):
        start_index = i * size_batch
        end_index = size_batch+(i *size_batch)
        train_set = full_X_train[start_index:end_index,:]
        target_set = full_Y_train[start_index:end_index,:]
        
        for j in range(0,n):
            autoencoder = load_model('./mnist_models/autoencoder_model'+str(j)+'.h5')
            score_for_autoencoder.append(autoencoder.evaluate(train_set, train_set))

        print(score_for_autoencoder)
        if threshold > np.min(score_for_autoencoder):
            # no need TO create new autoencoders
            ## continue training it
            add_autoencoder = False
            min_error_index = np.argmin(score_for_autoencoder)
            threshold = 0.15
            autoencoder = load_model('./mnist_models/autoencoder_model'+str(min_error_index)+'.h5')  
            classifier = load_model('./mnist_models/newmodel_model'+str(min_error_index)+'.h5')
        else:
            print('A new autoencoder and classifier is created')
            add_autoencoder = True
            autoencoder, classifier = create_autoencoder_classifer(input_shape)
            autoencoder.save('./mnist_models/autoencoder_model'+str(n)+'.h5')
            classifier.save('./mnist_models/newmodel_model'+str(n)+'.h5')
            # add extra classifer
            n = n+1
            threshold = 0.7

        score_for_autoencoder = []

        autoencoder.fit(train_set,train_set,epochs=1,batch_size=128,shuffle=True)
        classifier.fit(train_set, target_set,epochs=1,batch_size=128,shuffle=True)

        if add_autoencoder == True:
            autoencoder.save('./mnist_models/autoencoder_model'+str(n)+'.h5') 
            classifier.save('./mnist_models/newmodel_model'+str(n)+'.h5')
        else:
            autoencoder.save('./mnist_models/autoencoder_model'+str(min_error_index)+'.h5')
            classifier.save('./mnist_models/newmodel_model'+str(min_error_index)+'.h5')
    ## evaluate on validation data
    for l in range(0, number_of_validation_batches):
        start_index = l * validation_batch_size
        end_index = validation_batch_size + (l * validation_batch_size)
        valid_X = final_X_valid[start_index:end_index, :]
        valid_Y = final_Y_valid[start_index:end_index, :]

        for j in range(0, n):
            autoencoder = load_model('./mnist_models/autoencoder_model'+str(j)+'.h5')
            score_for_autoencoder.append(autoencoder.evaluate(valid_X, valid_X))

        min_error_index = np.argmin(score_for_autoencoder)
        classifier = load_model('./mnist_models/newmodel_model'+str(min_error_index)+'.h5')
        predictions = classifier.predict(valid_X)
        f1 = f1_score(valid_Y, predictions.round(), labels=labels, average='macro')
        total_F1 = total_F1 +f1

        score_for_autoencoder = []
    f1_average = total_F1/number_of_validation_batches
    total_F1 = 0
    if (previous_F1< f1_average):
        previous_F1 = f1_average
    else:
        # reduce epochs
        k = epochs-1


# # Evaluate the results for the trained autoencoders


number_of_batches = 5
test_batch_size = int(len(full_X_test)/number_of_batches)
#number_of_classifiers = 1
evaluation_errors = []
list_of_predictions = []




statistics = []
statistics_for_batch = []

total_acc = 0
total_f1 = 0


print(n)



for i in range(0,number_of_batches):
    print(i)
    start_index = i * test_batch_size
    end_index = test_batch_size + (i * test_batch_size)
    test_X_set = full_X_test[start_index:end_index, :]
    test_Y_set = full_Y_test[start_index:end_index, :]

    for j in range(0,n):
        autoencoder = load_model('./mnist_models/autoencoder_model'+str(j)+'.h5')
        evaluation_errors.append(autoencoder.evaluate(test_X_set, test_X_set))

    min_error_index = np.argmin(evaluation_errors)
    classifier = load_model('./mnist_models/newmodel_model'+str(min_error_index)+'.h5')
    predictions = classifier.predict(test_X_set)
    
    # collect evaluation results
    f1 = f1_score(test_Y_set,predictions.round(), labels= labels, average='macro')
    acc = accuracy_score(test_Y_set,predictions.round())
    
    statistics_for_batch.append(acc)
    statistics_for_batch.append(f1)
    statistics.append(statistics_for_batch)
    
    evaluation_errors = []
    statistics_for_batch = []



print('\n                           Accuracy,       F1 score ')
for i in range(0,number_of_batches):
    print('\nBatch '+str(i)+' Statistics: '+str(statistics[i]))

