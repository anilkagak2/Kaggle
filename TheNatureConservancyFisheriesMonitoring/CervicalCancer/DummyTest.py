from __future__ import print_function
import numpy as np
import sys
import os
from cntk import *

# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy. 
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable 
    X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)
    
    # Specify the data type to match the input variable used later in the tutorial 
    # (default type is double)
    X = X.astype(np.float32)    
    
    # converting class 0 into the vector "1 0 0", 
    # class 1 into vector "0 1 0", ...
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

def linear_layer(input_var, output_dim):
    
    input_dim = input_var.shape[0]
    weight_param = parameter(shape=(input_dim, output_dim))
    bias_param = parameter(shape=(output_dim))
    
    mydict['w'], mydict['b'] = weight_param, bias_param

    return times(input_var, weight_param) + bias_param

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))
        
    return mb, training_loss, eval_error

if __name__=='__main__':
    # Define the network
    input_dim = 2
    num_output_classes = 2

    # Ensure we always get the same amount of randomness
    np.random.seed(0)

    # Create the input variables denoting the features and the label data. Note: the input 
    # does not need additional info on number of observations (Samples) since CNTK creates only 
    # the network topology first 
    mysamplesize = 32
    features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)
    feature = input(input_dim, np.float32)

    # Define a dictionary to store the model parameters
    mydict = {"w":None,"b":None} 

    output_dim = num_output_classes
    z = linear_layer(feature, output_dim)

    label = input((num_output_classes), np.float32)
    loss = cross_entropy_with_softmax(z, label)

    eval_error = classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    learning_rate = 0.5
    lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch) 
    learner = sgd(z.parameters, lr_schedule)
    trainer = Trainer(z, (loss, eval_error), [learner])


    # Initialize the parameters for the trainer
    minibatch_size = 25
    num_samples_to_train = 20000
    num_minibatches_to_train = int(num_samples_to_train  / minibatch_size)

    # Run the trainer and perform model training
    training_progress_output_freq = 50
    for i in range(0, num_minibatches_to_train):
        features, labels = generate_random_data_sample(minibatch_size, input_dim, num_output_classes)
    
        # Specify input variables mapping in the model to actual minibatch data to be trained with
        trainer.train_minibatch({feature : features, label : labels})
        batchsize, loss, error = print_training_progress(trainer, i, 
                                                         training_progress_output_freq, verbose=1)


    # Run the trained model on newly generated dataset
    test_minibatch_size = 25
    features, labels = generate_random_data_sample(test_minibatch_size, input_dim, num_output_classes)

    trainer.test_minibatch({feature : features, label : labels})

    out = softmax(z)
    result = out.eval({feature : features})

    print("Label    :", [np.argmax(label) for label in labels])
    print("Predicted:", [np.argmax(result[i,:]) for i in range(len(result))])
