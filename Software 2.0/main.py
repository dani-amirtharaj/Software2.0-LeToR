# coding: utf-8

# In[1]:


import numpy as np  # This package is used for storing and manipulating multi-demnsional arrays and matrices
import \
    tensorflow as tf  # This is the tensorflow package which helps represents the system as a dataflow graph and provides functions to implement ML algorithms
from tqdm import tqdm_notebook  # This package shows progress of each iteration when called with an iterator
import pandas as pd  # This package supplies the dataframe structure to represent spreadsheets/tabular data in Python
from keras.utils import \
    np_utils  # This package is used to import the to_categorical function to encode output labels to binary representation

# Below line is used to send plots to the notebook inline to be outputted within the notebook itself
# %matplotlib inline


# ## Logic Based FizzBuzz Function [Software 1.0]

# In[2]:


"""This function implements the simple logic,
if n is divisible by 3 and 5 then return FizzBuzz or 
else if n is divisible by 3 then return Fizz or
else if n is divisible by 5 then return Buzz or 
else just return Other"""


def fizzbuzz(n):
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


# ## Create Training and Testing datasets in CSV Format

# In[3]:


# This function takes in the range of numbers to be generated, and generates a .CSV file for the inputs and output labels of the range of numbers, thereby generating the dataset
def createInputCSV(start, end, filename):
    # 2 empty lists are declared to store data later
    inputData = []
    outputData = []

    # In the empty lists created before, data obtained from the fizzbuzz function are appended to corresponding list
    for i in range(start, end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))

    # A dictionary is created, to which the input and output lists are passed as values to the appropriate keys "input" and "label"
    dataset = {}
    dataset["input"] = inputData
    dataset["label"] = outputData

    # The dictionary created is passed on to the pandas DataFrame function which converts the dataset to a .csv file with name, filename
    pd.DataFrame(dataset).to_csv(filename)

    print(filename, "Created!")


# ## Processing Input and Label Data

# In[4]:


# This function takes in data read from the .csv file and passes them to the encode functions to 'pre-process' the input data and output labels, and map them to different values that would help the ML system learn better from the data
def processData(dataset):
    data = dataset['input'].values
    labels = dataset['label'].values

    processedData = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel


# In[5]:


# This function encodes the input number to a list of 10 features which comprise of bits of the number's binary representation
def encodeData(data):
    processedData = []

    for dataInstance in data:
        # This operation converts the input data to 10 features, based on its binary value. We use the range 10 so we can accomodate our range of numbers, i.e 1000 < 2 ** 10
        # For example the number 10 is encoded to [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        # Input data is bitwise shifted using >> and ANDed bitwise with numbers from 1 to 10
        processedData.append([dataInstance >> d & 1 for d in range(10)])

    # This returns an array of lists of length 10 each
    return np.array(processedData)


# In[6]:


# This function maps the 4 different labels to integers and then to binary bit representation (not binary value) that can be learnt by the algorithm 
def encodeLabel(labels):
    processedLabel = []

    # Assigining integer labels for each label in the dataset
    for labelInstance in labels:
        if (labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif (labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif (labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    # to_categorical converts the integer label to binary bits signifying the class based on the index
    # For example, 1 is converted to [0,1,0,0] and 3 to [0,0,0,1]
    return np_utils.to_categorical(np.array(processedLabel), 4)


# In[7]:


# Create datafiles
createInputCSV(101, 1001, 'training.csv')
createInputCSV(1, 101, 'testing.csv')

# In[8]:


# Read Dataset
trainingData = pd.read_csv('training.csv')
testingData = pd.read_csv('testing.csv')

# Process Dataset
processedTrainingData, processedTrainingLabel = processData(trainingData)
processedTestingData, processedTestingLabel = processData(testingData)

# ## Tensorflow Model Definition

# In[9]:


# Defining Placeholder
inputTensor = tf.placeholder(tf.float32, [None, 10])
outputTensor = tf.placeholder(tf.float32, [None, 4])
testLabel = tf.placeholder(tf.float32, [None, None])

# In[10]:


# Defining number of neurons in each layer
NUM_HIDDEN_NEURONS_LAYER_1 = 250
NUM_HIDDEN_NEURONS_LAYER_2 = 250

# Defining the learning rate which will be used by the Optimizer to reach an optimal point or minima
LEARNING_RATE = 0.07


# Initializing the weights to Normal Distribution, this is done to increase the probability that the weights will achieve a good minima, and not get stuck at a bad minima
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# Initializing the input to hidden layer weights
input_hidden_weights_1 = init_weights([10, NUM_HIDDEN_NEURONS_LAYER_1])

# Initializing the hidden 1 to hidden 2 layer weights
input_hidden_weights_2 = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, NUM_HIDDEN_NEURONS_LAYER_2])

# Initializing the hidden to output layer weights
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_2, 4])

# Computing values at the hidden layers using the relu activation function
hidden_layer_1 = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights_1))
hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, input_hidden_weights_2))

# Computing values at the output layer by matrix multiplication
output_layer = tf.matmul(hidden_layer_2, hidden_output_weights)

# Defining Error Function, here cross entropy loss function is used to minimise loss, along with softmax activation at the output layer
# Softmax ensures the probability of each class is obtained as the output of the network
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters passed
training = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function, gives class which the algorithm gives the max probability of being correct
prediction = tf.argmax(output_layer, 1)

# Class wise accuracy metric is defined, to get accuracy of prediction of each output class/ label
accuracy, accuracyOp = tf.metrics.mean_per_class_accuracy(labels=tf.argmax(testLabel, 1),
                                                          predictions=tf.argmax(output_layer, 1), num_classes=4)

# # Training the Model

# In[11]:


# Number of Epochs determine the number of times the algorithm is repeatedly trained on the training data
NUM_OF_EPOCHS = 3000

# Batch size determines the size of the input matrix that is given to the network to train at one go, and is usually kept between 50 and 256
BATCH_SIZE = 150

# Defining number of times the algorithm will be trained with different random weight initializations but same hyper-parameters and network settings, in order to get mean and variance of accuracies of the model
NUM_OF_INIT = 5

# The same network setting will be used to collect data on accuracies, loss and predictions for different random weight initializations and the following lists are initialized to hold this data
training_accuracy_list = []
training_loss_list = []
predictedTestLabel = []
accuracyList = []

# Starting tensorflow session
with tf.Session() as sess:

    # Run NUM_OF_INIT times to run over NUM_OF_INIT random weight initializations
    for itr in range(NUM_OF_INIT):
        training_accuracy = []
        training_loss = []

        # Sets values for the variables in the TensorFlow network
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # Ouputs the progress meter in the notebook
        for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):

            # Shuffles the Training Dataset at each epoch
            p = np.random.permutation(range(len(processedTrainingData)))
            processedTrainingData = processedTrainingData[p]
            processedTrainingLabel = processedTrainingLabel[p]

            # Start batch training
            for start in range(0, len(processedTrainingData), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end],
                                              outputTensor: processedTrainingLabel[start:end]})
            # Training accuracy for an epoch
            training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                                             sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                                             outputTensor: processedTrainingLabel})))
            # Training loss for an epoch
            training_loss.append(sess.run(error_function, feed_dict={inputTensor: processedTrainingData,
                                                                     outputTensor: processedTrainingLabel}))
        # Collects training accuracies over each epoch for different weight initializations 
        training_accuracy_list.append(training_accuracy)
        training_loss_list.append(training_loss)

        # Testing: Collects predictions output by the network over the testing inpupt data
        predictedTestLabel.append(sess.run(prediction, feed_dict={inputTensor: processedTestingData}))

        # Gets the training accuracies of each class by comparing prediction and testing output labels
        accuracyList.append(
            sess.run(accuracyOp, feed_dict={inputTensor: processedTestingData, testLabel: processedTestingLabel}))

# In[12]:


# The matplotlib package is used for plotting graphs
import matplotlib
import matplotlib.pyplot as plt

# Outputs training accuracy and loss against epoch for NUM_OF_INIT random initializations

# for itr in range(NUM_OF_INIT):
#     fig1, ax1=plt.subplots(figsize=(23,8))
#     ax1.plot(np.array(training_accuracy_list[itr]))
#     ax1.set(xlabel='Number of Epochs', ylabel='Training accuracies')
#     ax1.grid()
#     fig2, ax2=plt.subplots(figsize=(23,8))
#     ax2.plot( training_loss_list[itr])
#     ax2.set(xlabel='Number of Epochs', ylabel='Training Loss')
#     ax2.grid()

# Outputs training accuracy and loss against epoch for the first random initialization only
fig1, ax1 = plt.subplots(figsize=(23, 8))
ax1.plot(np.array(training_accuracy_list[1]))
ax1.set(xlabel='Number of Epochs', ylabel='Training accuracies')
ax1.grid()
fig2, ax2 = plt.subplots(figsize=(23, 8))
ax2.plot(training_loss_list[1])
ax2.set(xlabel='Number of Epochs', ylabel='Training Loss')
ax2.grid()

plt.show()


# In[13]:


# Decodes the integer values to the corresponding labels
def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# # Testing the Model [Software 2.0]

# In[14]:


predictedTestLabelListFin = []
errors = []
correct = []
accuracy = []

# Predictions of different weight initializations are looped over to be able to calculate accuracy for each random weight initialization
for itr in range(NUM_OF_INIT):
    wrong = 0
    right = 0
    m = 0
    predictedTestLabelList = []
    for i, j in zip(processedTestingLabel, predictedTestLabel[itr]):
        # 'others' class
        if j == 0:
            # Appends the input number itself when the class is 'others'
            predictedTestLabelList.append(testingData["input"][m])
        # 'Fizz', 'FizzBuzz' and 'Buzz' classes
        else:
            # Appends the corresponding class Fizz', 'FizzBuzz' or 'Buzz'
            predictedTestLabelList.append(decodeLabel(j))
        m += 1
        # Counts the number of errors and correct predictions over the testing dataset
        if np.argmax(i) == j:
            right = right + 1
        else:
            wrong = wrong + 1
    # Appends the predicted list for each random initialization
    predictedTestLabelListFin.append(predictedTestLabelList)

    # Appends the number of wrong, right predictions with the accuracy percentage for each random initialization
    errors.append(wrong)
    correct.append(right)
    accuracy.append(right / (right + wrong) * 100)

# Prints the best training accuracy over the different random initializations
print("Best Testing Accuracy: " + str(max(accuracy)))
# Prints number of errors, and correct predictions for the random weight initialization that had the best  accuracy
print("Errors: " + str(min(errors)), " Correct :" + str(max(correct)) + '\n')

# Prints tesing accuracies and mean and variance of testing accuracies taken with different weight initializations
print("Testing Accuracies over 5 random weight initializations: " + str(accuracy))
print("Mean of Testing Accuracies: " + str(np.mean(accuracy)))
print("Variance of Testing Accuracies: " + str(np.var(accuracy)))

# Prints mean fizz, buzz and fizzbuzz accuracy over the outputs of the network for different random weight intializations
print("Mean Fizz Accuracy: ", np.mean(np.array(accuracyList)[:, 1]) * 100)
print("Mean Buzz Accuracy: ", np.mean(np.array(accuracyList)[:, 2]) * 100)
print("Mean FizzBuzz Accuracy: ", np.mean(np.array(accuracyList)[:, 3]) * 100, '\n')

# Please input your UBID and personNumber 
testDataInput = testingData['input'].tolist()
testDataLabel = testingData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "damirtha")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50291137")

# Inserts null into the first 2 rows of the predicted labels in the .csv file
predictedTestLabelListFin[accuracy.index(max(accuracy))].insert(0, "")
predictedTestLabelListFin[accuracy.index(max(accuracy))].insert(1, "")

# Creating dictionary output with appropriate keys, and value lists
output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

# Inserts the labels obtained with the best accuracy under different weight initializations into the output dictionary
output["predicted_label"] = predictedTestLabelListFin[accuracy.index(max(accuracy))]

# Converts dictionary to DataFrame
opdf = pd.DataFrame(output)

# Writes DataFrame to .csv file
opdf.to_csv('output.csv')
