import sys
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from save_csv import results_to_csv



np.random.seed(42)
spam_data = np.load('../data/spam-data-hw3.npz')
mnist_data = np.load('../data/mnist-data-hw3.npz')


mnistTest = mnist_data['test_data']
mnistTrain = mnist_data['training_data']
mnistLabels = mnist_data['training_labels']



# flatten the training table
mnistOriginalTable = mnistTrain[np.arange(mnist_data['training_data'].shape[0])]
mnistOriginalTable = mnistOriginalTable.reshape(mnistOriginalTable.shape[0], -1)

mnistTest = mnistTest[np.arange(mnist_data['test_data'].shape[0])]
mnistTest = mnistTest.reshape(mnistTest.shape[0], -1)


# Part1
# define helper function that normalize each row of a matrix
def normalize_image(mat):
    for i in np.arange(mat.shape[0]):
        norml2 = norm(mat[i])
        mat[i] = mat[i]/norml2

cov_lst = []
mean_lst = []
normalize_image(mnistOriginalTable)
for i in range(10):
    data = mnistOriginalTable[mnistLabels==i]
    mean = np.mean(data,axis=0)
    cov = np.cov(data.transpose())
    mean_lst.append(mean)
    cov_lst.append(cov)


# Part 2
fig, ax = plt.subplots()
ax.imshow(cov_lst[3])
plt.show()

#Part 3


#shuffle index
mnistIndex = np.arange(mnistOriginalTable.shape[0])
np.random.shuffle(mnistIndex)
#set 10000 validation
mnistValidationIndex = mnistIndex[-10000:]
mnistValidationTable = mnistOriginalTable[mnistValidationIndex,:]
mnistValidationLabels = mnistLabels[mnistValidationIndex]


error_list = []
training_sizes = [100,200, 500, 1000, 2000, 5000, 10000, 30000, 50000]

#part a
#Hold out 10,000 as validation set:
#set training table:

trainDigitError = []

for training_size in training_sizes:
    mnistTrainIndex = mnistIndex[training_size:]
    mnistTrainTable= mnistOriginalTable[mnistTrainIndex,:]
    mnistTrainLabels = mnistLabels[mnistTrainIndex]

    cov_lst = []
    for i in range(10):
        cov_lst.append(np.cov(mnistTrainTable[mnistTrainLabels==i].transpose()))

    weight_cov = 0    
    for i in range(len(cov_lst)):
        weight = mnistTrainTable[mnistTrainLabels==i].shape[0]
        weight_cov = weight_cov + cov_lst[i] * weight
    weight_cov = weight_cov / training_size  



    all_classes = []
    for i in range(10):
        #compute class mean:
        mean = np.mean(mnistTrainTable[mnistTrainLabels==i], axis = 0)
        #compute logpdf
        logObject = multivariate_normal(mean,weight_cov, allow_singular = True)
        logPDF = logObject.logpdf(mnistValidationTable) + np.log(mnistTrainTable[mnistTrainLabels==i].shape[0]/50000)
        all_classes.append(logPDF)
    generated_labels = np.argmax(all_classes,axis = 0)

#part d start:
    correctly_predicted = generated_labels[generated_labels == mnistValidationLabels]
    accuracy = []
    for i in range(10):
        accuracy.append(1 - sum(correctly_predicted==i)/sum(mnistValidationLabels==i))
    trainDigitError.append(accuracy)

for i in range(10):
    modified = [sizeList[i] for sizeList in trainDigitError]
    plt.plot(training_sizes, modified, label = "Digit" + str(i))
plt.xlabel("Training Data Size")
plt.ylabel("Error Rate on the Validation Set")
plt.title("LDA for MNIST Dataset, per digit")
plt.legend()
plt.show()

# part d end

    error = 1 - sum(generated_labels == mnistValidationLabels)/len(generated_labels)
    error_list.append(error)


plt.plot(training_sizes, error_list)
plt.xlabel("trianing sizes")
plt.ylabel("error rate")
plt.title("Error Rate VS Training Sizes LDA")
plt.show()


#part b QDA

trainDigitError = []



for training_size in training_sizes:
    mnistTrainIndex = mnistIndex[training_size:]
    mnistTrainTable= mnistOriginalTable[mnistTrainIndex,:]
    mnistTrainLabels = mnistLabels[mnistTrainIndex]

    all_classes = []
    for i in range(10):
         #compute class mean:
        mean = np.mean(mnistTrainTable[mnistTrainLabels==i], axis = 0)
        data = mnistTrainTable[mnistTrainLabels==i]
        demeaned_data = data-mean
        classCov = np.dot(demeaned_data.T, demeaned_data) / demeaned_data.shape[0] + np.eye(data.shape[1]) * 0.001

        
        #compute logpdf
        logObject = multivariate_normal(mean, classCov)
        logPDF = logObject.logpdf(mnistValidationTable) + np.log(mnistTrainTable[mnistTrainLabels==i].shape[0]/50000)
        all_classes.append(logPDF)
    generated_labels = np.argmax(all_classes,axis = 0)
    error = 1 - sum(generated_labels == mnistValidationLabels)/len(generated_labels)
    error_list.append(error)


plt.plot(training_sizes, error_list)
plt.xlabel("trianing sizes")
plt.ylabel("error rate")
plt.title("Error Rate VS Training Sizes LDA")
plt.show()

#part d start:
    correctly_predicted = generated_labels[generated_labels == mnistValidationLabels]
    accuracy = []
    for i in range(10):
        accuracy.append(1 - sum(correctly_predicted==i)/sum(mnistValidationLabels==i))
    trainDigitError.append(accuracy)

for i in range(10):
    modified = [sizeList[i] for sizeList in trainDigitError]
    plt.plot(training_sizes, modified, label = "Digit" + str(i))
plt.xlabel("Training Data Size")
plt.ylabel("Error Rate on the Validation Set")
plt.title("QDA for MNIST Dataset, per digit")
plt.legend()
plt.show()

#part d end







plt.plot(training_sizes, error_list)
plt.xlabel("trianing sizes")
plt.ylabel("error rate")
plt.title("Error Rate VS Training Sizes LDA")
plt.show()


#kaggle:

#normalize test data
normalize_image(mnistTest)

mnistTrainTable= mnistOriginalTable[mnistIndex,:]
mnistTrainLabels = mnistLabels[mnistIndex]


all_classes = []    
for i in range(10):
    #compute class mean:
    
    cov = np.cov(mnistTrainTable[mnistTrainLabels==i].transpose())
    identity_mat = np.eye(cov.shape[0])
    padded_cov = cov + 0.00003 * identity_mat
    mean = np.mean(mnistTrainTable[mnistTrainLabels==i], axis = 0)
    #compute logpdf
    logObject = multivariate_normal(mean,padded_cov)
    logPDF = logObject.logpdf(mnistTest) + np.log(mnistTrainTable[mnistTrainLabels==i].shape[0]/50000)
    all_classes.append(logPDF)
generated_labels = np.argmax(all_classes,axis = 0)
results_to_csv(generated_labels)





#Q5

# load spam data
spamTest = spam_data['test_data']
spamTrain = spam_data['training_data']
spamLabels = spam_data['training_labels']
# get a shuffled index of the size of training table 
# get the first 20% of the shuffled index
# and create a validation table and extract labels accordingly
spamShuffledIndex = np.arange(spamTrain.shape[0])
np.random.shuffle(spamShuffledIndex)
spamValidationIndex = spamShuffledIndex[:int(spamTrain.shape[0]*0.2)]
spamValidationTable = spamTrain[spamValidationIndex,:]
spamValidationLabels = spamLabels[spamValidationIndex]
spamRestIndex = spamShuffledIndex[int(spamTrain.shape[0]*0.2):]
spamRestTable= spamTrain[spamRestIndex,:]
spamRestLabels = spamLabels[spamRestIndex]



all_classes = [] 
spamTrainTbl = spamTrain[spamShuffledIndex]
spamTrainLbl = spamLabels[spamShuffledIndex]
for i in range(2):
    #compute class mean:
    cov = np.cov(spamTrainTbl[spamTrainLbl==i].transpose())
    identity_mat = np.eye(cov.shape[0])
    padded_cov = cov + 0.0003 * identity_mat
    mean = np.mean(spamTrainTbl[spamTrainLbl==i], axis = 0)
    #compute logpdf
    logObject = multivariate_normal(mean,padded_cov)
    logPDF = logObject.logpdf(spamTest) + np.log(spamTrainTbl[spamTrainLbl==i].shape[0]/4172)
    all_classes.append(logPDF)
generated_labels = np.argmax(all_classes,axis = 0)
results_to_csv(generated_labels)