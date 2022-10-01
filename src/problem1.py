import numpy as np
import pandas as pd


# Returns a vector with the estimations for a list of x values and a given theta array
def rs(theta, x):
    return np.sum(theta*x, axis = 1)

# Returns the cost based on the distance between estimation and provided ground true values
def cost(theta, x, y):
    return np.sum((rs(theta, x) - y)**2)/(2*len(y))

# Returns improved theta values by applying gradient descent, that is slowly modifying the theta values based on partial differentiation
def tuneTheta(theta, x, y, learningRate, iterations, PRINT = False):
    for i in range(0, iterations):
        if (PRINT):
            print("Iteration ["+str(i+1)+"] cost: "+str(cost(theta,x,y)))
        newTheta = theta
        for k in range(0, len(theta)):
            newTheta[k] = theta[k] - learningRate * (1/len(x)) * (np.sum((rs(theta, x) - y) * x.iloc[:, k]))
        theta = newTheta
    return theta


def predictDataset(ds, theta):
    # print(ds)
    # For y-values we will use the zeroth (first) column of the provided data
    ys = ds.iloc[:,0]
    # print(ys)
    # For x-values we'll use columns one (second) thru nine (tenth)
    xs = ds.iloc[:, [1,2,3,4,5,7,8,9]]
    # Inserts a 0th column in the dataset filled with ones, this is for theta_0 as the bias
    xs.insert(0, 0, np.ones(len(xs)), True)

    # print(xs)

    return rs(theta, xs)


def learnDataset(ds, learningRate, iterations, PRINT = False):
    # TODO: Make arguments for which columsn to use for which data
    # For y-values we will use the zeroth (first) column of the provided data
    ys = ds.iloc[:,0]
    # For x-values we'll use columns one (second) thru nine (tenth)
    # TODO: Make this use the remaining columns, not hardcoded values
    xs = ds.iloc[:, [1,2,3,4,5,7,8,9]]
    # Inserts a 0th column in the dataset filled with ones, this is for theta_0 as the bias
    xs.insert(0, 0, np.ones(len(xs)), True)
    
    # Produce a warning if for some reason the length of y and x is different
    if (len(ys) != len(xs)):
        print("[WARNING] Uneven length of ys and xs")

    # Initialize theta values, any number should be fine and for example zeros or ones could be used. I chose random such that running the alogrithm multiple times could result in a different local minima
    # TODO: Make this a choice using arguments.
    # TODO: Implement a choice to run the function with multiple different sets of random numbers, to explore local minimas.
    theta = np.random.rand(len(xs.columns))
 
    # We are now ready to run the multivariate linear regression as such:
    # tuneTheta(theta, xs, ys, 0.0002, 500)
    # for the purposes of leave-one-out cross-validation this function will just return the tuned values for theta
    return tuneTheta(theta, xs, ys, learningRate, iterations, PRINT)





# Filenmae for CSV file
FILENAME = '../Resources/spotify_data.csv'


# Import the CSV file with pandas
dataset = pd.read_csv(FILENAME, header=None)


print('Imported dataset from file: '+FILENAME)
print('Dataset is of length: '+str(len(dataset)))

# Basic usage is 
# learnDataset(dataset, 0.0002, 250)
# which will learn from the provided dataset, using the provided learnRate and iterations, and return proper theta values.
# learnDataset(dataset, 0.0002, 250, True)
# TODO: In the future I'd like to implement a function that automatically determines a proper learnRate and number of iterations by analyzing the cost delta.




# Implementation of leave-one-out cross-validation
# We need to train models on datasets from the original dataset, but with a single row of data left out as validation
# This will be done N times to get a prediction for all values

def loocv(learningRate, iterations):
    # First we initalize a list to store out predictions
    predictions = {"i": [], "p": [], "t": []}

    # Then we loop thru the range of the dataset
    for l in range(0, len(dataset)):
    # for l in range(0, 2):
        # For each row we want to use it as validation

        # Select row l as validationData, drop the rest
        trainingList = list(range(0,len(dataset)))
        trainingList.remove(l)
        validationData = dataset.drop(trainingList)

        # Select row all rows except l as trainingData, drop l
        trainingData = dataset.drop(l)

        # Train model using training data
        th = learnDataset(trainingData, learningRate, iterations, False)

        # Predict validation using model
        p = predictDataset(validationData, th)
        
        print("["+str(l)+"/"+str(len(dataset))+"] Prediction: "+str(p[l])+", Ground truth: "+str(validationData[0][l]))

        predictions['i'].append(l)
        predictions['p'].append(p[l])
        predictions['t'].append(validationData[0][l])
    return predictions




# Implementation of plotting data
# TODO: Plotting should be separated from calculation of data to make it easier to compute data, save it, then plot it multiple times
import matplotlib.pyplot as plt
plotData = loocv(0.00025, 400)

plt.scatter(plotData['i'], plotData['p'])
plt.scatter(plotData['i'], plotData['t'])

plt.show()
