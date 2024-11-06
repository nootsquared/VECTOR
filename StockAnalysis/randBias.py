import numpy as np

numOfIntervals = 20
weightCoeff = 0.0
spreadCoeff = 4

def randBias(index):
    unnormalized = np.exp(-(((index - (numOfIntervals/2)-weightCoeff)**2)/(2*spreadCoeff)))
    return unnormalized

def allNormalizedBiases():
    biases = []
    for i in range(numOfIntervals):
        biases.append(randBias(i))
    biases = np.array(biases)
    normalized_biases = biases / np.sum(biases)
    return normalized_biases

def setIntervals(intervals):
    global numOfIntervals 
    numOfIntervals = intervals

def changeWeightCoeff(coeff):
    global weightCoeff
    weightCoeff += coeff

def setWeightCoeff(coeff):
    global weightCoeff
    weightCoeff = coeff

def changeSpreadCoeff(coeff):
    global spreadCoeff
    spreadCoeff += coeff

def setSpreadCoeff(coeff):
    global spreadCoeff
    spreadCoeff = coeff

def getRandomNumberFromArray(arr):
    global numOfIntervals
    numOfIntervals = len(arr)
    probabilities = allNormalizedBiases()
    return np.random.choice(arr, p=probabilities)

# Example usage:
# setWeightCoeff(1.0)
# setSpreadCoeff(1.0)
# random_number = getRandomNumberFromArray([1, 2, 3, 4, 5])
# print(random_number)