# FILE: BiasSimulator.py

import numpy as np
import matplotlib.pyplot as plt
from randBias import setIntervals, setWeightCoeff, setSpreadCoeff, allNormalizedBiases

class BiasSimulator:
    def __init__(self, intervals, weight_coeff, spread_coeff):
        setIntervals(intervals)
        setWeightCoeff(weight_coeff)
        setSpreadCoeff(spread_coeff)
        self.biases = allNormalizedBiases()
    
    def simulate_selection(self, num_selections):
        choices = np.arange(len(self.biases))
        probabilities = np.array(self.biases).flatten()
        selected_values = np.random.choice(choices, size=num_selections, p=probabilities)
        return selected_values

    def run_simulations(self, num_simulations=10000):
        selections = self.simulate_selection(num_simulations)
        counts = np.bincount(selections, minlength=len(self.biases))
        return counts

    def plot_results(self, counts):
        intervals = np.arange(len(self.biases))
        plt.bar(intervals, counts)
        plt.xlabel('Intervals')
        plt.ylabel('Number of Selections')
        plt.title('Selection Distribution')
        plt.show()

# Example usage:
if __name__ == "__main__":
    simulator = BiasSimulator(intervals=20, weight_coeff=0, spread_coeff=4)
    counts = simulator.run_simulations(num_simulations=10000)
    simulator.plot_results(counts)