from __future__ import division, print_function, unicode_literals
from Chapt4.linearRegression import *

import numpy as np
import matplotlib.pyplot as plt

eta = 0.1 # learning rate
n_iterations = 1000
m = 100

theta = np.random.rand(2,1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients