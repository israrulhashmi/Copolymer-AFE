#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:32:46 2023

@author: tarak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score


np.random.seed(42)
tf.random.set_seed(42)
# Load predictor data from the new directory (modify the file path)
predictor_data = pd.read_csv('/home/tarak/hashmi/afedata/predictor.txt',header=None)
target_data = np.load('/home/tarak/hashmi/Ising_Model/Energy.npy')


#%%%
# Function to train and evaluate the model for a given number of data points

