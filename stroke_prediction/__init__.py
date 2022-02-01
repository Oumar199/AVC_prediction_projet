"""Package contenant l'ensemble des traitements effectués pour le projet
"""
# Let's import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from pandas_profiling import ProfileReport
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import plot_roc_curve, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import joblib
import pickle