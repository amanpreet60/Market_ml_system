import logging
import sys
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder


import mlflow

# Temporary solution for cross-folder imports
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

# ZenML imports
from zenml import step, Model, ArtifactConfig
from zenml.client import Client

# Your custom modules
from src.ingest_data import ZipDataIngestor, CsvDataIngestor
from src.handle_missing_values import fill_missing_values, remove_missing_value
from src.handle_outliers import feature_skew_kurt
from src.data_split import SimpleTrainTestSplit
from src.feature_engineering import (
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
    drop,
)
from src.model_evaluator import RegressionModelEvaluation
from analysis.analysis_src.univariate_analysis import hist_plot

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
