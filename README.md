# 🏠 House Price Prediction Project

## Overview

This project provides a comprehensive framework for predicting house prices, encompassing everything from initial data exploration and preprocessing to model training, evaluation, and deployment. Our goal is to build a robust and scalable machine learning solution for real estate valuation.

## 📌 Project Objective

The goal is to build a scalable and reproducible ML pipeline to predict housing prices using a dataset of real estate listings. This project is built as a learning experiment to explore the integration of ZenML and MLflow in a real-world workflow.

---

## 🧰 Features

- Clean and reproducible ML pipeline using ZenML
- Modular pipeline steps (ingest, clean, model, evaluate, deploy)
- Integrated with MLflow for experiment tracking
- Model deployment for local inference
- Sample prediction script for testing deployed model

---

## 📂 Project Structure

The project is meticulously organized into logical directories to ensure clarity, maintainability, and efficient workflow management.

### `analysis/`
_Uncover insights from your data!_ This directory houses all scripts and notebooks dedicated to deep-diving into the datasets.
- `analysis_src/`: Core analytical scripts for detailed inspections.
    - `basic_inspection.py`: Quick checks and initial data summaries.
    - `bivariate_analysis.py`: Exploring relationships between two variables.
    - `missing_values.py`: Pinpointing and understanding data gaps.
    - `multivariate_analysis.py`: Unraveling complex relationships among multiple variables.
    - `univariate_analysis.py`: Analyzing individual feature distributions.
      
    - `EDA.ipynb`: Interactive playground for Exploratory Data Analysis, packed with visualizations and statistical summaries.

### `extracted_data/`
_Your raw data repository!_ This is where original or freshly extracted datasets reside.
- `__MACOSX/`: (System-generated, usually safe to ignore or delete. 🗑️)
- `train.csv`: The primary dataset for model training.
- `updated_housing_data.csv`: A refined or newer version of housing data.

### `house_price_data/`
_The heart of our housing datasets!_ Specific datasets tailored for the house price prediction challenge. Took from kaggle.
"https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data"
- `data_description.txt`: A detailed explanation of the dataset features.
- `sample_submission.csv`: A template for submitting your predictions.
- `test.csv`: The dataset used for making predictions and evaluating the final model.
- `train.csv`: The training dataset, specific to the house price prediction task, distinct from general `extracted_data`.

### `pipelines/`
_Automated workflows for seamless execution!_ Contains scripts that orchestrate end-to-end ML processes.
- `deployment_pipeline.py`: The blueprint for pushing your trained model to production. 🚀
- `training_pipeline.py`: The master script for training your machine learning models from start to finish.

### `src/`
_The core intelligence of the project!_ Modularized Python scripts for various functionalities.
- `data_split.py`: Strategically dividing datasets for training and validation.
- `feature_engineering.py`: Crafting powerful features from raw data to enhance model performance. ✨
- `handle_missing_values.py`: Robust strategies for filling in data gaps.
- `handle_outliers.py`: Techniques to identify and manage extreme data points.
- `ingest_data.py`: Efficiently loading and preparing raw data.
- `model_building.py`: Where the magic happens! Scripts for constructing and training ML models. 🧠
- `model_evaluator.py`: Assessing model performance with key metrics. 📊

### `step/`
_Granular control over your ML workflow!_ Individual, sequential steps that combine to form larger pipelines.
- `_0_predictor.py`: The initial step for making predictions.
- `_1_data_ingestion_step.py`: The first step in data processing: getting the data in!
- `_2_handle_the_missing_value_s...`: The crucial step for addressing missing information.
- `_3_outlier_detection_step.py`: Identifying and processing anomalous data.
- `_4_data_splitter_step.py`: Segmenting data for robust training and testing.
- `_5_feature_engineering.py`: The step where raw data transforms into model-ready features.
- `_6_model_building_step.py`: The actual model construction phase.
- `_7_model_evaluator_step.py`: Measuring how well your model performs.
- `_8_dynamic_importer.py`: A utility for dynamic module loading.
- `_9_prediction_service_load...`: The final step for loading the prediction service.

---

## 🎯 Root Level Files

These files are essential for project setup, management, and documentation.

- `.gitignore`: Keeps our repository clean by ignoring unnecessary files.
- `LICENSE`: Details the terms under which this project is licensed.
- `README.md`: **You're reading it!** Your guide to the project.
- `index.ipynb`: A central Jupyter notebook, possibly serving as a main entry point or dashboard.
- `mlflow.db`: The database for MLflow tracking, logging experiments, parameters, and models. 📈
- `requirements.txt`: All necessary Python dependencies listed here.
- `run_deployment.py`: The script to kickstart the deployment of your model.

---

## 📄 License

This project is open-sourced under the **[Apache License](LICENSE)**. See the `LICENSE` file for full details.

---

_Happy Predicting!_ 🏡✨
