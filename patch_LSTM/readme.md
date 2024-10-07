# Photovoltaic Power Generation Prediction Using Deep Learning

This repository contains the implementation and analysis of photovoltaic (PV) power generation prediction using deep learning techniques. The project explores short-term and long-term forecasting methods to improve the accuracy of PV power generation predictions, leveraging minimal features from available datasets.

## Table of Contents
- [Motivation](#motivation)
- [Experiment](#experiment)
  - [Project Statement](#project-statement)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Forecasting Methods](#forecasting-methods)
- [Conclusion](#conclusion)
- [Usage](#usage)

## Motivation
With the increasing importance of clean and renewable energy sources, accurately predicting photovoltaic power generation is crucial. This project aims to improve the accuracy of these predictions using AI, specifically deep learning models, by analyzing time-series data from solar energy sensors.

## Experiment
### Project Statement
The goal of this project is to develop a deep learning-based prediction system that can forecast both short-term and long-term PV power generation with high accuracy, using minimal sensor data.

### Dataset
We use PV sensor data from Zhejiang Taizhou. The dataset contains over 15,000 rows and 29 columns, including timestamp, power generation, temperature, weather, and other relevant features.

### Model Architecture
Our model primarily uses a **Patch-LSTM** architecture for time-series prediction, with comparisons to **ARIMA** and other deep learning models such as MLP and DLinear.

### Forecasting Methods
- **Short-term Forecasting**: Patch-LSTM outperforms ARIMA, especially near zero values, although some deviations occur near peak values.
- **Long-term Forecasting**: Patch-LSTM performs significantly better than both traditional deep learning models (MLP) and the more recent DLinear model.

## Conclusion
In both short-term and long-term forecasting, Patch-LSTM demonstrated superior performance compared to ARIMA, MLP, and DLinear models, especially when using minimal features such as weather or historical data.

## Usage
To run this project, clone the repository and follow the installation instructions provided in the `requirements.txt` file.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
python my_run.py
