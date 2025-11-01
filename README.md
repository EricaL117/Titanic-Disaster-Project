# Titanic Survival Prediction Using Docker (Python & R)

This project predicts whether a passenger survived the Titanic disaster using logistic regression models implemented in both **Python** and **R**. Each workflow is containerized with **Docker**.



## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Python Container](#running-the-python-container)
- [Running the R Container](#running-the-r-container)



## Overview
This repository contains two Docker-based implementations of the Titanic survival prediction model — one written in Python and one in R. Both versions follow the same steps:

1. Load and preprocess the Titanic training dataset (`train.csv`)
2. Train a logistic regression model
3. Predict survival outcomes for unseen passengers (`test.csv`)
4. Save the prediction results to a CSV file



## Project Structure
```bash
Titanic-Disaster-Project/
├─ src/
│ ├─ data/ # place train.csv and test.csv here
│ ├─ python/
│ │ ├─ main.py
│ │ ├─ requirements.txt
│ │ └─ Dockerfile
│ └─ R/
│ ├─ main.R
│ ├─ install_packages.R
│ └─ Dockerfile
├─ .gitignore
└─ README.md
```


## Installation and Setup

### Prerequisites
- Docker Desktop installed and running  
- Terminal or command-line access  
- Titanic dataset files (`train.csv`, `test.csv`)

### Download the Data
1. Download both files from the Kaggle [**Titanic: Machine Learning from Disaster**](https://www.kaggle.com/competitions/titanic/data).
2. Place them inside this directory:
   src/data/



## Running the Python Container

### Build the Docker image
```bash
docker build -f src/python/Dockerfile -t titanic-python:latest src/python
```
### Run the container
```bash
docker run --rm -v "$pwd/src/data":/app/../data titanic-python:latest
```
Output:
src/data/prediction_Python.csv



## Running the R Container

### Build the Docker image
```bash
docker build -f src/R/Dockerfile -t titanic-r:latest src/R
```
### Run the container
```bash
docker run --rm -v "$pwd/src/data":/app/../data titanic-r:latest
```
Output:
src/data/prediction_R.csv

---

After placing the datasets and building the containers, both Python and R versions will run end-to-end inside Docker and save the prediction files in src/data/.
