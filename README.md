# 01_SportsPrediction

Intro to AI mid-sem project.

## Team Members

- John Anatsui Edem.
- David Saah.

## Overview of Problem

In sports prediction, large numbers of factors including the historical performance of the teams, results of matches, and data on players, have to be accounted for to help different stakeholders understand the odds of winning or losing.

In this project, we are tasked to build a model(s) that predict a player's overall rating given the player's profile.

## Milestones: ML Life Cycle

### Data preparation & feature extraction

- [x] Data collection and labelling.
  - [x] Acquire data.
- [x] Data cleaning.
  - [x] Imputing missing values.
- [x] Data processing.
  - [x] Feature selection.
  - [x] Feature subsetting.
  - [x] Normalising data.
  - [x] Scaling data.

### Model engineering

- [x] Get training & testing data.
- [x] Train the model with cross-validation.
- [x] Test the accuracy of the model.
- [x] Fine tune model (optimisation).
- [x] Use different models.
  - [x] Train 3 models.
- [x] Perform ensembling.

## Directory Structure

- **data:** Datasets
  - players_21.csv -> training data.
  - players_22.csv -> testing data.
- **src:** Source codes for model training. (.py and .ipynb files)
- **app:** Source code for model deployment.
- **demo:** Demo video.

## Chosen Features

1. potential
2. wage_eur
3. passing
4. dribbling
5. attacking_short_passing
6. movement_reactions
7. power_shot_power
8. mentality_vision
9. mentality_composure

## Model(s) Used

- XGBoost Regressor
- Random Forest Regressor
- AdaBoost Regressor

### Reasons for choosing the XGBoost Regressor

- Random forest model is very large compared tp XGBoost and AdaBoost.
- XGBoost and AdaBoost have similar performance, but XGBoost is performs better.
  - R-squared score for XGBoost is 0.94 while that of AdaBoost is 0.86.
- XGBoost is the best model for this dataset.

## Deployment

- Website link: [insert web link here]

## Video Demo

[Insert video here]
