# Credit_Risk_Analysis

## Introduction
In this challenge we are going to apply machine learning to solve a real-world challenge, credit card risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans.
   Here we are going to use the credit card credit dataset from LendingClub, a peer-to-peer lending services company. We’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, we compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Finally we evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.
## Analysis
## Credit Risk Resampling Techniques
We evaluate three machine learning models by using resampling to determine which is better at predicting credit risk using imbalanced-learn and scikit-learn libraries.  First, we use the oversampling RandomOverSampler and SMOTE algorithms, and then  use the undersampling ClusterCentroids algorithm. First we resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

### Naive Random Oversampling
<img width="765" alt="Screen Shot 2022-04-24 at 11 08 40 AM" src="https://user-images.githubusercontent.com/72629108/164990548-7ddcf89a-7d95-4dfe-91cd-02e7be3b8925.png">
### SMOTE Oversampling
<img width="774" alt="Screen Shot 2022-04-24 at 11 09 06 AM" src="https://user-images.githubusercontent.com/72629108/164990562-e9ad2401-41f7-4709-a8cb-820aa7faf0fb.png">
### ClusterCentroids resampler
<img width="774" alt="Screen Shot 2022-04-24 at 11 09 56 AM" src="https://user-images.githubusercontent.com/72629108/164990568-e88a087d-0c1e-4f4e-834b-d483150c6c61.png">

### SMOTEENN
Next we use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk. First, we resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report
<img width="787" alt="Screen Shot 2022-04-24 at 11 10 26 AM" src="https://user-images.githubusercontent.com/72629108/164990574-835b6e07-e51c-43df-bbb3-7df195bb3b7f.png">


## Credit risk ensemble
Using your knowledge of the imblearn.ensemble library, we train and compare two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. . First we resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
## Balanced Random Forest Classifier
<img width="788" alt="Screen Shot 2022-04-24 at 11 23 37 AM" src="https://user-images.githubusercontent.com/72629108/164990836-c3dccc7e-f529-4cdb-a286-73cfc6e7def9.png">
## EasyEnsembleClassifier
<img width="780" alt="Screen Shot 2022-04-24 at 11 24 04 AM" src="https://user-images.githubusercontent.com/72629108/164990854-aeb9b402-e7b7-4fc5-9ff8-7680ca9e549b.png">
