# Credit-Card-fraud-detection-system-using-Machine-Learning
Detect Fraudulent Credit Card transactions using different Machile Learning models and compare performances

In this notebook, I explore various Machine Learning models to detect fraudulent use of Credit cards. I compare each model performance and results. The best performance is achieved using SMOTE technique.

# Problem Statement

In this project we want to identify fraudulent transactions with Credit Cards.
Our objective is to build a Fraud detection system using Machine learning techniques.
In the past, such systems were rule-based. Machine learning offers powerful new ways.

The project uses a dataset of 300,000 fully anonymized transactions. Each transation is labelled either fraudulent or not fraudulent.
Note that prevalence of fraudulent transactions is very low in the dataset. Less than 0.1% of the card transactions are fraudulent. This means that a system predicting each transaction to be normal can reach an accuracy of over 99.9% despite not detecting any fraudulent transaction. This will necessitate adjustment techniques.

# Techniques used in the project
The project compares the results of different techniques :
- Machine learning techniques:
  - Random Forest
  - Decision Trees
- Deep Learning techniques:
  - Neural network using fully connected layers.

Performance of the neural network is compared for different optimization approaches:
- plain binary cross-entropy loss minimization
- minimization using weights to compensate for the class imbalance
- down-sampling of the non-fraudulent class to match the fraudulent class
- up-sampling of the fraudulent class to match the non-fraudulent one by implementing SMOTE technique. The SMOTE method allows to generate a new vector using 2 existing datapoints.

# Results

The best results are achieved by up-sampling the under-represented fraudulent class using SMOTE (synthetic minority oversample technique).
With this approach, the model is able to detect 100% of all fraudulent transactions in the unseen test set. This fully satisfies the primary objective to detect all abnormal transactions (or the vast majority). The technique and model implemented remain simple and can be updated in real-time.

In addition, the number of false positive remains acceptable. This means a lot less verification work (on legitimate transactions) for the fraud departement. 

Confusion matrix achived using SMOTE up-sampling and a simple dense neural network:

![](confusion_matrix.png)

Comparison of key performance indicators between the tested approaches:

![](benchmark.png)

