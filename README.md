# heart-disease-ml-app
Logistic Regression Metrics
Accuracy: 0.8097560975609757
AUC: 0.9298095238095238
Precision: 0.7619047619047619
Recall: 0.9142857142857143
F1 Score: 0.8311688311688312
MCC: 0.630908308763638


Decision Tree Metrics (Regularized)
Accuracy: 0.8878048780487805
AUC: 0.9584761904761905
Precision: 0.8942307692307693
Recall: 0.8857142857142857
F1 Score: 0.8899521531100478
MCC: 0.775566572814591

KNN Metrics
Accuracy: 0.8634146341463415
AUC: 0.9629047619047618
Precision: 0.8737864077669902
Recall: 0.8571428571428571
F1 Score: 0.8653846153846154
MCC: 0.7269351910363394

Naive Bayes Metrics
Accuracy: 0.8292682926829268
AUC: 0.9042857142857142
Precision: 0.8070175438596491
Recall: 0.8761904761904762
F1 Score: 0.8401826484018264
MCC: 0.6601634114374199

Random Forest Metrics (Regularized)
Accuracy: 0.9024390243902439
AUC: 0.9714285714285714
Precision: 0.8899082568807339
Recall: 0.9238095238095239
F1 Score: 0.9065420560747663
MCC: 0.8051910364710595

XGBoost Metrics
Accuracy: 0.9902439024390244
AUC: 1.0
Precision: 0.9813084112149533
Recall: 1.0
F1 Score: 0.9905660377358491
MCC: 0.9806539873934406

a. Problem Statement : 

The objective of this assignment is to build and evaluate multiple machine learning classification models to predict the presence of heart disease in a patient based on clinical and physiological attributes. The task involves implementing different classification algorithms, comparing their performance using standard evaluation metrics, and deploying the trained models using an interactive Streamlit web application.

This assignment demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, comparison, and deployment.




b. Dataset Description

The Heart Disease dataset was used for this assignment. The dataset was obtained from a public repository (Kaggle/UCI) and contains clinical data collected from patients for heart disease diagnosis.

Dataset Characteristics:

Number of instances: 1025
Number of features: 13
Target variable: target
    0 → No heart disease
    1 → Presence of heart disease
Problem type: Binary classification
Missing values: None
Feature type: Numerical (no categorical encoding required)

The dataset includes attributes such as age, cholesterol level, resting blood pressure, maximum heart rate, chest pain type, and other medically relevant indicators.



c. Models Used and Evaluation Metrics

The following six classification models were implemented and evaluated on the same dataset using a consistent train-test split and preprocessing pipeline.

| ML Model Name             | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression       | 0.810    | 0.930 | 0.762     | 0.914  | 0.831    | 0.631 |
| Decision Tree             | 0.888    | 0.958 | 0.894     | 0.886  | 0.890    | 0.776 |
| kNN                       | 0.863    | 0.963 | 0.874     | 0.857  | 0.865    | 0.727 |
| Naive Bayes               | 0.829    | 0.904 | 0.807     | 0.876  | 0.840    | 0.660 |
| Random Forest (Ensemble)  | 0.902    | 0.971 | 0.890     | 0.924  | 0.907    | 0.805 |
| XGBoost (Ensemble)        | 0.990    | 1.000 | 0.981     | 1.000  | 0.991    | 0.981 |





d. Model Performance Observations



| ML Model Name            | Observation about Model Performance                                 |
| ------------------------ | ------------------------------------------------------------------- |
| Logistic Regression      | Logistic Regression provided a strong baseline with high recall and AUC, indicating good sensitivity. However, its linear nature limited overall accuracy compared to more complex models.                                            |
| Decision Tree            | The regularized Decision Tree achieved balanced performance with improved generalization after controlling tree depth and minimum samples per split, reducing overfitting seen in the unconstrained model.                            |
| kNN                      | The k-Nearest Neighbors model showed stable and balanced performance with good accuracy and F1 score. Its performance benefited significantly from feature scaling but may be computationally expensive for larger datasets.          |
| Naive Bayes              | Naive Bayes performed reasonably well with high recall, making it effective for identifying positive cases. However, its strong feature independence assumption limited overall accuracy and precision.                               |
| Random Forest (Ensemble) | After regularization, Random Forest demonstrated strong and consistent performance across all metrics. The ensemble bagging approach reduced variance and improved generalization compared to a single decision tree.                 |
| XGBoost (Ensemble)       | XGBoost achieved the best overall performance, with near-perfect accuracy, AUC, and MCC. The boosting mechanism and regularization allowed the model to capture complex patterns effectively while maintaining strong generalization. |



