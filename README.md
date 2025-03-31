![Data Loading](./Data/images/chase-top-banner3.png)


# chase

*Author: Ajay Varma*  
*Date: 2025-MAR-29*  
*Project Directory: `/Users/ajayvarma/Documents/VS Code/Workspace/Data Science/Projects/Chase/`*  
*Type: Portfolio Project*

## <span style="color:#1979d5">Customer Churn Prediction</span>

---

### Project Overview

#### Objective
The aim of this project is to develop a machine learning model capable of predicting customer churn for a bank. By identifying customers who are likely to leave, the bank can take proactive measures to retain them, improve customer satisfaction, and reduce revenue loss.

#### Dataset
The dataset contains customer information such as:
- **Demographics**: Age, gender, geography
- **Account Information**: Balance, credit score
- **Product Usage**: Number of products, credit card ownership
- **Engagement Metrics**: Activity status

The target variable, **Exited**, indicates whether the customer churned (1) or remained with the bank (0).

#### Methodology

1. **Data Preprocessing**:  
   The dataset was cleaned by removing irrelevant columns, handling missing values, and eliminating duplicates. 

2. **Feature Engineering**:  
   Categorical variables like **Geography** and **Gender** were encoded using **One-Hot Encoding** and **Binary Encoding**, respectively.

3. **Class Imbalance Handling**:  
   Oversampling was applied to the minority class (churned customers) to balance the dataset and prevent the model from being biased towards the majority class.

4. **Model Development**:  
   Multiple classification algorithms, including **Logistic Regression**, **Random Forest**, **Decision Tree**, and **XGBoost**, were implemented and compared to identify the best performer.

5. **Feature Scaling**:  
   **Standardization** was applied to ensure that all features contributed equally to the model’s predictions.

6. **Model Evaluation**:  
   Models were evaluated using metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrices** to assess performance from different perspectives.

---

### Key Performance Metrics

- **Accuracy**: Measures how well the model predicts both churned and non-churned customers.
- **Precision**: The proportion of correctly predicted churned customers among all predicted churned customers.
- **Recall**: The ability of the model to identify actual churned customers.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure of model performance.

---

### Tools and Libraries Used:
- **Python Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
- **Data Science Techniques**: Data preprocessing, feature engineering, class imbalance handling, model training and evaluation, feature importance analysis, cross-validation


---

### Results

- Among the models tested, **Random Forest** was the standout performer, with the highest **accuracy** (95.26%) and **recall** (98.49%).
- **Feature importance analysis** revealed that key predictors of churn include **Age**, **Balance**, and **Geography**.
- Cross-validation confirmed that **Random Forest** generalizes well across different subsets of the data, with a mean accuracy of **94.69%**.

---

### Next Steps / Recommendations to Improve Recall and Overall Model Performance:

1. **Random Forest as the Best Model**:
   - The **Random Forest** model exhibited exceptional performance with **accuracy of 95.26%**, **recall of 98.49%**, and **F1-score of 95.41%**. It is recommended to prioritize **Random Forest** for further fine-tuning and deployment.
   - **Cross-validation results** (mean accuracy of **94.69%**) demonstrate its stability and generalizability across different data subsets.

2. **Hyperparameter Tuning**:
   - To further improve the **Random Forest** model’s performance, hyperparameters like `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` can be fine-tuned using **GridSearchCV** or **RandomizedSearchCV**.
   - Fine-tuning should focus on improving recall and **F1-score**.

3. **Feature Engineering Based on Feature Importance**:
   - **Age**, **Balance**, and **EstimatedSalary** emerged as the most influential features. Exploring transformations such as **non-linear scaling** or **binning** for these variables could enhance predictive power.
   - **Geography** and **Gender** had lower importance, but experimenting with interaction terms (e.g., combining **Geography** with **Age**) might reveal new insights.

4. **Addressing Class Imbalance**:
   - **Random Forest** has excellent recall, but to improve precision without sacrificing recall, techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** or **SMOTETomek** (which combines SMOTE with Tomek links) can be applied.
   - **Undersampling** the majority class (non-churned customers) is another option, though it risks losing valuable information.

5. **Threshold Adjustment**:
   - Since recall is a priority, experimenting with lowering the decision threshold could increase recall at the expense of precision. This adjustment helps capture more churned customers, though it may increase false positives.

6. **Model Comparison and Ensemble Methods**:
   - **XGBoost** and **Decision Tree** models showed strong performance (recall values of 94.29% and 98.43%, respectively). Exploring **stacking** or **bagging** ensemble techniques could combine the strengths of these models, improving overall performance.

7. **Cross-Validation and Generalization**:
   - The **Random Forest** model has robust **cross-validation** results, indicating it generalizes well. To further ensure the model's performance across different data splits, increasing the number of cross-validation folds (e.g., from 10 to 20) could provide more granular insights.

8. **Model Deployment & Real-Time Monitoring**:
   - Given the model’s impressive performance, it is recommended to deploy **Random Forest** into production. An **automated monitoring system** should be set up to track key metrics such as **recall**, **precision**, and **F1-score** in real-time to detect model drift.
   - **Model retraining**: Periodic retraining using new data will be essential to ensure the model adapts to changing customer behaviors and maintains its predictive power over time.

---

### Conclusion

- **Random Forest** proved to be the most effective model for predicting customer churn, balancing **precision** and **recall**.
- **Recall optimization** ensures that churned customers are effectively captured, minimizing missed opportunities for retention efforts.
- By implementing **hyperparameter tuning**, **advanced feature engineering**, and **ensemble methods**, further improvements can be made to the model.
- Once optimized, the model can be deployed in production with real-time monitoring and retraining to maintain its efficacy in detecting customer churn.

---

![Data Loading](./Data/images/chase-bottom-banner.png)
