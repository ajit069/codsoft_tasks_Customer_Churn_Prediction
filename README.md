#  Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to leave (churn) a subscription-based service using historical customer data. This project compares multiple classification algorithms to determine the best-performing model.

---

##  Project Overview

Customer churn prediction is an important business problem. Retaining existing customers is often more cost-effective than acquiring new ones. This project uses customer demographic and account information to predict whether a customer will exit the service.

---

##  Features

- Data preprocessing and cleaning
- Label encoding for categorical features
- Feature scaling using StandardScaler
- Train-test split for model evaluation
- Multiple Machine Learning algorithms:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
- Performance evaluation using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Model accuracy comparison using a bar chart

---

##  Project Structure

```
Customer_Churn_Prediction/
│
├── Churn_Modelling.csv
├── churn_prediction.py
├── README.md
└── requirements.txt
```

---

##  Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

##  Dataset

The project uses the **Churn Modelling Dataset**, which contains customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

### Target Variable

- **Exited**
  - **0** → Customer Stayed
  - **1** → Customer Churned

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Customer_Churn_Prediction.git
```

Move into the project folder:

```bash
cd Customer_Churn_Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

##  Run the Project

```bash
python churn_prediction.py
```

---

##  Machine Learning Models

The following algorithms are implemented:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Each model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

##  Expected Results

| Model | Expected Accuracy |
|--------|-------------------|
| Logistic Regression | 80% - 82% |
| Random Forest | 86% - 88% |
| Gradient Boosting | 87% - 89% |

> Actual results may vary slightly depending on the dataset split and library versions.

---

##  Output

The program displays:

- Model Accuracy
- Confusion Matrix
- Classification Report
- Accuracy Comparison Graph

---

##  Requirements

Create a `requirements.txt` file with the following:

```text
pandas
numpy
matplotlib
scikit-learn
```

Or install manually:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

##  Future Improvements

- Hyperparameter tuning
- Cross-validation
- XGBoost implementation
- Feature importance visualization
- ROC Curve and AUC Score
- Deploy using Flask or Streamlit

---

##  Author

Developed by **ajit069**

GitHub: https://github.com/your-username

---

## ⭐ If you found this project useful, consider giving it a Star!
