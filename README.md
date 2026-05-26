# 🎓 Student Success Prediction — Semi-Supervised ML Pipeline

---

## 📌 Overview

A **semi-supervised machine learning pipeline** that identifies students at risk of failing an introductory course — before the course ends — without using any ground truth labels during training.

The system works by clustering unlabeled student data, using a Subject Matter Expert (SME) to label representative students, propagating those labels to full clusters, and training a supervised classifier on the propagated labels. The final model is deployed in a production notebook that generalizes to unseen datasets.

**Final Production Performance (on unseen data):**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| At-Risk (0) | 0.96 | 0.63 | 0.76 |
| **Success (1)** | **0.78** | **0.98** | **0.87** |
| **Overall Accuracy** | | | **0.83** |

---

## 🎯 Problem Statement

Campus Success Analytics was asked by a team from an introductory course to identify students at risk of failing **prior to the conclusion of the course**, so that early interventions (tutoring, advisor contact, supplementary instruction) can be applied in time.

**The core challenge:** No labels are provided. The pipeline must:
1. Discover structure in unlabeled data through clustering
2. Obtain a limited number of SME labels (budget: ≤500 queries)
3. Propagate those labels to the full dataset
4. Train a production-ready classifier — without ever touching the ground truth

---

## 🔄 Pipeline Workflow

```
Unlabeled Student Data
        ↓
Step 1: EDA & Data Validation
        ↓
Step 2: Feature Engineering (rate-based + composite features)
        ↓
Step 3: KMeans Clustering (K=3)
        ↓
Step 4: SME Labeling (3 representatives + 400 confirmations = 403 queries / 500 budget)
        ↓
Step 5: Label Propagation (cluster-wide)
        ↓
Step 6: Supervised Model Training (3 classifiers, 5-fold CV, Pipelines)
        ↓
Step 7: Hyperparameter Tuning (GridSearchCV, macro F1)
        ↓
Step 8: Save Model → Production Notebook
        ↓
Production: Predict on unseen data → Classification Report
```

---

## 🔬 Exploratory Data Analysis

**Dataset:** 10,000 students, no missing values, no duplicates, all logical constraints validated.

Key EDA insights:

| Insight | Impact |
|---------|--------|
| Positive correlation between attendance and assignment submission | Both captured in engagement features |
| Positive correlation between quiz scores and prior GPA | Combined into `academic_performance` score |
| Negative correlation between late submissions and quiz performance | `late_submission_ratio` is a key predictor |
| Work/commute hours weakly negatively correlated with study time | Captured in `workload_index` |
| Raw counts incomparable without denominators | All features converted to rates |

---

## 🛠️ Feature Engineering

9 engineered features from raw counts, all validated to be in expected ranges with no NaN values:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `attendance_rate` | classes_attended / classes_total | Proportion of classes attended (0–1) |
| `assignment_completion_rate` | submitted / total | Proportion of assignments turned in |
| `quiz_score_rate` | earned / possible | Quiz score as a proportion |
| `practice_exam_rate` | earned / possible | Practice exam score proportion |
| `late_submission_ratio` | late / submitted | Proportion of late submissions |
| `engagement_score` | 0.3×attendance + 0.3×assignments + 0.2×discussions + 0.2×LMS logins | Composite engagement metric |
| `total_burden` | work_hours + (commute_minutes / 60) | Total time burden outside class |
| `academic_performance` | 0.5×quiz + 0.3×practice_exam + 0.2×(GPA/4.0) | Composite academic performance |
| `workload_index` | total_burden / (study_hours + 1) | Ratio of burden to study time |

---

## 🔵 Clustering — Student Group Discovery

**Algorithm:** KMeans (K=3) on scaled engineered features.

**Why K=3?** Directly mirrors how academic advisors categorize students:
- **High-performing** — strong attendance, timely submission, high grades
- **At-risk** — poor attendance, late submissions, low grades
- **Borderline** — average participation, mixed performance

Cluster validation: Box plots confirm significant differentiation across all key features. The at-risk cluster shows the lowest academic performance, attendance rate, and engagement score with the highest late submission ratio.

---

## 🏷️ Semi-Supervised Labeling

SME budget used: **403 / 500 queries**

| Step | Queries | Purpose |
|------|---------|---------|
| 3 representatives (1 per cluster via `np.argmin` on centroid distance) | 3 | Initial cluster labels |
| Additional confirmations on borderline (middle) cluster | 400 | Improve label reliability |
| **Total** | **403** | Under budget ✅ |

**Label propagation:** Each cluster's representative label is assigned to all students in that cluster. Where SME labels conflict with propagated labels, SME labels take precedence.

---

## 📊 Evaluation Metric: Macro F1-Score

**Why not accuracy?**
- An "always predict success" model would score ~57% accuracy while missing all at-risk students
- Both false positives (wasted interventions) and false negatives (missed at-risk students) have real consequences
- Macro F1 weights both classes equally, penalizing models that ignore the minority class

---

## 🤖 Model Training

Three classifiers trained inside **sklearn Pipelines** with 5-fold cross-validation:

| Feature | Reason |
|---------|--------|
| Pipeline (Scaler + Classifier) | Prevents data leakage — scaler only fits on training folds |
| 5-fold Cross-Validation | Robust estimate of generalization performance |
| Macro F1 as scoring metric | Equal weight to both at-risk and success classes |

**Models evaluated:** Logistic Regression, Random Forest, XGBoost

**Tuning:** GridSearchCV over best model's hyperparameter grid, optimizing macro F1.

---

## 🚀 Production Notebook

The production notebook is intentionally **separate** from the training notebook. On deployment day, the professor provides two new URLs (unseen data):

```python
production(
    X_path='URL_to_unlabeled_features.csv',
    y_path='URL_to_true_labels.csv'
)
```

The notebook:
1. Loads `student_success_model.pkl` (full sklearn Pipeline — scaler + classifier)
2. Loads `model_features.pkl` (exact feature list and order)
3. Applies identical feature engineering
4. Generates predictions
5. Prints classification report against true labels

**Production results on held-out test set (2,500 students):**

```
              precision    recall  f1-score   support

 At-Risk (0)       0.96      0.63      0.76      1069
 Success (1)       0.78      0.98      0.87      1431

    accuracy                           0.83      2500
   macro avg       0.87      0.81      0.81      2500
```

---

## 📁 Repository Structure

```
student-success-prediction/
│
├── notebooks/
│   ├── Final_project_FINAL.ipynb              # Full training pipeline
│   ├── FINAL_production_notebook_FINAL.ipynb  # Production notebook (v1)
│   └── FINAL_production_notebook_FINAL_new.ipynb  # Production notebook (v2 — final)
│
├── models/
│   ├── student_success_model.pkl              # Trained sklearn Pipeline
│   └── model_features.pkl                    # Feature list for production
│
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation and feature engineering |
| `scikit-learn` | KMeans, Pipelines, cross-validation, GridSearchCV, classifiers |
| `joblib` | Model serialization for production |
| `matplotlib`, `seaborn` | Cluster visualization, correlation heatmaps |

---

## 🚀 Getting Started

```bash
git clone https://github.com/KHUSHI0809/student-success-prediction.git
cd student-success-prediction
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Run training pipeline
jupyter notebook notebooks/Final_project_FINAL.ipynb

# Run production evaluation
jupyter notebook notebooks/FINAL_production_notebook_FINAL_new.ipynb
```

---

## 🏷️ Topics

`semi-supervised-learning` `student-success-prediction` `kmeans-clustering` `label-propagation` `sklearn-pipeline` `feature-engineering` `educational-analytics` `machine-learning` `python` `scikit-learn` `early-warning-system`
