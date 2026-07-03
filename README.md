# Spam Detection using Machine Learning

Classifying SMS messages as **Spam** or **Ham (Not Spam)** using TF-IDF feature extraction and a comparison of classic ML models.

## Project Overview

Email/SMS spam affects communication and productivity. This project builds a machine learning pipeline that classifies text messages as spam or ham.

**Workflow:**
1. Load the dataset
2. Clean the data
3. Exploratory Data Analysis (EDA)
4. Text preprocessing
5. Feature extraction (TF-IDF)
6. Model training
7. Model evaluation & comparison

---

## Dataset

The [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (`spam.csv`), loaded with `latin-1` encoding, containing **5,572 messages**.

- Dropped extra empty columns, keeping only the label and message text
- Renamed columns to `label` and `message`
- Mapped labels to numeric form: `ham → 0`, `spam → 1`

### Class Distribution

![Ham vs Spam Distribution](images/ham_vs_spam_distribution.png)

The dataset is imbalanced, with far more ham messages than spam — a common trait of real-world spam datasets.

---

## Text Preprocessing

A `clean_text()` function normalizes each message:
- Lowercases all text
- Replaces URLs with the token `url`
- Replaces numbers with the token `number`
- Strips punctuation
- Collapses extra whitespace

## Train/Test Split

An 80/20 stratified split preserves the ham/spam ratio in both sets:

- **Training set:** 4,457 messages
- **Testing set:** 1,115 messages

![Train vs Test Class Distribution](images/train_test_class_distribution.png)

---

## Feature Extraction — TF-IDF

Raw text is converted into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**:

- English stop words removed
- Unigrams **and** bigrams considered (`ngram_range=(1,2)`)
- Words appearing in fewer than 2 documents ignored (`min_df=2`)

This produced a vocabulary of **7,116 features**.

### Top TF-IDF Features (Average Weight)

![Top TF-IDF Features](images/top_tfidf_features.png)

---

## Model — Multinomial Naive Bayes

The primary model is **Multinomial Naive Bayes**, a strong baseline for text classification, trained with a smoothing parameter of `alpha=0.3` to avoid zero probabilities for rare words.

### Results

| Metric | Score |
|---|---|
| Accuracy | 98.39% |
| Precision | 99.25% |
| Recall | 88.59% |
| F1-Score | 93.62% |

```
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       966
        spam       0.99      0.89      0.94       149

    accuracy                           0.98      1115
   macro avg       0.99      0.94      0.96      1115
weighted avg       0.98      0.98      0.98      1115
```

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

---

## Model Comparison

Four models were trained and evaluated on identical TF-IDF features:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest

| Model | Accuracy | Precision | Recall | F1 | Train Time (s) |
|---|---|---|---|---|---|
| **Linear SVM** | 98.65% | 95.89% | 93.96% | **94.92%** | 0.006 |
| Multinomial Naive Bayes | 98.39% | 99.25% | 88.59% | 93.62% | 0.001 |
| Random Forest | 97.94% | 100.00% | 84.56% | 91.64% | 0.884 |
| Logistic Regression | 97.13% | 92.09% | 85.91% | 88.89% | 0.035 |

**Linear SVM** achieved the best F1-score, edging out Naive Bayes, with Random Forest showing perfect precision but the lowest recall (missing more spam messages) and by far the slowest training time.

### F1-Score Comparison

![Model Comparison — F1 Score](images/model_comparison_f1.png)

### Full Metric Comparison

![Model Comparison — All Metrics](images/model_comparison_all_metrics.png)

---

## Most Predictive Words

Using the log-probability difference between spam and ham classes from the Naive Bayes model, the words most strongly associated with each class:

![Top Spam vs Ham Words](images/top_spam_ham_words.png)

---

## Live Prediction Function

A helper function `predict_sms()` cleans, vectorizes, and classifies a new message, printing both the predicted label and the spam probability.

**Example:**
```
Message : hi i am aliza
Result  : ✅ HAM (not spam)
Spam probability: 2.83%
```

---

## Tech Stack

- **Python** — pandas, numpy
- **scikit-learn** — TfidfVectorizer, MultinomialNB, LogisticRegression, LinearSVC, RandomForestClassifier
- **matplotlib / seaborn** — visualization

## Key Takeaways

- TF-IDF with unigrams + bigrams provides strong signal for spam detection even with simple models.
- Naive Bayes gives very high precision (few false spam flags) but misses more spam (lower recall) than Linear SVM.
- Linear SVM offers the best overall balance (highest F1) while still training almost instantly.
- Random Forest, despite perfect precision, is the weakest on recall and by far the most expensive to train — not worth it here.
