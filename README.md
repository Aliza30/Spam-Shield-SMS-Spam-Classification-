# 📩 SMS Spam Detection using Machine Learning

Ever wondered how your messaging app knows when something is spam? 🤔

This project builds a machine learning pipeline that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and classic machine learning algorithms. From cleaning raw text to comparing multiple models, this project covers the complete workflow of an NLP classification task.

---

# 🚀 Project Overview

Spam texts are everywhere—from fake lottery wins to suspicious links. The goal of this project is to automatically identify spam messages while keeping genuine conversations safe.

The project walks through the entire machine learning pipeline:

- 📥 Data collection and cleaning
- 📊 Exploratory Data Analysis (EDA)
- 🧹 Text preprocessing
- 🔤 TF-IDF feature extraction
- 🤖 Model training
- 📈 Performance evaluation
- ⚖️ Model comparison
- 💬 Predicting new SMS messages

---

# 📂 Dataset

This project uses the **SMS Spam Collection Dataset**, which contains **5,572 SMS messages** labeled as either **Spam** or **Ham**.

### Dataset Summary

| Category | Count |
|-----------|------:|
| Total Messages | 5,572 |
| Ham | 4,825 |
| Spam | 747 |

### Data Cleaning

Before training the models, the dataset was cleaned by:

- Removing unnecessary columns
- Renaming the columns to `label` and `message`
- Encoding labels (`ham → 0`, `spam → 1`)
- Removing duplicate and missing records

---

# 🧹 Text Preprocessing

Raw text isn't something machine learning models can understand directly, so each message goes through several preprocessing steps:

- Convert text to lowercase
- Replace URLs with `url`
- Replace numbers with `number`
- Remove punctuation
- Remove extra spaces
- Tokenize text
- Remove English stop words

This helps reduce noise and improve the quality of the features.

---

# ✂️ Train-Test Split

To ensure fair evaluation, the dataset was split using an **80:20 stratified ratio**, preserving the original spam-to-ham distribution.

| Dataset | Messages |
|----------|---------:|
| Training | 4,457 |
| Testing | 1,115 |

---

# 🔤 Feature Engineering

The cleaned messages were transformed into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

### Configuration

- English stop words removed
- Unigrams + Bigrams
- `min_df = 2`

This produced a vocabulary containing more than **7,000 meaningful text features**.

---

# 🤖 Models Trained

Four machine learning models were trained and compared:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Linear Support Vector Machine (Linear SVM)

---

# 📊 Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------:|----------:|--------:|---------:|
| 🥇 Linear SVM | **98.65%** | 95.89% | **93.96%** | **94.92%** |
| 🥈 Multinomial Naive Bayes | 98.39% | **99.25%** | 88.59% | 93.62% |
| 🥉 Random Forest | 97.94% | **100.00%** | 84.56% | 91.64% |
| Logistic Regression | 97.13% | 92.09% | 85.91% | 88.89% |

---

# 🏆 Best Model

**Linear SVM** came out on top.

Why?

- Highest F1-score
- Great balance between precision and recall
- Fast training time
- Strong performance on unseen messages

While Multinomial Naive Bayes achieved slightly better precision, Linear SVM delivered the strongest overall performance.

---

# 📈 Confusion Matrix

The Multinomial Naive Bayes model produced:

- ✅ True Ham: **965**
- ❌ False Spam: **1**
- ❌ Missed Spam: **17**
- ✅ True Spam: **132**

Overall Performance:

- Accuracy → **98.39%**
- Precision → **99.25%**
- Recall → **88.59%**
- F1 Score → **93.62%**

---

# 💬 Try It Yourself

```python
predict_sms("Congratulations! You won a free iPhone.")
```

Output

```
Spam
```

Another example:

```python
predict_sms("Hey, are we still meeting tomorrow?")
```

Output

```
Ham
```

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# 📁 Project Structure

```
SMS-Spam-Detection/
│
├── spam_detection.ipynb
├── spam.csv
├── spam_cleaned.csv
├── README.md
└── requirements.txt
```

---

# ⚡ Getting Started

### Clone the repository

```bash
git clone https://github.com/your-username/SMS-Spam-Detection.git
```

### Move into the project folder

```bash
cd SMS-Spam-Detection
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook spam_detection.ipynb
```

---

# 🎯 Key Results

- Achieved **98%+ accuracy** on SMS spam classification.
- Compared four popular machine learning algorithms.
- Linear SVM delivered the best overall performance.
- TF-IDF proved highly effective for representing SMS text.
- Built a reusable prediction function for classifying new messages.

---

# 🔮 Future Improvements

Some cool next steps:

- 🚀 Fine-tune model hyperparameters
- 🧠 Try Deep Learning models (LSTM/GRU)
- 🤖 Experiment with BERT or other Transformer models
- 🌐 Deploy as a Flask/FastAPI API
- 🎨 Build a Streamlit web app for real-time predictions

---

# 💡 Final Thoughts

This project shows that you don't always need complex deep learning models to solve real-world NLP problems. With thoughtful preprocessing, TF-IDF feature engineering, and well-chosen machine learning algorithms, it's possible to build a highly accurate spam detection system that's both fast and reliable.

---

# 👩‍💻 Author

**Aliza Razi**

If you found this project interesting, feel free to ⭐ the repository or connect with me on LinkedIn!