# ML_Project52-BOW_TFIDF_Xgboost_Update

## BOW TFIDF XGBoost Update
This project aims to explore different text representation techniques such as Bag-of-Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF) combined with the XGBoost algorithm for binary classification of duplicate questions in the Quora dataset.

### Overview
Duplicate question identification is a crucial task in natural language processing (NLP) and has various applications such as question-answering systems, search engines, and online forums moderation. This project utilizes machine learning techniques to automatically identify duplicate questions from the Quora dataset.

### Dataset
The Quora dataset contains pairs of questions along with labels indicating whether they are duplicates or not. The dataset is preprocessed to remove any missing values and irrelevant columns.

### Feature Engineering

Text Cleaning: The questions are preprocessed by removing special characters, punctuation, and non-ASCII characters. Additionally, text normalization techniques such as converting contractions to their expanded forms and replacing numerical values with a generic token are applied.

##### Text Representation:
1.Bag-of-Words (BOW): The questions are transformed into BOW vectors using the CountVectorizer from scikit-learn.

2.TF-IDF: The questions are represented using TF-IDF vectors with different granularity levels (word, n-gram, and character) using the TfidfVectorizer from scikit-learn.

### Model Training
The XGBoost algorithm is employed for classification, which is a popular choice for handling structured/tabular data and often yields competitive results. The model is trained and evaluated using various text representation techniques to compare their performance.

### Results
The performance of the XGBoost model is evaluated using metrics such as precision, recall, and F1-score on both training and validation datasets for different text representation techniques:

##### BOW Model:
```
Training F1-score: 0.6178
Validation F1-score: 0.6154
```
##### Word Level TF-IDF Model:
```
Training F1-score: 0.8493
Validation F1-score: 0.7577
```
##### N-gram Level TF-IDF Model:
```
Training F1-score: 0.7194
Validation F1-score: 0.6747
```
###### Character Level TF-IDF Model:
```
Training F1-score: 0.9845
Validation F1-score: 0.8008
```
The character level TF-IDF model achieved the highest validation F1-score of 0.8008, indicating its effectiveness in capturing meaningful patterns in text data.

### Dependencies
``
Python 3
pandas
scikit-learn
xgboost
seaborn
matplotlib
```

### Usage
Ensure you have all dependencies installed.

Download the Quora dataset (quora_train.csv) or provide your own dataset in a similar format.

Run the provided Jupyter notebook to train and evaluate the models.

Experiment with different hyperparameters, text preprocessing techniques, and model architectures to improve performance.
