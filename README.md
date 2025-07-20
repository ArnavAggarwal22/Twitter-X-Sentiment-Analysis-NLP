# Twitter Sentiment Analysis

This project performs sentiment analysis on a large dataset of tweets to classify them as either positive or negative. It utilizes Natural Language Processing (NLP) techniques and a Logistic Regression model to achieve this classification.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)

## 📝 Project Overview

The goal of this project is to build a machine learning model capable of understanding the sentiment expressed in a tweet. The model is trained to distinguish between positive and negative sentiments, a common task in NLP and social media analytics. This can be used for various applications like brand monitoring, public opinion tracking, and customer feedback analysis.

## 📊 Dataset

This project uses the **Sentiment140 dataset**, which is publicly available on Kaggle.

- **Total Number of Tweets:** 1,600,000
- **Labels:** The dataset is balanced with:
    - **800,000** negative tweets (labeled as `0`)
    - **800,000** positive tweets (labeled as `1`)
- **Source:** The data was collected via the Twitter API.

## ⚙️ Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading & Initial Exploration**: The dataset is loaded using Pandas. Initial checks are performed to understand its structure and size.

2.  **Data Preprocessing (NLP)**: This is a crucial step to clean and prepare the text data for the model using NLP techniques.
    - **Label Conversion**: The original positive label `4` is mapped to `1` for binary classification.
    - **Text Cleaning & Stemming**: A custom function performs several key tasks:
        - Removes non-alphabetic characters (like punctuation, numbers, and symbols).
        - Converts all text to lowercase.
        - Removes common English **stopwords** (e.g., "the", "a", "is") using the NLTK library.
        - Applies **Porter Stemming** to reduce words to their root form (e.g., "running" becomes "run"). This helps the model generalize better.

3.  **Feature Extraction (NLP)**:
    - The cleaned text data is converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique. TF-IDF reflects how important a word is to a document in a collection or corpus, which is ideal for text classification tasks.

4.  **Model Training**:
    - The dataset is split into a training set (80%) and a testing set (20%).
    - A **Logistic Regression** model is trained on the TF-IDF vectors from the training data.

5.  **Model Evaluation**:
    - The model's performance is evaluated using the **accuracy score** on both the training and testing sets to check for generalization and prevent overfitting.

6.  **Model Persistence**:
    - The trained model is saved to a file (`trained_model.sav`) using `pickle`. This allows the model to be reloaded and used for predictions later without needing to be retrained.

## 🛠️ Technologies Used

* **Core Concepts:**
    * **Natural Language Processing (NLP):** Text preprocessing, feature extraction.
    * **Machine Learning:** Supervised learning, classification.

* **Python 3**

* **Libraries:**
    * **Pandas:** For data manipulation and loading.
    * **NLTK (Natural Language Toolkit):** For stopwords and stemming.
    * **Scikit-learn:** For machine learning tasks (`TfidfVectorizer`, `train_test_split`, `LogisticRegression`, `accuracy_score`).
    * **re (Regular Expressions):** For text cleaning.
    * **Pickle:** For saving and loading the trained model.

* **Jupyter Notebook / Google Colab:** For development and experimentation.
