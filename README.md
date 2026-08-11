# Pre-Publication Tweet Engagement Prediction

A machine learning project that predicts the engagement of a Tamil political tweet **before it is posted**, using multilingual BERT representations and tweet-level features.

## Overview

Predicting tweet engagement is a challenging regression problem because engagement depends on both the content of a tweet and many external factors.

This project explores whether information available **at the time of posting** can be used to estimate the eventual number of likes a tweet receives.

The final pipeline combines:

* Multilingual BERT (`bert-base-multilingual-cased`) embeddings
* Sentiment features
* Tweet text characteristics
* Mentions, hashtags and URLs
* Media information
* Tweet type
* Temporal features such as time of day and day of week
* Neural Network regression
* LightGBM regression
* Ensemble modelling and feature-importance analysis

## Pipeline

```text
Tamil Tweets
     │
     ▼
Data Cleaning
     │
     ├───────────────┐
     ▼               ▼
Tweet Text       Structured Features
     │               │
     ▼               ├── Text length
Multilingual BERT    ├── Mentions
     │               ├── Hashtags
     ▼               ├── URLs
768-dim Embedding    ├── Media
     │               ├── Tweet type
     │               ├── Sentiment
     │               └── Temporal features
     │               │
     └───────┬───────┘
             ▼
       Combined Features
             │
       ┌─────┴─────┐
       ▼           ▼
 Neural Network  LightGBM
       │           │
       └─────┬─────┘
             ▼
        Model Evaluation
```

## Dataset

The project uses a dataset of Tamil tweets containing tweet text, timestamps, language information, tweet type, media information and engagement metrics.

The target variable is:

```text
favorite_count
```

which represents the number of likes received by a tweet.

Post-publication engagement metrics are **not used as model features**.

The dataset is not included in this repository.

## Feature Engineering

Several features are extracted from each tweet.

### Text features

* Tweet text
* Text length
* Multilingual BERT representation

### Tweet structure

* Number of mentions
* Number of hashtags
* Presence of URLs
* Presence/type of media
* Tweet type, such as original tweet, reply or retweet

### Sentiment

Sentiment is obtained using:

`cardiffnlp/twitter-xlm-roberta-base-sentiment`

### Temporal features

* Hour of day
* Day of week
* Time-of-day category

## Text Representation

The project uses `bert-base-multilingual-cased` as a frozen feature extractor.

For each tweet, the final hidden-state representation corresponding to the `[CLS]` token is extracted, producing a **768-dimensional embedding**.

These embeddings are concatenated with the engineered tweet features before being passed to the downstream models.

## Models

### Neural Network

The neural network uses the combined BERT and engineered features.

Architecture:

```text
Input
  ↓
Linear(794 → 512)
  ↓
ReLU + Dropout
  ↓
Linear(512 → 256)
  ↓
ReLU + Dropout
  ↓
Output
```

### LightGBM

LightGBM is trained on the same combined feature representation.

Hyperparameters were tuned using cross-validation.

The best configuration in the final experiment was:

```text
learning_rate = 0.05
n_estimators = 200
num_leaves = 40
```

## Results

The final random train/test split produced the following results:

| Model                  |    Test R² |
| ---------------------- | ---------: |
| Neural Network         | **0.3181** |
| LightGBM               | **0.4157** |
| NN + LightGBM Ensemble | **0.4146** |

### Best Model

**LightGBM — R² = 0.4157**

LightGBM slightly outperformed the ensemble, so it is treated as the final model for this version of the project.

The ensemble was retained as an experiment to evaluate whether combining different model architectures could improve performance.

## Feature Importance

LightGBM feature importance was used to understand which parts of the representation contribute most to the model.

The engineered features with the highest importance included:

1. `text_length`
2. `mention_count`
3. `has_media`
4. `type_Retweet`
5. `media_is_photo`
6. `type_Reply`
7. `sentiment_Positive`
8. `is_Evening`
9. `media_is_video`

The feature importance was also aggregated into two broad groups:

| Feature Group       | Total Importance |
| ------------------- | ---------------: |
| mBERT Embeddings    |         **7413** |
| Engineered Features |          **387** |

This indicates that the model relies predominantly on the semantic representation obtained from mBERT, while the engineered features provide additional predictive signal.

## Evaluation

The primary evaluation metric is **R² (coefficient of determination)**.

The final experiment uses a random train/test split, with preprocessing fitted on the training data only.

A chronological evaluation was also investigated during experimentation. The chronological split showed substantial temporal distribution shift in the dataset, particularly toward December 2024, resulting in significantly lower predictive performance. This version therefore retains the random-split experiment as the primary baseline rather than presenting the chronological experiment as the final model.

## Limitations

Tweet engagement is inherently difficult to predict because many important factors are unavailable before publication, including:

* Audience size and follower dynamics
* Current events and external context
* Platform recommendation algorithms
* Post-publication interactions
* Virality and other unpredictable effects

Additionally, the dataset represents a specific collection of Tamil tweets, so the model should not automatically be assumed to generalize to other languages, platforms or populations.

## Future Improvements

Potential directions for improving the current baseline include:

* Fine-tuning multilingual BERT instead of using frozen embeddings
* Exploring alternative multilingual language models
* Experimenting with different pooling strategies for BERT representations
* More systematic feature ablation
* Improved regression objectives and target transformations
* More detailed error analysis
* More robust temporal evaluation
* Comparing additional tree-based and neural architectures

## Repository Structure

```text
tweet-engagement-prediction/
│
├── README.md
├── tweet_engagement_prediction.ipynb
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Running the Notebook

The notebook was developed with GPU acceleration and is suitable for running in a Kaggle environment.

The dataset path used during development is configured in the notebook:

```text
/kaggle/input/datasets/harishgs145/tamil-tweets/merged_file.csv
```

Update `DATA_PATH` if the dataset is stored elsewhere.

Because BERT embeddings and sentiment analysis are computationally expensive, a GPU is recommended.

## Key Takeaway

The project demonstrates an end-to-end approach to **pre-publication tweet engagement prediction**, combining transformer-based language representations with structured feature engineering and gradient-boosted regression.

The current baseline achieves an **R² of 0.4157 using LightGBM**, providing a foundation for further experimentation with transformer representations and model architecture.
