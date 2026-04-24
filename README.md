# DSC 232R Group Project
Gloria Kao, Mahir Oza, Ali Karim, Michael Nodini

## Abstract
Online forums like Reddit are often interested in identifying trends and patterns in user behavior to suggest uniquely curated topics of interest or channels to collaborate and discuss. This dataset is found on Kaggle and sourced from multiple Reddit subreddits (i.e. forums of different topics), and contains Reddit submission posts ranging from July 2021 to February 2023, totaling over 130GB of data, with each month provided as its own CSV file. Since this dataset contains NSFW topics (labeled as “over 18”), our project will analyze a subset of the dataset, produced during the data cleaning section by removing inappropriate topics. Nonetheless, the expected dataset size following our cleaning pipeline will still be over 50GB, requiring a high level of computing power that cannot be done by any normal consumer machine. Thus, we need to use distributed computing to load and work with the full dataset. Such a method provides cheap efficiency and makes the large dataset scalable for our project to work in a faster environment. Since much of the dataset is text-based, our research will focus on Natural Language Processing (NLP) to conduct Sentiment Analysis by different categories of subreddit (e.g. most/least positive subreddits), and Subreddit Prediction to train a classification model to predict the most suitable subreddit from unseen Reddit posts. The expected analysis would be useful for Reddit in cases that may involve moderation of subreddits or subreddit suggestions for users who may not know where to post.

## Datasets
“Reddit Submissions July 2021 to Oct 2022” from Kaggle : https://www.kaggle.com/datasets/noahpersaud/reddit-submissions-july-2021-to-oct-2022 

“Reddit Submissions Dec 2022 to Feb 2023” from Kaggle : https://www.kaggle.com/datasets/noahpersaud/reddit-submissions-dec-2022-to-feb-2023 

## SDSC Expanse Environment Setup
**SparkSession Configuration**

```python
# Insert Code for SparkSession Configuration
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "8g") \
    .config('spark.executor.instances', 8) \
    .appName("KaggleData") \
    .getOrCreate()
```
With our raw dataset sitting at approximately 132GB, with the memory of the driver allocated at 2GB, the best option for our setup requires an executor instance of 15 where we have 16 cores with one assigned to the driver. Additonally, with 15 executors needing to compute a dataset at this size (132GB with 2GB set aside for the driver), the memory allocated for each executor would be about 8GB.

*justification (2b); note: executor instances = Total Cores - 1 & Executor Memory = (Total Memory - Driver Memory)/Executor Instances*

*screenshot of SparkUI showing active executors*

## Data Exploration Using Spark

**Number of Observations in Raw Dataset: n**

**Columns (Scales, Distributions, Categorical/Continuous Type, & Feature/Target) of Dataset:**

| Column | Description | Scale | Distribution | Categorical/Quantitative (Type) | Feature/Target|
|---|---|---|---|---|---|
| title | Provides the naming of the post made by some reddit user | string/text-based naming | any sequence of characters of any length | categorical | feature |
| post_id | Links unique identifier to each post entry made by users on site | string | distinct 6-digit code | categorical | feature |
| over_18 | - | - | - | - | - |
| subreddit | - | - | - | - | - |
| link_flair_text | - | - | - | - | - |
|self_text | - | - | - | - | - |

**Missing/Duplicate Values Within Dataset:**
This data does contain missing values that are primarily seen in features for link_flair_text and self_text. 
*Note: Dataset contains no image data - completely text based*

## Data Plots

*Spark Aggregation based visualizations*

*descriptions and insights*

## Preprocessing Plan

**Handling Missing Values:**
Since the primary feature we will be looking at to determine subreddit is the post title ('title'), any posts with a missing or duplicate title will be dropped from the usable set. Similarly, any entries missing a subreddit will also be dropped from consideration for our training, validation, and test sets since it would not be possible to predict and compare on a post missing the target variable, subreddit. Since the other features will be less important for prediction, any missing values encountered for those posts will be kept to potentially make more accurate predictions. 

**Data Imbalance:**
Since this dataset contains thousands of different subreddits, it becomes clear that some of these forums appear very few times (many only once) while other subreddits are seen much more frequently. When training our models to predict subreddits for posts, many subreddits will have multiple posts to train up on compared to other subreddits which would have few to likely no subreddits to train up on. This means that when running our prediction on a validation/test set, those subreddits that the model had multiple entries to train on are going to be easier to predict while there will be many subreddits that the model has not seen and will struggle to accurately predict leading to this imbalance within the feature set of our data.

**Data Transformations (Scaling, Encoding, Feature Engineering):**

**Spark Operations for Preprocessing:**
