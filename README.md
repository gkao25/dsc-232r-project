# DSC 232R Group Project
Gloria Kao, Mahir Oza, Ali Karim, Michael Nodini

## Abstract
Online forums like Reddit are often interested in identifying trends and patterns in user behavior to suggest uniquely curated topics of interest or channels to collaborate and discuss. This dataset is found on Kaggle and sourced from multiple Reddit subreddits (i.e. forums of different topics), and contains Reddit submission posts ranging from July 2021 to February 2023, totaling over 130GB of data, with each month provided as its own CSV file. Since this dataset contains NSFW topics (labeled as “over 18”), our project will analyze a subset of the dataset, produced during the data cleaning section by removing inappropriate topics. Nonetheless, the expected dataset size following our cleaning pipeline will still be over 50GB, requiring a high level of computing power that cannot be done by any normal consumer machine. Thus, we need to use distributed computing to load and work with the full dataset. Such a method provides cheap efficiency and makes the large dataset scalable for our project to work in a faster environment. Since much of the dataset is text-based, our research will focus on Natural Language Processing (NLP) to conduct Sentiment Analysis by different categories of subreddit (e.g. most/least positive subreddits), and Subreddit Prediction to train a classification model to predict the most suitable subreddit from unseen Reddit posts. The expected analysis would be useful for Reddit in cases that may involve moderation of subreddits or subreddit suggestions for users who may not know where to post.

## Datasets
“Reddit Submissions July 2021 to Oct 2022” from Kaggle : https://www.kaggle.com/datasets/noahpersaud/reddit-submissions-july-2021-to-oct-2022 

“Reddit Submissions Dec 2022 to Feb 2023” from Kaggle : https://www.kaggle.com/datasets/noahpersaud/reddit-submissions-dec-2022-to-feb-2023 

## SDSC Expanse Environment Setup

### SparkSession Configuration

```python
# Insert Code for SparkSession Configuration
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "10g") \
    .config('spark.executor.instances', 15) \
    .appName("KaggleData") \
    .getOrCreate()
```
With our raw dataset sitting at approximately 132GB, with the memory of the driver allocated at 2GB, the best option for our setup requires an executor instance of 15 where we have 16 cores with one assigned to the driver. Additonally, with 15 executors needing to compute a dataset at this size (132GB with 2GB set aside for the driver), the memory allocated for each executor would be about 10GB (130GB/15 executors).

- Executor instances = Total Cores - 1 = 15
- Executor Memory = (Total Memory - Driver Memory) / Executor Instances = (132-2) / 15 = 8.67

### Screenshot of SparkUI Showing Active Executors:
<img width="715" height="107" alt="image" src="https://github.com/user-attachments/assets/2dd277dc-4817-4459-a5f8-da4be6c83dc7" />


## Data Exploration Using Spark

**Number of Observations in Raw Dataset: 654,221,435**

### Columns (Scales, Distributions, Categorical/Continuous Type, & Feature/Target) of Dataset:

| Column | Description | Scale | Distribution | Categorical/Quantitative (Type) | Feature/Target|
|---|---|---|---|---|---|
| title | Provides the naming of the post made by some reddit user | string/text-based naming | any sequence of characters of any length | categorical | feature |
| post_id | Links unique identifier to each post entry made by users on site | string | distinct 6-digit code | categorical | feature |
| over_18 | Boolean identifier to flag if a post/subreddit is NSFW (TRUE) or SFW and appropriate (FALSE) | Boolean | True or False | categorical (binary) | feature |
| subreddit | Title descriptor for forum on which users can communicate, hold discussions, and interact | string | any sequency of characters of any length | categorical | target |
| link_flair_text | Tags on post to help identify specific features contained within the post | string | any sequence of characters typically of a relatively short length | categorical | feature |
|self_text | Primary body that makes up the forum post | string | any sequence of characters of any length | categorical | feature |

### Missing/Duplicate Values Within Dataset:
This data does contain missing values that are primarily seen in features for link_flair_text and self_text. Additionally, self_text contains text like '[deleted]' or '[removed]' which we will consider as missing data. We do see duplicate data for the subreddits, but since this is both expected and desired, where we would expect the dataset to include multiple posts for likely the same forum, we will not be dropping or handling any duplicates in the subreddit target column. The only feature to worry about having duplicates would be the post_id since this is a unique identifier for each post made. If there are any duplicte post id's our plan to handle it would be to test and see if the each duplicate instance is all the same for all 6 columns. If it is, then we will keep only one instance and drop the rest; if it is not, we will drop every instance of the duplicate post_id.

### Null and empty values count:
| Column           | Missing Count |
|------------------|--------------|
| title            | 336          |
| post_id          | 17933        |
| over_18          | 20405        |
| subreddit        | 21505        |
| link_flair_text  | 425,449,504  |
| self_text        | 345,790,643  |

*Note: Dataset contains no image data - completely text based*

## Data Plots

*Spark Aggregation based visualizations*

*descriptions and insights*

## Preprocessing Plan

### Handling Missing Values:
The primary feature we will be looking at to determine subreddit is the post title ('title') and the post itself ('self_text') so any posts with a missing or duplicate title or post text will be dropped from the usable set. These features are vital to calculating sentiment score's in predicting the subreddit, so making predictions with missing data in these columns would cause the model to be unable to make subreddit predictions. Similarly, any entries missing a subreddit will also be dropped from consideration for our training, validation, and test sets since it would not be possible to predict and compare on a post missing the target variable, subreddit. Since the other features will be less important for prediction, any missing values encountered for those posts will be kept to potentially make more accurate predictions. 

### Data Imbalance:
Since this dataset contains thousands of different subreddits, it becomes clear that some of these forums appear very few times (many only once) while other subreddits are seen much more frequently. When training our models to predict subreddits for posts, many subreddits will have multiple posts to train up on compared to other subreddits which would have few to likely no subreddits to train up on. This means that when running our prediction on a validation/test set, those subreddits that the model had multiple entries to train on are going to be easier to predict while there will be many subreddits that the model has not seen and will struggle to accurately predict leading to this imbalance within the feature set of our data. To ensure fairness to different subreddits, we will be dropping any subreddits that have fewer than 10 occurrences within the overall dataset so that we can expect our model to be able to train up on the subreddits it would expect to see from the validation/test sets.

### Data Transformations (Scaling, Encoding, Feature Engineering):
Unfortunately, a good portion of this dataset contains NSFW content, highlighted by the over_18 feature, the first step in cleaning and transforming our data into a format we feel comfortable working with for this project is to drop any posts from our dataset that have a TRUE boolean value for this column. This should not only drop a good portion of rows, many of which contain few, distinct, unique subreddits, but will also make our dataset much more scalable as we move forward with our modeling plan. While we acknowledge these subreddits are important to deterministic aspects to reddits business model, from an academic and comfrtability standpoint this is the most appropriate path forward for our group. We will also be leveraging transformation encodings such as TF-IDF, One-Hot Encoding (OHE), or Word2Vec methods. This will be necessary for the NLP techniques we plan to implement in order to process the thousands of text-based post features we are utilizing so that our model can predict subreddits accurately. We will apply sentiment analysis to each of the self_text rows, to then group all sentiment scores (on a scale from -1 for most negative to +1 for most positive) according to sub-reddit. This will provide us a way to see which 5 subreddits have the most positive or negative sentiment.

### Spark Operations for Preprocessing:

```python
df.count() # Number of Entries: 654221435
df.select("subreddit").distinct().count() # Unique Subreddits: 6857314

# get subset of entries that are under 18 
df = df.where("over_18 = false")
```
