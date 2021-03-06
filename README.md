# Stackoverflow.com project
***created by Yongheng Wang***


## Executive Summary
#### Project objective
The main objective of our project is to create different machine learning models for post `Tags` and `Score` based on `Body` `Title` `ViewCount` and other related objects, that is, for a user posted a question, we can predict: 1. whether the topic of this post is related to _calculus_ or not, 2. whether the score of this post is high or low, 3. we also build a KNN model with NLP to obtain IDs of other users who ask or answer similar questions given the ID of a certain user.

#### Methods
* Since the file is in XML format, which cannot be converted directly to data frame, we have to parse the xml into RDD by using a `lambda` function.
* For the `Tags` predictive model, we train a **binary classifier** on a target tag, and then predict if a post belongs to this target tag according to its `Title` and `Body`. 
* For the `Score` predictive model, we implemented 3 models: **logistic regression**, **random forest** and **XGBoosting** to predict post score based on post attributes, including the view count, answer count, post creation year and so on. 
* We use KNN model to find the nearest users who always ask or answer similar questions, and then generate the wordcould of most frequently used words.

#### Results
* AUC score of the `Tags` predictive model is 0.6264.
* AUC `Score` of logistic regression is 0.6217. AUC `Score` of random forest is 0.7084. AUC `Score` of XGBoost model is 0.7735.
* Given a certain user’s ID, our model can generate wordclouds of user most frequently used words and a list of user who share the similar context.


## Introduction
The dataset that we choose is ***Stack Exchange Data Dump***, which is an anonymized dump of all user-contributed content on the _Stack Exchange network_. Each site is formatted as a separate archive consisting of XML files zipped via 7-zip using bzip2 compression. Each site archive includes `Posts`, `Users`, `Votes`, `Comments`, `PostHistory` and `PostLinks`. We consider `Posts` as the most interesting and challenging part, so we choose to dig into it. 

Our goal is to: 

* Create a predictive model that predicts post Tags based on `Body` and `Title`, 

* Create a predictive model that predicts post score based on `View Count`, `Answer Count`, `Posty Type` etc.

On top of building predictive models, we also consider doing some user-level analysis such as :

* Finding the users who always ask similar questions with the specific user;

* Finding the users who always provide similar answers with the specific user.

In doing so, we are trying to make users feel convenient to locate other users who ask or answer similar questions.
The standard that we utilize to determine which users' questions or answers are similar to those of a target user is the suitability of ****words frequency****.


## Exploratory analysis section

* ***Score Distribution:*** 
The `Score` column has a wide span, so it's difficult to predict a precise score. To make the prediction more accurate, we create a new feature classifying posts with a score over **5** as **high**, and the rest of which as **low**.

<img src="https://my4dbucket.s3.amazonaws.com/math/score_distribution.png">

* ***View Count Mean Score:***
This table shows at different levels of view count, the average post score. It tells us the post with a higher view count tends to have a higher score. We will include this feature in our model.  

<img src="https://my4dbucket.s3.amazonaws.com/math/view-meanscore.png" width = "230">

* ***Comment Count Mean Score:***
The plot shows the relationship of comment count and average post score. It tells us the posts have slightly lower scores when the comment amount get higher in general. This is counter intuitive and we will hold this view and explore it further.

<img src="https://my4dbucket.s3.amazonaws.com/math/comment-score.png" width = "400">


## Methods section:

* **To source, ingest the data** To ingest this bulky data, we download the files from https://archive.org/download/stackexchange. Since it is in 7zipped format and cannot be unzipped directly from HDFS, we upload these files to our S3 bucket and read it through Pyspark.

* ***To clean, prepare the dataset*** As the file is in XML format and cannot be straightly converted to well-structured RDD or dataframe, we parse the raw RDD manually by using a *lambda* function. 

* ***To model the dataset, techniques used*** By doing EDA, we figure out features in relationship to our target variable. We use `groupBy`, `plotting tools` etc. To make plotting workable, we randomly sampled a proportion of the data to make sure the instance and hold this huge size data. After deciding to use a  classification model to predict post score, we pick basic `logistic regression` model, `random forest` and `XGBoost` model since they're classic and accurate. We also train a `binary classifier` for a given tag. For instance, for the tag “calculus” one classifier will be created which can predict a post that is about the calculus-related questions. Instead of training a multi-label classifier, we train this binary classifier for simplification. However, many classifiers might be created for frequent labels, say statistics, calculus, probability, linear-algebra etc. Another model we adopted was `KNN`，which aims at searching for the nearest users based on the text of the question or the answer so as to generate the wordcloud of the most frequently used words and a list of user Ids with the highest matching degree.


## Results/Conclusions Section:
* We've learned how to utilize cloud to deal with big data: using _wget_ to download, S3 bucket to store and EMR to play with the dataset.
* We import ****BinaryClassificationMetrics**** in model evaluation part, using ***area under ROC(AUC)*** to measure model performance. AUC of the `Tags` predictive model is 0.6264. 

* AUC `Score` of logistic regression is 0.6213 as shown below. The AUC `Score` of random forest is 0.70839. For XGBoost model, since it's had the best performance, we tune its parameters and get it's best AUC result 0.7736. 

* **Logistic Regression Model ROC:**
<img src="https://my4dbucket.s3.amazonaws.com/math/roc.png" width = "420">

* **Features Importance:**
As we can tell from the feature importance result, the `year` when the post is created has the greatest contribution to our model. `View count` also plays a great role in our score predictive model.

<img src="https://my4dbucket.s3.amazonaws.com/math/feature-importance.png" width = "230">

* **Word Cloud:** Given the id number of a certain user could generate the following wordclouds indicating the most frequent words mentioned and the Ids of other users who ask or answer the most frequent words mentioned and the Ids of other users who ask or answer the most questions that are similar.
![question_wordcloud](https://my4dbucket.s3.amazonaws.com/math/Question.png)
![answer_wordcloud](https://my4dbucket.s3.amazonaws.com/math/Answer.png)



## Code files
* Tags_predictive_model.ipynb
* KNN_wordcloud.ipynb
* EDA_Score_Predictive_model.ipynb

