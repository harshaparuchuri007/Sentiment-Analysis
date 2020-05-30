# -*- coding: utf-8 -*-
"""

@author: galahad
"""

#importing necessary libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
import wordcloud 
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%matplotlib inline

train  = pd.read_csv('Train_Democrats_Dummy.csv')       #Taking input the Train data
test = pd.read_csv('Test_Democrats_Dummy.csv')          #Taking input the Test data

combi = train.append(test, ignore_index=True)       #Combining the Train and Test dataset

def remove_pattern(input_txt, pattern):             #removing the pattern ‘@user’ from all the tweets
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt
#create a new column tidy_tweet, it will contain the cleaned and processed tweets.
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['sentence'], "@[\w]*")  # pick any word starting with ‘@’
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")         # remove special characters, numbers, punctuations
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) # Removing Short Words. Removes all the words having length 3 or less.

#Tokenizing the tweets
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

#Stemming the tweets- removing the suffixes from the words.
from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()

#Integrating the tokens
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweet'] = tokenized_tweet

#Visualization of data into Maps and Graphs using the wordcloud plot
all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

# Display the sentiment Map for the All the frequent Words found in the data for tweets
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Display the sentiment Map for the Positive Words found in the data for tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['Polarity'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Display the sentiment Map for the Negative Words found in the data for tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['Polarity'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

#extracting hashtags from positive tweets
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['Polarity'] == 0])

#extracting hashtags from negative tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['Polarity'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

# check the hashtags in the Positive tweets
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})

# selecting top 10 most frequent hashtags and displaying the Histogram.
d = d.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

# Display the Histogram words for Negative words
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#Creating the bag of words feature
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])

#Building model using Bag-of-Words features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
bow1=bow.toarray()
train_bow = bow1[:36000,:]
test_bow = bow1[36000:,:]
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['Polarity'], random_state=42, test_size=0.3)

from sklearn import preprocessing
lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int))  #F1 score for Bag of Words

#this model predicts for the test data
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['Polarity'] = test_pred_int
submission = test[['id','Polarity']]
submission.to_csv('Predict_Democrats3.csv', index=False) # writing data to a CSV file

#Building model using TF-IDF features
train_tfidf = tfidf[:36000,:]  # Giving the Limit of rows for which the Data is to be trained
test_tfidf = tfidf[36000:,:]   #Giving the Limit of rows for which the Data is to be tested
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
lreg.fit(xtrain_tfidf, ytrain)
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int))   #f1 score for TD-IDF

#Displaying the confusion Matrix for the given data
from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(yvalid, prediction_int)
print(classification_report(yvalid,prediction_int))
print(confusion_matrix(yvalid,prediction_int))

#Plotting the ROC Curve of the given Data
import matplotlib.pyplot as plt
#Accuracy()
lreg.score(xtrain_bow,ytrain)

# Displaying the ROC curve
from sklearn.metrics import roc_curve
Y_pred_prob=lreg.predict_proba(xvalid_bow)[:,1]
print("The mean of the Polarity is",np.mean(Y_pred_prob))
fpr,tpr,thresholds=roc_curve(yvalid,Y_pred_prob)
print("\nBelow is the ROC Curve for Republicans:")
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel('False postive rate')
plt.ylabel('True Postive rate')
plt.show()

#Determining The Accuracy of the Algorithm
accuracy=lreg.score(xvalid_bow, yvalid)*100
print ("The Accuracy of the Algorithm is: ",accuracy,"percent")

#ROC AUC score
from sklearn.metrics import roc_auc_score
print("The ROC Accuracy Score is ",roc_auc_score(yvalid,Y_pred_prob))

#ROC AUC  CV
from sklearn.model_selection import cross_val_score
cv_scores=cross_val_score(lreg,train_bow,train["Polarity"],cv=5,scoring='roc_auc')
print ("The CV Score is: ",cv_scores.mean())

#hyperparameter tuning
lreg.get_params()
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
c_space=np.logspace(-5,8,15)

#classifier.get_params()
param_grid={'C':c_space,'penalty':['l1','l2']}
logistic_cv=GridSearchCV(lreg,param_grid,cv=5)
logistic_cv.fit(xtrain_bow,ytrain)
print("The Best Parameters Used should be: ",logistic_cv.best_params_)
logistic_cv.best_score_


