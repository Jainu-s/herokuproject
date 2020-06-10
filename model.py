# Libraries to work in the project:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pickle
import joblib
from langdetect import detect
import urllib.request
import urllib.parse
from selenium import webdriver
from urllib.request import Request, urlopen
from flashtext.keyword import KeywordProcessor
import re
import ssl
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from inscriptis import get_text
import pickle
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score,recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer




## Data Importing/Reading:

## Importing alcohol scraped saved text files data from folder
## Change loc path
loc = '/home/user/Desktop/Allinone/al'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r') as myfile:
        data.append(myfile.read())

df = pd.DataFrame(data, columns=['Data'])
print(df.shape)
new_df = df.loc[0:45]
# new_df['Label'] = 'alcohol'
new_df.insert(1, 'Label', 'alcohol')
new_df = new_df.loc[0:45]
new_df.shape
new_df.head()

## Importing war/violence/weapon scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/wvw'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df1 = pd.DataFrame(data, columns=['Data'])
new_df1 = df1[:46]
new_df1.insert(1, 'Label', 'warvioweapon')
print(new_df1.shape)
new_df1.head()

## Importing religion scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/re'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df2 = pd.DataFrame(data, columns=['Data'])
new_df2 = df2[:46]
new_df2.insert(1, 'Label', 'religion')
print(new_df2.shape)
new_df2.head()

## Importing porn scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/po'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df3 = pd.DataFrame(data, columns=['Data'])
new_df3 = df3[:46]
new_df3.insert(1, 'Label', 'porn')
print(new_df3.shape)
new_df3.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/dr'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df4 = pd.DataFrame(data, columns=['Data'])
new_df4 = df4[:150]
new_df4.insert(1, 'Label', 'drug')
print(new_df4.shape)
new_df4.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/fin'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df5 = pd.DataFrame(data, columns=['Data'])
new_df5 = df5[:150]
new_df5.insert(1, 'Label', 'finance')
print(new_df5.shape)
new_df5.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/school'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df6 = pd.DataFrame(data, columns=['Data'])
new_df6 = df6[:150]
new_df6.insert(1, 'Label', 'education')
print(new_df6.shape)
new_df6.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/searchen'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df7 = pd.DataFrame(data, columns=['Data'])
new_df7 = df7[:150]
new_df7.insert(1, 'Label', 'Search Engine')
print(new_df7.shape)
new_df7.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/job'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df8 = pd.DataFrame(data, columns=['Data'])
new_df8 = df8[:150]
new_df8.insert(1, 'Label', 'Job/Career')
print(new_df8.shape)
new_df8.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/POLITICS'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df9 = pd.DataFrame(data, columns=['Data'])
new_df9 = df9[:150]
new_df9.insert(1, 'Label', 'Politics')
print(new_df9.shape)
new_df9.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/Hacking'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df10 = pd.DataFrame(data, columns=['Data'])
new_df10 = df10[:150]
new_df10.insert(1, 'Label', 'Hacking')
print(new_df10.shape)
new_df10.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/chat'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df11 = pd.DataFrame(data, columns=['Data'])
new_df11 = df11[:150]
new_df11.insert(1, 'Label', 'Chat')
print(new_df11.shape)
new_df11.head()

## Importing scraped saved text files data from folder
## Change loc path

loc = '/home/user/Desktop/Allinone/shopping'
os.chdir(loc)
filelist = os.listdir()
# print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))

data = []
path = loc
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
    with open(f, 'r', encoding="utf8", errors='ignore') as myfile:
        data.append(myfile.read())

df12 = pd.DataFrame(data, columns=['Data'])
new_df12 = df12[:150]
new_df12.insert(1, 'Label', 'Shopping')
print(new_df12.shape)
new_df12.head()



## appending both dataframes
data1 = new_df.append(new_df1, ignore_index=True)
data2 = data1.append(new_df2,ignore_index=True)
data3 = data2.append(new_df3,ignore_index=True)
data12 = data3.append(new_df4,ignore_index=True)
data5 = data12.append(new_df5,ignore_index=True)
data6 = data5.append(new_df6,ignore_index=True)
data7 = data6.append(new_df7,ignore_index=True)
data8 = data7.append(new_df8,ignore_index=True)
data9 = data8.append(new_df9,ignore_index=True)
data10 = data9.append(new_df10,ignore_index=True)
data11 = data10.append(new_df11,ignore_index=True)
data12 = data11.append(new_df12,ignore_index=True)




print(data12.head(),data12.tail())


## Data Preprocessing and cross checking
print(data12['Label'].unique())

data12['nlabel'] = data12['Label'].factorize()[0]
print(data12.head(),data12.tail())

category_id_df = data12[['Label', 'nlabel']].drop_duplicates().sort_values('nlabel')
print(category_id_df)

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['nlabel', 'Label']].values)

## Checking how many unique labels and their data
data12.groupby('Label').nlabel.count()

## Visualizing the categories
data12.groupby('Label').nlabel.count().plot.bar(ylim=0)


## TFIDF Vectorizer creating object and fitting the data
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(data12.Data).toarray()
labels = data12.nlabel
print(features.shape)

from sklearn.feature_selection import chi2

N = 3  # We are going to look for top 3 categories

# For each category finding words that are highly co-related
for Category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]

    # Converts indices to feature names ( in increasing order of chi-squared stat values)
    unigrams = [v for v in feature_names if len(
        v.split(' ')) == 1]  # List of single word features ( in increasing order of chi-squared stat values)
    bigrams = [v for v in feature_names if
               len(v.split(' ')) == 2]  # List for two-word features ( in increasing order of chi-squared stat values)
    print("# '{}':".format(Category))
    print("  . Most correlated unigrams:\n       . {}".format(
        '\n       . '.join(unigrams[-N:])))  # Print 3 unigrams with highest Chi squared stat
    print("  . Most correlated bigrams:\n       . {}".format(
        '\n       . '.join(bigrams[-N:])))  # Print 3 bigrams with highest Chi squared stat
    print(features_chi2)

# Importing t-distributed Stochastic Neighbor Embedding which is used for visualizing high dimensional data
from sklearn.manifold import TSNE
# Sampling a subset of our dataset because t-SNE is computationally expensive
SAMPLE_SIZE = int(len(features) * 0.3)
np.random.seed(0)
# Randomly select 30 % of samples
indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
# Array of all projected features of 30% of Randomly chosen samples
projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
my_id = 0 # Select a category_id
projected_features[(labels[indices] == my_id).values]
colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey','yellow','blue','red','black','violet','gold','teal','turquoise']


# Find points belonging to each category and plot them
for category, category_id in sorted(category_to_id.items()):
    points = projected_features[(labels[indices] == category_id).values]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",fontdict=dict(fontsize=15))
plt.legend()

# creating objects for models
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0)]

# Cross Validate with 5 different folds of 20% data
CV = 5
#Create a data frame that will store the results for all 5 trials of the 3 different models
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = [] # Initially all entries are empty

#For each Algorithm
for model in models:
  model_name = model.__class__.__name__
  # create 5 models with different 20% test sets, and store their accuracies
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  # Append all 5 accuracies into the entries list
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

# Store the entries into the results dataframe and name its columns
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)

# Mean accuracy of each algorithm
print(cv_df.groupby('model_name').accuracy.mean())
print(cv_df)

# Model fit Logistic regression with 33% of data randomly chosen for test

model = LogisticRegression(random_state=0)

#Split Data
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data12.index, test_size=0.33, random_state=0)

#Train Algorithm
model.fit(X_train, y_train)

# Make Predictions
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Print confusion matrix in test data using seaborn

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# Print confusion matrix in test data using seaborn

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Testing data
X_train, X_test, y_train, y_test = train_test_split(data12['Data'], data12['nlabel'],test_size=0.2)
print(type(X_test))
print(X_test,'X_test')
test_features = tfidf.transform(X_test.tolist())
Y_pred = model.predict(test_features)
print(Y_pred)

# Category ids are converted to Category name
Y_pred_name =[]
for cat_id in Y_pred :
    Y_pred_name.append(id_to_category[cat_id])
print(Y_pred_name)

# using pickle to save model
pickle.dump([model,tfidf,id_to_category,cat_id], open('MultiLiModel.pkl', 'wb' ))

#Sample data for testing
sample = ["India Today League PUBG Mobile Invitational 2020 Day 2 Live: Team Hydra won the Chicken Dinner in the 1st match of Day 2 that was played in Erangel. A solid show saw them taking 14 points. On the other hand, Team SouL and team Fnatic came up with disappointing display in the Day 2 opener.",
          "google in a blog post cool stated that all advertisers will now be required to verify their identities and country of origin. This was done for political ads in 2018.",
          "t's good to have beers around. In the fridge. At the office. In the cooler. Not so much in the car. But almost everywhere else. It's like a comfort blanket for adults. And there are so many different beer types, you'll never get bored. Fancy beers, cheap beers, flavorful beers, light beers—they've all got their time and place. That place mostly being in our hearts.",
         "AAP chief Arvind Kejriwal will be sworn-in as Delhi's chief minister for a third consecutive term at Ramlila Maidan on Sunday.",
         "Welcome to beer company, in the restaurant and the brewery under one roof.The idea that has long come to us has become a reality.We have reviewed all of our past experience with beer production and with the building of breweries in our country and in the world. We have developed our own technology and practices that make our beers different and exceptional.We want to show the best of our skills.Great beer includes fine cuisine, so even our chefs do not neglect anything. Welcome in our beer factory."]

model,tfidf,id_to_category,cat_id = pickle.load(open('MultiLiModel.pkl','rb'))
sample1 = tfidf.transform(sample)
print(model.predict(sample1))