# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor
#%%
review_sentences = pd.read_pickle('review_sentences.pkl')

#LABELS
price = 1
size = 2
flavour = 3
ingredients = 4
shipping = 5
damage = 6
irrelevant = 7
functionality = 8
presentation = 9

pricelist = r"(expensive|cheap|price|money|dollar|pricey|pricy|value|priced|(\bover priced)|(\brip off)|cheapest)"
sizelist =  r"(big|size|small|portion|tiny|huge|amount)"
flavourlist =  r"(edible|pastry|tender|crunch|taste|eating|butter|sucker|tasty|yuk|delicious|yummy|yum|puke|sweet|tast|eat|eating|flavour|flavor|punch|food|smell|wet|vodka|wine|beer|breakfast|meal|lunch|dinner|light|fragrance|treat|treats|moist|coffee|refreshing|drink|tastes|smells|chocolate|snack|creamy|rich|candy)"
ingredientlist = r"(ingredient|healthy|quality|protein|carbs|vegetable|meat|beef|chicken|chunks|(\bfilled with)|spinach|veg|fruit|diet|fresh|garbage|bin|raw|sugar|chemical|burn)"
shippinglist = r"(shipping|delivery|came|quickly|delivered|shipped)"
packaginglist = r"(burst|damaged|damage|opened|openned|tear|packaging|seal|sealing)"
functionalitylist = r"(construct|cleaning|works|perform|performance|help|helped|use|easy|works|experience|device|functionality|setup|connection|skin|face|(\bused this product)|conditioning|protection|sound|display)"
presentationlist = r"(looks|visual|appear|presented|neat|laminate|glow|sparkly|look|colour|color|boxy)"

#%%
review_sentences['label'] = [list() for x in range(len(review_sentences.index))]
#%%
import re
review_sentences.label.values.tolist()
for row in range(0, len(review_sentences)):
  if re.search(pricelist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,6].append('price')

for row in range(0, len(review_sentences)):
  if re.search(sizelist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,6].append('size')

for row in range(0, len(review_sentences)):
  if re.search(flavourlist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,6].append('flavour')
    
for row in range(0, len(review_sentences)):
  if re.search(ingredientlist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,6].append('ingredients') 

for row in range(0, len(review_sentences)):
  if re.search(shippinglist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,6].append('shipping') 
    
for row in range(0, len(review_sentences)):
  if re.search(packaginglist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,6].append('packaging') 
    
for row in range(0, len(review_sentences)):
  if re.search(functionalitylist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,6].append('functionality') 

for row in range(0, len(review_sentences)):
  if re.search(presentationlist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,6].append('presentation')     


#%%
review_random_set['label_pred'] = [list() for x in range(len(review_random_set.index))]

import re
review_random_set.label_pred.values.tolist()
for row in range(0, len(review_random_set)):
  if re.search(pricelist, review_random_set.iloc[row,5]):
    print(review_random_set.iloc[row,5])
    review_random_set.iloc[row,17].append('price')

for row in range(0, len(review_random_set)):
  if re.search(sizelist, review_random_set.iloc[row,5]):
    print(review_random_set.iloc[row,5])
    review_random_set.iloc[row,17].append('size')

for row in range(0, len(review_random_set)):
  if re.search(flavourlist, review_random_set.iloc[row,5]):
    print(review_random_set.iloc[row,5])
    review_random_set.iloc[row,17].append('flavour')
    
for row in range(0, len(review_random_set)):
  if re.search(ingredientlist, review_random_set.iloc[row,5]):
    print(review_random_set.iloc[row,5])
    review_random_set.iloc[row,17].append('ingredients') 

for row in range(0, len(review_random_set)):
  if re.search(shippinglist, review_random_set.iloc[row,5]):
    print(review_random_set.iloc[row,5])
    review_random_set.iloc[row,17].append('shipping') 
    
for row in range(0, len(review_random_set)):
  if re.search(packaginglist, review_random_set.iloc[row,5]):
    print(review_random_set.iloc[row,5])
    review_random_set.iloc[row,17].append('packaging') 
    
for row in range(0, len(review_random_set)):
  if re.search(functionalitylist, review_random_set.iloc[row,5]):
    print(review_random_set.iloc[row,5])
    review_random_set.iloc[row,17].append('functionality') 

for row in range(0, len(review_random_set)):
  if re.search(presentationlist, review_random_set.iloc[row,5]):
    print(review_random_set.iloc[row,5])
    review_random_set.iloc[row,17].append('presentation') 
    
review_random_set.label_pred =review_random_set.label_pred.astype(str)
#binarising the labels (one hot)
review_random_set['flavour1'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('flavour', review_random_set.iloc[row,17]):
        print('found')
        review_random_set.iloc[row, 18] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 18] = 0

review_random_set['price1'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('price', review_random_set.iloc[row,17]):
        print('found')
        review_random_set.iloc[row, 19] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 19] = 0
        
review_random_set['ingredients1'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('ingredients', review_random_set.iloc[row,17]):
        print('found')
        review_random_set.iloc[row, 20] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 20] = 0
        
review_random_set['size1'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('size', review_random_set.iloc[row,17]):
        print('found')
        review_random_set.iloc[row, 21] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 21] = 0
        
review_random_set['shipping1'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('shipping', review_random_set.iloc[row,17]):
        print('found')
        review_random_set.iloc[row, 22] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 22] = 0
        
review_random_set['functionality1'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('functionality', review_random_set.iloc[row,17]):
        print('found')
        review_random_set.iloc[row, 23] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 23] = 0
        
review_random_set['packaging1'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('packaging', review_random_set.iloc[row,17]):
        print('found')
        review_random_set.iloc[row, 24] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 24] = 0
        
review_random_set['presentation1'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('presentation', review_random_set.iloc[row,17]):
        print('found')
        review_random_set.iloc[row, 25] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 25] = 0
    
# performance
report1 = classification_report(review_random_set.iloc[:, 9:17], review_random_set.iloc[:, 18:], target_names = categories)
    

# %%
#by retailer

review_sentences.label.value_counts()
### visualisation
review_sentences.iloc[33,6].remove('functionality')
oc = review_sentences[review_sentences['retailer'] == 'John-Lewis'].sample(50)
#drop amazon pantry de
### 

review_sentences['productNO'].value_counts().head(10)

review_random_set = review_sentences.sample(1000)
#review_random_set.to_csv('random_set.csv')


products = pd.read_csv('products.csv')
#creating dict with product key and category
product_dict = dict(zip(products.product_id, products.category))
product_dict[review_random_set.iloc[900,2]]
# %%brand
product_dict_brand = dict(zip(products.product_id, products.brand))

review_random_set['brand'] = np.nan
for row in range(0, len(review_random_set)):
        try:
            review_random_set.iloc[row, 17] = product_dict_brand[review_random_set.iloc[row,2]]
            print(review_random_set.iloc[row,2])
        except KeyError:
            print("Oops!  That was no valid number.  Try again...")
            
           
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,6.27)})
sns.set(style="darkgrid")
ax = sns.countplot(x = review_random_set.brand,
                   data = review_random_set.head(10), 
                   order = review_random_set['brand'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)                
            
            
            
            
            
            
            
            
            
            
            
            
            
# %%
####
review_random_set = pd.read_csv('random_set.csv')
review_random_set = review_random_set.drop(columns = ['Unnamed: 0'])
#adding the category using the dictionary
review_random_set['category'] = np.nan
for row in range(0, len(review_random_set)):
        try:
            review_random_set.iloc[row, 7] = product_dict[review_random_set.iloc[row,2]]
            print(review_random_set.iloc[row,2])
        except KeyError:
            print("Oops!  That was no valid number.  Try again...")

#review_random_set.to_csv('random_set.csv')

######checking flavour
print(review_random_set.iloc[22,3])
#creating the input to the classifier
ex = review_random_set.iloc[22,4] + ' '+review_random_set.iloc[22,5] + ' '+review_random_set.iloc[22,7] 
#binarising the labels (one hot)
review_random_set['flavour'] = np.nan

###treating nans in subject and category
review_random_set.fillna('', inplace=True)
review_random_set['combined'] = ''
for row in range(0, len(review_random_set)):
    review_random_set.iloc[row,8] = review_random_set.iloc[row,4] + ' '+review_random_set.iloc[row,5] + ' '+review_random_set.iloc[row,7]




#binarising the labels (one hot)
review_random_set['flavour'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('flavour', review_random_set.iloc[row,6]):
        print('found')
        review_random_set.iloc[row, 9] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 9] = 0

review_random_set['price'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('price', review_random_set.iloc[row,6]):
        print('found')
        review_random_set.iloc[row, 10] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 10] = 0
        
review_random_set['ingredients'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('ingredients', review_random_set.iloc[row,6]):
        print('found')
        review_random_set.iloc[row, 11] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 11] = 0
        
review_random_set['size'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('size', review_random_set.iloc[row,6]):
        print('found')
        review_random_set.iloc[row, 12] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 12] = 0
        
review_random_set['shipping'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('shipping', review_random_set.iloc[row,6]):
        print('found')
        review_random_set.iloc[row, 13] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 13] = 0
        
review_random_set['functionality'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('functionality', review_random_set.iloc[row,6]):
        print('found')
        review_random_set.iloc[row, 14] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 14] = 0
        
review_random_set['packaging'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('packaging', review_random_set.iloc[row,6]):
        print('found')
        review_random_set.iloc[row, 15] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 15] = 0
        
review_random_set['presentation'] = np.nan
for row in range(0, len(review_random_set)):
    if re.search('presentation', review_random_set.iloc[row,6]):
        print('found')
        review_random_set.iloc[row, 16] = 1
    else:
        print('nothing')
        review_random_set.iloc[row, 16] = 0
        
        
######       
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,6.27)})
sns.set(style="darkgrid")
ax = sns.countplot(x = review_random_set.category,
                   data = review_random_set.head(10), 
                   order = review_random_set['category'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)     
        
        
        
        
        
        
######
        

df = review_random_set.iloc[:, 9:17]
counts = []
categories = list(df.columns.values)
for i in categories:
    counts.append((i, df[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number'])
    
df_stats.plot(x='category', y='number', kind='bar', legend=False, grid=True, figsize=(8, 5))
plt.title("Number of instances per category - Random Dataset")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Category',  fontsize=12)  
plt.xticks(rotation=45)
plt.show()



####### number of words
lens = review_random_set.combined.str.len()
plt.hist(lens, bins = np.arange(0,500,10))      
plt.title('Distribution of review text length')  
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of Words',  fontsize=12)  
plt.show()     
#######
#review_random_set.to_csv('binary_reviews.csv')   
review_random_set = pd.read_csv('random_complete.csv')
    
####grouping
review_random_set.productNO.value_counts()
df_grouped = review_random_set.groupby(['productNO'])['size'].sum()
df_grouped = review_random_set.groupby(['reviewid'])['combined'].sum()

for item in df_grouped[1]:
    df_grouped[1] = df_grouped[1][item].extend()

print(df_grouped[1][1])        
######        
        
#review_random_set.to_csv('random_complete.csv')       
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 250
max_words = 20000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(review_random_set.combined)
sequences = tokenizer.texts_to_sequences(review_random_set.combined)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen = maxlen)
print(data)
#####
review_random_set['flavour'].value_counts()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(data[:300], review_random_set.iloc[:300, 8], test_size=0.25, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

#random forrest

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, classification_report

clf = RandomForestClassifier(n_estimators = 200, max_depth=100, random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred, average='micro')


############# pre processing
dfs = review_random_set.combined
import nltk 
nltk.download('punkt')
from nltk.tokenize import word_tokenize
def tokenisation(text):
    return word_tokenize(text)

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

dfs = dfs.apply(tokenisation)

def stem(token):
    return [stemmer.stem(i) for i in token]
dfs = dfs.apply(stem)

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemma = WordNetLemmatizer()

def lem(token):
    return [lemma.lemmatize(word = w, pos = 'v') for w in token]
dfs = dfs.apply(lem)

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def stop(token):
    return [item for item in token if item not in stop_words]

dfs = dfs.apply(stop)
#back into string
def st(lis):
    return ' '.join(lis)

dfs = dfs.apply(st)

########checking label values
xg = xgboost.XGBClassifier()
review_random_set.label.value_counts()
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators = [('lr', LogisticRegression()), 
                                        ('nb', MultinomialNB(alpha = 1e-1),
                                         ('b', xg)
                                         )],voting = 'soft')







######multilabel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

###
review_random_set['flavour'] = review_random_set['flavour'].apply(str)
review_random_set['packaging'] = review_random_set['packaging'].apply(str)
review_random_set['price'] = review_random_set['price'].apply(str)
review_random_set['size'] = review_random_set['size'].apply(str)
review_random_set['presentation'] = review_random_set['presentation'].apply(str)
review_random_set['functionality'] = review_random_set['functionality'].apply(str)
review_random_set['ingredients'] = review_random_set['ingredients'].apply(str)
review_random_set['shipping'] = review_random_set['shipping'].apply(str)
###
review_random_set = review_random_set.drop(columns = ['Unnamed: 0'])

# Convert the multi-labels into arrays
mlb = MultiLabelBinarizer()
y = review_random_set.iloc[:,9:17]#mlb.fit_transform(review_random_set.label)
X = review_random_set.combined
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import LabelPowerset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# LabelPowerset allows for multi-label classification
# Build a pipeline for multinomial naive bayes classification
text_clf = Pipeline([('vect', CountVectorizer(stop_words = "english",
                                              lowercase = True, 
                                              ngram_range=(1, 1))),
                     
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     
                     ('clf', LabelPowerset(svm.LinearSVC())), ]  )

voting_clf = Pipeline([('vect', CountVectorizer(stop_words = "english",
                                              lowercase = True, ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', ClassifierChain(svm.LinearSVC())),])

text_clf = text_clf.fit(X_train, y_train)

voting_clf = voting_clf.fit(X_train, y_train)

sc= text_clf.fit(X_train, y_train).score(X_test, y_test)
predicted = text_clf.predict(X_test)
predicted = voting_clf.predict(X_test)



##
predicted_matrix = predicted.todense()
# Calculate accuracy
np.mean(predicted == y_test)

recall_score(y_test, predicted,
                      average='weighted')
precision_score(y_test, predicted,
                      average='weighted')

f1_score(y_test, predicted,
                      average='weighted')

report = classification_report(y_test, predicted, target_names = categories)

print("Accuracy = ",accuracy_score(y_test, predicted))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(text_clf, X_train, y_train, cv=5, scoring='f1_micro')

from sklearn.metrics import f1_score
f1_score(y_test, predicted, average='micro')
from sklearn.metrics import hamming_loss,confusion_matrix

print("Hamming_loss:", hamming_loss(y_test, predicted))

from sklearn.metrics import classification_report

print(classification_report(y_test, predicted))

from sklearn.metrics import multilabel_confusion_matrix
xm1 = multilabel_confusion_matrix(review_random_set.iloc[:, 9:17],
                                  review_random_set.iloc[:, 18:])

cm = multilabel_confusion_matrix(y_test, predicted)
########## curve precision recall
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, predicted)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(text_clf, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


# %% space for statistical testing
#5 x 2 cross validation

from mlxtend.evaluate import paired_ttest_5x2cv


t, p = paired_ttest_5x2cv(estimator1=text_clf,
                          estimator2=voting_clf,
                          X=X, y=y, 
                          random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

#bootstrap

# configure bootstrap
n_iterations = 500
n_size = int(len(review_random_set) * 0.50)


from sklearn.utils import resample
# run bootstrap
stats = list()
statsf1 = list()
for i in range(n_iterations):
# prepare train and test sets
    train = resample(review_random_set.values, n_samples=n_size)
    test = pd.DataFrame([x for x in review_random_set.values if x.tolist()
                  not in train.tolist()])
    
    train = pd.DataFrame(train)
	# fit model
    model = text_clf
   
    model.fit(train.iloc[:,8], train.iloc[:,9:17].astype(float))
	# evaluate model
    predictions = model.predict(test.iloc[:,8])
    score = accuracy_score(test.iloc[:,9:17], predictions)
    scoref1 = f1_score(test.iloc[:,9:17], predictions, average = 'weighted')
    print(score)
    stats.append(score)
    statsf1.append(scoref1)

# plot scores
plt.hist(stats)
plt.show()
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Accuracy',  fontsize=12)  
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, 
                                                      lower*100, upper*100))   
    

# plot scores
plt.hist(statsf1)
plt.show()
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Accuracy (b)/ F1 Score (o)',  fontsize=12)  
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(statsf1, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(statsf1, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, 
                                                      lower*100, upper*100))  

# %%
##### one vs rest bu t all together
clf1vr = Pipeline([('vect', CountVectorizer(stop_words = "english",
                                              lowercase = True, ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', OneVsRestClassifier(MultinomialNB(alpha=1e-1))),])

clf1vr.fit(X_train, y_train)

predictions1vr = clf1vr.predict(X_test)


report1vr = classification_report(y_test, predictions1vr, target_names = categories)



#####classifier chain
# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
# initialize classifier chains multi-label classifier
clf = Pipeline([('vect', CountVectorizer(stop_words = "english",ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', ClassifierChain(LogisticRegression())),])
# Training logistic regression model on train data
clf.fit(X_train, y_train)
# predict
predictions = clf.predict(X_test)
# accuracy
np.mean(predictions == y_test)

from sklearn.metrics import accuracy_score
print("Accuracy = ",accuracy_score(y_test, predictions))
f1_score(y_test, predictions, average='weighted')
print("Hamming_loss:", hamming_loss(y_test, predictions))
reportcc = classification_report(y_test, predictions, target_names = categories)


###### binary relevance

# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = Pipeline([('vect', CountVectorizer(stop_words = "english",ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', BinaryRelevance(MultinomialNB(alpha=1e-1))),])
classifier.fit(X_train, y_train)
# predict
predictions = classifier.predict(X_test)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
reportbr = classification_report(y_test, predictions, target_names = categories)

# %% smote
## need a keras vectoriser 


from imblearn.over_sampling import SMOTE

pipe = Pipeline([('vect', CountVectorizer(stop_words = "english",ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                    ])



oversample = SMOTE()
smt = SMOTE(random_state=0)
X_train_SMOTE, y_train_SMOTE = smt.fit_sample(X_train, y_train)

# %%


##### new per category if it does not work go to the next block
from sklearn.model_selection import train_test_split
train, test = train_test_split(review_random_set, 
                               random_state=0, test_size=0.3)


x_train = train.combined
y_train = train.iloc[:,9:17]
x_test = test.combined
y_test = test.iloc[:,9:17]

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,f1_score, precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import xgboost
from math import trunc
from sklearn.neighbors import KNeighborsClassifier

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([('vect', CountVectorizer(stop_words = "english",
                                                     ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                ('clf', Label_Powerset(svm.LinearSVC(), n_jobs=-1)),
            ])

LogReg_pipeline2 = Pipeline([('vect', CountVectorizer(stop_words = "english",
                                                     ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                ('clf',  OneVsRestClassifier(KNeighborsClassifier(n_neighbors =3 ),n_jobs = -1)),
            ])

report_matrix = pd.DataFrame(columns = ['acc', 'recall', 'precision', 'f1'])
c_mat = pd.DataFrame(columns = ['matrix'])
#c_3d = np.zeros((0, 0, 0))


c = 0
fig, axs = plt.subplots(1, 8, sharey=True, tight_layout=True)

for category in categories:
    '''
    print('**Processing {} comments...**'.format(category))
    
    # Training logistic regression model on train data
    LogReg_pipeline2.fit(x_train, train[category])
    
    # calculating test accuracy
    prediction = LogReg_pipeline2.predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    #print(test[category])
    #print(prediction.reshape(-1,1))
    print('Test recall is {}'.format(recall_score(test[category], prediction,
                                                  average='binary')))
    print('Test f1 is {}'.format(f1_score(test[category], prediction,
                                          average='binary')))
    print('Test precision is {}'.format(precision_score(test[category], 
                                                        prediction, average='binary')))
    '''
    ###########statistical testing
    stats = list()
    statsf1 = list()
    n_iterations = 500
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(review_random_set.values, n_samples=n_size)
        test = pd.DataFrame([x for x in review_random_set.values if x.tolist()
                      not in train.tolist()])
        
        train = pd.DataFrame(train)
    	# fit model
        model = LogReg_pipeline
        
        model.fit(train.iloc[:,8], train.iloc[:, 9 + c].astype(float))
    	# evaluate model
        predictions = model.predict(test.iloc[:,8])
        score = accuracy_score(test.iloc[:, 9 + c], predictions)
        scoref1 = f1_score(test.iloc[:, 9 + c], predictions, average = 'binary')
        #print(score)
        stats.append(score)
        statsf1.append(scoref1)
    
    #######
    # plot scores
    axs[c].hist(statsf1)
    #axs[c].show()
    #axs[c].ylabel('# of Occurrences', fontsize=12)
    #axs[c].xlabel('Accuracy',  fontsize=12)  
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(statsf1, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(statsf1, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, 
                                                          lower*100, upper*100))   
    #print(confusion_matrix(test[category], prediction))
    '''
    print("\n")
    new_row = {'acc':truncate(accuracy_score(test[category], prediction),2),
               'recall':truncate(recall_score(test[category], prediction,
                                                  average='binary'),2),
               'precision':truncate(precision_score(test[category], 
                                                        prediction, 
                                                        average='binary'),2),
               'f1':truncate(f1_score(test[category], prediction,
                                          average='binary'),2)}
    
    matrix_row = {'matrix':confusion_matrix(test[category], prediction)}
    c_mat_row = confusion_matrix(test[category], prediction)
    #c_3d = np.append(c_3d, np.atleast_3d(c_mat_row), axis = 2)
    print(c_mat_row)
    report_matrix = report_matrix.append(new_row, ignore_index = True)
    c_mat = c_mat.append(matrix_row, ignore_index= True)'''
    c +=1
    print(c)
    #per category this works

from sklearn.model_selection import train_test_split
train, test = train_test_split(review_random_set, random_state=0, test_size=0.25, shuffle=True)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train.combined)
vectorizer.fit(test.combined)
x_train = vectorizer.transform(train.combined)
y_train = train.iloc[:,9:17]
x_test = vectorizer.transform(test.combined)
y_test = test.iloc[:,9:17]


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,f1_score, precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import xgboost

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=-1)),
            ])



model = xgboost.XGBClassifier()

from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(MultinomialNB(alpha = 1e-1), n_estimators = 10, 
                            max_samples = 200, bootstrap = True, n_jobs = -1)

report_matrix = pd.DataFrame(columns = ['acc', 'recall', 'precision', 'f1'])

for category in categories:
    print('**Processing {} comments...**'.format(category))
    
    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])
    
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    #print(test[category])
    #print(prediction.reshape(-1,1))
    print('Test recall is {}'.format(recall_score(test[category], prediction,
                                                  average='macro')))
    print('Test f1 is {}'.format(f1_score(test[category], prediction, average='weighted')))
    print('Test precision is {}'.format(precision_score(test[category], 
                                                        prediction, average='macro')))
    
    
    print(confusion_matrix(test[category], prediction))
    print("\n")
    


######
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize=(40,25))
# clean
subset = review_random_set[review_random_set.flavour==1]
text = subset.combined.values
cloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))
plt.axis('off')
plt.title("Flavour",fontsize=40)
plt.imshow(cloud)

review_random_set = pd.read_csv('binary_reviews.csv')
review_random_set = review_random_set.drop(columns = ['Unnamed: 0'])
# %%
##### figuring out the sentiment part and representation
copy = review_random_set.copy()
copy['sentiment'] = 1
copy.iloc[500:, 18 ] = -1

for row in range(0, len(copy)):
    copy.iloc[row, 9] = copy.iloc[row, 9] * copy.iloc[row, 18]
    
for row in range(0, len(copy)):
    copy.iloc[row, 10] = copy.iloc[row, 10] * copy.iloc[row, 18]
    
for row in range(0, len(copy)):
    copy.iloc[row, 11] = copy.iloc[row, 11] * copy.iloc[row, 18]
    
for row in range(0, len(copy)):
    copy.iloc[row, 12] = copy.iloc[row, 12] * copy.iloc[row, 18]
    
for row in range(0, len(copy)):
    copy.iloc[row, 13] = copy.iloc[row, 13] * copy.iloc[row, 18]
    
for row in range(0, len(copy)):
    copy.iloc[row, 14] = copy.iloc[row, 14] * copy.iloc[row, 18]
    
for row in range(0, len(copy)):
    copy.iloc[row, 15] = copy.iloc[row, 15] * copy.iloc[row, 18]
    
for row in range(0, len(copy)):
    copy.iloc[row, 16] = copy.iloc[row, 16] * copy.iloc[row, 18]


df = copy.iloc[:, 9:17][copy.retailer =='Amazon-US']
countspos = []
countsneg = []
categories = list(df.columns.values)
for i in categories:
    countspos.append((df[i][df[i]>0].sum()))
for i in categories:
    countsneg.append((df[i][df[i]<0].sum()))   
    
    
df_stats = pd.DataFrame(counts, columns=['category', 'number'])
    
df_stats.plot(x='category', y='number', kind='bar', 
              legend=False, grid=True, figsize=(8, 5))
plt.title("Number of instances per category- Tesco")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)  
plt.show()




#countsneg = [-18, -6, -2, -6, -3, 0, 0, -1]

countsneg = [x * -1 for x in countsneg]


x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, countspos, width, label='Pos')
rects2 = ax.bar(x + width/2, countsneg, width, label='Neg')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
ax.set_title('Category sentiments')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()


#copy.to_csv('random_full.csv')
#### radar plot
from math import pi
df = copy.iloc[:, 9:17][copy.brand =='Tesco']

#values = df.sum().values.flatten().tolist()
#values += values[:1] # repeat the first value to close the circular graph
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8),
                       subplot_kw=dict(polar=True))

plt.xticks(angles[:-1], categories, color='grey', size=12)
plt.yticks(np.arange(-1, 1), ['0', '1', '2'],
           color='grey', size=12)
plt.ylim(-1,1)
ax.set_rlabel_position(30)
 
#ax.plot(angles, values, linewidth=1, linestyle='solid')
#x.fill(angles, values, 'skyblue', alpha=0.4)

plt.show()
#####
# 2 brands 
df = copy.iloc[:, 9:17][copy.brand =='Tesco'].values

val_c1 = df.sum(0)/np.count_nonzero(df, axis = 0)#/(df!=0)#.sum(0)#.mean()
val_c1 = val_c1.tolist()
val_c1 += val_c1[:1]
ax.plot(angles, val_c1, linewidth=1,
        linestyle='solid', label='Client c1')
ax.fill(angles, val_c1, 'skyblue', alpha=0.4)
 
# part 2
df2 = copy.iloc[:, 9:17][copy.brand =='Sainsburys']

val_c2=df2.sum(0)/np.count_nonzero(df2, axis = 0)
val_c2 = val_c2.tolist()

val_c2 += val_c2[:1]
ax.plot(angles, val_c2, linewidth=1,
        linestyle='solid', label='Client c2')
ax.fill(angles, val_c2, 'lightpink', alpha=0.4)
 
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()


import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,6.27)})
sns.set(style="darkgrid")
ax = sns.countplot(x = review_random_set.brand,
                   data = review_random_set.head(10), 
                   order = review_random_set['brand'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)   
# %% Vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
review_random_set['sentiment'] = np.nan
analyser = SentimentIntensityAnalyzer()
scores = analyser.polarity_scores(review_random_set.iloc[11,8])
scores['compound']

for row in range(0, len(review_random_set)):
    scores = analyser.polarity_scores(review_random_set.iloc[row,8])
    review_random_set.iloc[row, 18] = scores['compound']

    
print(analyser.polarity_scores(review_random_set.iloc[11,8]))
print(review_random_set.iloc[11,8])
# %%
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

######another way
from sklearn.model_selection import train_test_split
train, test = train_test_split(review_random_set, random_state=42, 
                               test_size=0.30, shuffle=True)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word',
                             ngram_range=(1,1), norm='l2')
vectorizer.fit(train.combined)
vectorizer.fit(test.combined)
x_train = vectorizer.transform(train.combined)
y_train = train.iloc[:,9:]

x_test = vectorizer.transform(test.combined)
y_test = test.iloc[:,9:]



# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset
# initialize label powerset multi-label classifier
classifier = text_clf = Pipeline([('vect', CountVectorizer(stop_words = "english",
                                              lowercase = True, ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', LabelPowerset(MultinomialNB(alpha=1e-1))),])#(MultinomialNB(alpha = 1e-1))
# train
classifier.fit(X_train, y_train)
# predict
predictions = classifier.predict(X_test)
np.mean(predictions == y_test)

# accuracy
print("Accuracy = ",accuracy_score(y_test, predictions))
print("\n")
from sklearn.metrics import multilabel_confusion_matrix


multilabel_confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))


print(review_random_set.iloc[10,4], '+', 
      review_random_set.iloc[10,5], '+',review_random_set.iloc[10,7], 
      '=', review_random_set.iloc[10,8])