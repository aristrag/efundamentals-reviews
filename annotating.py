# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:58:04 2020

@author: atrag
"""
import numpy as np
import pandas as pd
review_sentences = pd.read_pickle('review_sentences.pkl')


products = pd.read_csv('products.csv')
#creating dict with product key and category
product_dict = dict(zip(products.product_id, products.category))
#add category
review_sentences['category'] = np.nan
for row in range(0, len(review_sentences)):
        try:
            review_sentences.iloc[row, 6] = product_dict[review_sentences.iloc[row,2]]
            print(review_sentences.iloc[row,2])
        except KeyError:
            print("Oops!  That was no valid number.  Try again...")
            
            
#creating the feature
###treating nans in subject and category
review_sentences.fillna('', inplace=True)
review_sentences['combined'] = ''
for row in range(0, len(review_sentences)):
    review_sentences.iloc[row,7] = review_sentences.iloc[row,4] + ' '+review_sentences.iloc[row,5] + ' '+review_sentences.iloc[row,6]

review_sentences.to_csv('sentences_full1.csv')
# %% classification using rule based annotation
pricelist = r"(expensive|cheap|price|money|dollar|pricey|pricy|value|priced|(\bover priced)|(\brip off)|cheapest)"
sizelist =  r"(big|size|small|portion|tiny|huge|amount)"
flavourlist =  r"(edible|pastry|tender|crunch|taste|eating|butter|sucker|tasty|yuk|delicious|yummy|yum|puke|sweet|tast|eat|eating|flavour|flavor|punch|food|smell|wet|vodka|wine|beer|breakfast|meal|lunch|dinner|light|fragrance|treat|treats|moist|coffee|refreshing|drink|tastes|smells|chocolate|snack|creamy|rich|candy)"
ingredientlist = r"(ingredient|healthy|quality|protein|carbs|vegetable|meat|beef|chicken|chunks|(\bfilled with)|spinach|veg|fruit|diet|fresh|garbage|bin|raw|sugar|chemical|burn)"
shippinglist = r"(shipping|delivery|came|quickly|delivered|shipped)"
packaginglist = r"(burst|damaged|damage|opened|openned|tear|packaging|seal|sealing)"
functionalitylist = r"(construct|cleaning|works|perform|performance|help|helped|use|easy|works|experience|device|functionality|setup|connection|skin|face|(\bused this product)|conditioning|protection|sound|display)"
presentationlist = r"(looks|visual|appear|presented|neat|laminate|glow|sparkly|look|colour|color|boxy)"

import re
review_sentences['label'] = [list() for x in range(len(review_sentences.index))]
review_sentences.fillna('', inplace=True)

review_sentences.label.values.tolist()
for row in range(0, len(review_sentences)):
  if re.search(pricelist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,8].append('price')

for row in range(0, len(review_sentences)):
  if re.search(sizelist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,8].append('size')

for row in range(0, len(review_sentences)):
  if re.search(flavourlist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,8].append('flavour')
    
for row in range(0, len(review_sentences)):
  if re.search(ingredientlist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,8].append('ingredients') 

for row in range(0, len(review_sentences)):
  if re.search(shippinglist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,8].append('shipping') 
    
for row in range(0, len(review_sentences)):
  if re.search(packaginglist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,8].append('packaging') 
    
for row in range(0, len(review_sentences)):
  if re.search(functionalitylist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,8].append('functionality') 

for row in range(0, len(review_sentences)):
  if re.search(presentationlist, review_sentences.iloc[row,5]):
    print(review_sentences.iloc[row,5])
    review_sentences.iloc[row,8].append('presentation')     


# %%  
#binarising the labels (one hot)
review_sentences.label =review_sentences.label.astype(str)
review_sentences.label
review_sentences['flavour'] = np.nan
for row in range(0, len(review_sentences)):
    if re.search('flavour', review_sentences.iloc[row,8]):
        print('found')
        review_sentences.iloc[row, 9] = 1
    else:
        print('nothing')
        review_sentences.iloc[row, 9] = 0

review_sentences['price'] = np.nan
for row in range(0, len(review_sentences)):
    if re.search('price', review_sentences.iloc[row,8]):
        print('found')
        review_sentences.iloc[row, 10] = 1
    else:
        print('nothing')
        review_sentences.iloc[row, 10] = 0
        
review_sentences['ingredients'] = np.nan
for row in range(0, len(review_sentences)):
    if re.search('ingredients', review_sentences.iloc[row,8]):
        print('found')
        review_sentences.iloc[row, 11] = 1
    else:
        print('nothing')
        review_sentences.iloc[row, 11] = 0
        
review_sentences['size'] = np.nan
for row in range(0, len(review_sentences)):
    if re.search('size', review_sentences.iloc[row,8]):
        print('found')
        review_sentences.iloc[row, 12] = 1
    else:
        print('nothing')
        review_sentences.iloc[row, 12] = 0
        
review_sentences['shipping'] = np.nan
for row in range(0, len(review_sentences)):
    if re.search('shipping', review_sentences.iloc[row,8]):
        print('found')
        review_sentences.iloc[row, 13] = 1
    else:
        print('nothing')
        review_sentences.iloc[row, 13] = 0
        
review_sentences['functionality'] = np.nan
for row in range(0, len(review_sentences)):
    if re.search('functionality', review_sentences.iloc[row,8]):
        print('found')
        review_sentences.iloc[row, 14] = 1
    else:
        print('nothing')
        review_sentences.iloc[row, 14] = 0
        
review_sentences['packaging'] = np.nan
for row in range(0, len(review_sentences)):
    if re.search('packaging', review_sentences.iloc[row,8]):
        print('found')
        review_sentences.iloc[row, 15] = 1
    else:
        print('nothing')
        review_sentences.iloc[row, 15] = 0
        
review_sentences['presentation'] = np.nan
for row in range(0, len(review_sentences)):
    if re.search('presentation', review_sentences.iloc[row,8]):
        print('found')
        review_sentences.iloc[row, 16] = 1
    else:
        print('nothing')
        review_sentences.iloc[row, 16] = 0
        
    
    
    
    
    
    
    
    
# %%    sentiment- perform after following cell
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
review_sentences['sentiment'] = np.nan
analyser = SentimentIntensityAnalyzer()
scores = analyser.polarity_scores(review_sentences.iloc[11,7])
scores['compound']

for row in range(0, len(review_sentences)):
    scores = analyser.polarity_scores(review_sentences.iloc[row,7])
    review_sentences.iloc[row, 18] = scores['compound']

    
print(analyser.polarity_scores(review_sentences.iloc[11,7]))
print(review_sentences.iloc[11,7])
# %%
#brands

product_dict_brand = dict(zip(products.product_id, products.brand))

review_sentences['brand'] = np.nan
for row in range(0, len(review_sentences)):
        try:
            review_sentences.iloc[row, 17] = product_dict_brand[review_sentences.iloc[row,2]]
            print(review_sentences.iloc[row,2])
        except KeyError:
            print("Oops!  That was no valid number.  Try again...")


# %% plotting
import matplotlib.pyplot as plt
df = review_sentences.iloc[:, 9:17]
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


copy = review_sentences.copy()


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

#copy.to_csv('with_sentiment.csv')
###neeeds fixing with the vader stuff
brands = copy.brand.value_counts()
df = copy.iloc[:, 9:17][copy.brand =='Nestle']
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
ax.set_title('Category sentiments - Nestle')
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
df = copy.iloc[:, 9:17][copy.brand =='Mars'].values

val_c1 = df.sum(0)/np.count_nonzero(df, axis = 0)#/(df!=0)#.sum(0)#.mean()
val_c1 = val_c1.tolist()
val_c1 += val_c1[:1]
ax.plot(angles, val_c1, linewidth=1,
        linestyle='solid', label='Mars')
ax.fill(angles, val_c1, 'skyblue', alpha=0.4)
 
# part 2 
df2 = copy.iloc[:, 9:17][copy.brand =='Nestle']

val_c2=df2.sum(0)/np.count_nonzero(df2, axis = 0)
val_c2 = val_c2.tolist()

val_c2 += val_c2[:1]
ax.plot(angles, val_c2, linewidth=1,
        linestyle='solid', label='Nestle')
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




