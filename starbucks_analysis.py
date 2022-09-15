# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 06:38:44 2022

@author: KalyanRuchiPC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('starbucks_data.csv')

print(df.info())

print(df.columns[1:12])

for col in df.columns[1:12]:
    df[col] = df[col].astype('category')

df.columns = [ 'Timestamp', 'Gender', 'Age', 'Employment', 'Income', 'visit_frequency', 'visit_type', 'visit_time_spent', 
                           'nearest_starbucks', 'memcard_available', 'frequent_purchase', 'avg_spent_per_visit',
                           'brand_rating', 'price_rating', 'promotion_rating', 'ambience_rating', 'wifi_rating',
                           'service_rating', 'choosing_stb_rating', 'promotion_heard_from', 'willing_to_visit_stb']
                           


#male female avg_spent_per_visit

gender_df = df.groupby('Gender')['avg_spent_per_visit'].value_counts().reset_index()

overall_spent_per_visit = df['avg_spent_per_visit'].value_counts().reset_index()


plt.pie(overall_spent_per_visit['avg_spent_per_visit'], labels = overall_spent_per_visit['index'], autopct='%.0f%%')
plt.title('overall spent per visit')
plt.show()


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))


ax1.pie(gender_df[gender_df['Gender'] == 'Male']['avg_spent_per_visit'], labels = gender_df[gender_df['Gender'] =='Male']['level_1'], autopct='%.0f%%')
ax1.set_title('Male avg spent')

ax2.pie(gender_df[ gender_df['Gender'] == 'Female']['avg_spent_per_visit'], labels = gender_df[ gender_df['Gender'] == 'Female']['level_1'], autopct='%.0f%%')
ax2.set_title('Female avg spent')

plt.show()



sns.lineplot(data = overall_spent_per_visit, x='index', y= 'avg_spent_per_visit', marker='o')
sns.barplot(data = gender_df, x='level_1', y='avg_spent_per_visit', hue='Gender')
plt.xticks(rotation = 45)
plt.show()


gender_df2 = df.groupby(['Gender', 'Age'])['avg_spent_per_visit'].value_counts().reset_index()

fig, ax = plt.subplots(figsize=(20,10))

sns.barplot(data=gender_df2, x='level_2', y='avg_spent_per_visit', hue=gender_df2[['Gender', 'Age']].apply(tuple, axis=1), order=['Zero', 'Less than RM20', 'Around RM20 - RM40', 'More than RM40'], ax=ax)
plt.xticks(rotation=45)
plt.show()


gender_df3 = df.groupby('Age')['avg_spent_per_visit'].value_counts().reset_index().sort_values('avg_spent_per_visit', ascending=True)
sns.barplot(data=gender_df3, x='level_1', y='avg_spent_per_visit', hue='Age', order=['Zero', 'Less than RM20', 'Around RM20 - RM40', 'More than RM40'])
plt.xticks(rotation=45)
plt.show()

gender_df4 = df.groupby('Employment')['avg_spent_per_visit'].value_counts().reset_index().sort_values('avg_spent_per_visit', ascending=True)
sns.barplot(data=gender_df4, x = 'level_1', y='avg_spent_per_visit', hue='Employment', order=['Zero', 'Less than RM20', 'Around RM20 - RM40', 'More than RM40'])
plt.xticks(rotation=45)
plt.show()


fig, ax2 = plt.subplots(figsize=(20,10))

gender_df4 = df.groupby(['Employment', 'Gender'])['avg_spent_per_visit'].value_counts().reset_index().sort_values('avg_spent_per_visit', ascending=True)
sns.barplot(data=gender_df4, x = 'level_2', y='avg_spent_per_visit', hue=gender_df4[['Employment', 'Gender']].apply(tuple, axis=1), ax=ax2, order=['Zero', 'Less than RM20', 'Around RM20 - RM40', 'More than RM40'])
plt.xticks(rotation=45)
plt.show()


#does more time spent result in more spending
time_df = df.groupby('visit_type')['avg_spent_per_visit'].value_counts().reset_index().sort_values('avg_spent_per_visit', ascending=True)

time_df = time_df[time_df['visit_type'].isin(['Take away', 'Dine in', 'Drive-thru'])]
sns.barplot(data=time_df, x='level_1', y='avg_spent_per_visit', hue=time_df['visit_type'])
plt.xticks(rotation=45)
plt.show()

fig, ax2 = plt.subplots(figsize=(20,10))

time_df2 = df.groupby(['visit_type', 'visit_time_spent'])['avg_spent_per_visit'].value_counts().reset_index().sort_values('avg_spent_per_visit', ascending=True)
time_df2 = time_df2[time_df2['visit_type'] == 'Dine in']
sns.barplot(data=time_df2, x = 'level_2', y='avg_spent_per_visit',hue=time_df2[['visit_type', 'visit_time_spent']].apply(tuple, axis=1), ax=ax2, order=['Zero', 'Less than RM20', 'Around RM20 - RM40', 'More than RM40'])
plt.xticks(rotation=45)
plt.show()

#visit time spent

dine_inzero = df[(df['avg_spent_per_visit'] == 'Zero')]

#rating box plot


fig, ax2 = plt.subplots(figsize=(30,20))

ax2.boxplot([df['brand_rating'], df['price_rating'], df['promotion_rating'], df['ambience_rating'], df['wifi_rating'], df['service_rating'], df['choosing_stb_rating']])
ax2.set_yticks([1,2,3,4,5])
plt.show()


fig, ax = plt.subplots(3, 3, figsize=(30,20))

sns.countplot(x=df['brand_rating'], ax=ax[0,0])
sns.countplot(x=df['price_rating'], ax=ax[0,1])
sns.countplot(x=df['promotion_rating'], ax=ax[0,2])
sns.countplot(x=df['ambience_rating'], ax=ax[1,0])
sns.countplot(x=df['wifi_rating'], ax=ax[1,1])
sns.countplot(x=df['service_rating'], ax=ax[1,2])
sns.countplot(x=df['choosing_stb_rating'], ax=ax[2,0])
sns.countplot(x=df['willing_to_visit_stb'], ax=ax[2,2])
plt.show()

#does membership affect the average spent per visit
fig, ax = plt.subplots(figsize=(30,20))
mem_df = df.groupby(['memcard_available'])['avg_spent_per_visit'].value_counts().reset_index().sort_values('avg_spent_per_visit', ascending=True)

sns.barplot(data=mem_df, x='level_1', y='avg_spent_per_visit', hue='memcard_available', ax=ax, order=['Zero', 'Less than RM20', 'Around RM20 - RM40', 'More than RM40'])
plt.show()

fig, ax2 = plt.subplots(figsize=(40,20))
mem_df2 = df.groupby(['Gender', 'Age', 'memcard_available'])['avg_spent_per_visit'].value_counts().reset_index().sort_values('avg_spent_per_visit', ascending=True)
sns.barplot(data=mem_df2, x='level_3', y='avg_spent_per_visit', hue=mem_df2[['Age', 'memcard_available']].apply(tuple, axis=1), ax=ax2, order=['Zero', 'Less than RM20', 'Around RM20 - RM40', 'More than RM40'])
plt.xticks(rotation=45)
plt.show()

df.describe()


#rating across different age groups

rat_df = df.groupby('Age').agg({'brand_rating': np.mean, 'price_rating': np.mean, 'ambience_rating': np.mean, 'promotion_rating': np.mean, 'wifi_rating': np.mean, 'service_rating': np.mean, 'choosing_stb_rating': np.mean })


#calculating CLT

sample = df['choosing_stb_rating'].sample(50, replace=True)

clt_data = []

for i in range(10000):
    clt_data.append(np.mean(df['choosing_stb_rating'].sample(50, replace=True)))

plt.hist(clt_data)
plt.title('95% C.I for Choosing STB = (3.24, 3.78)')

plt.show()

confi_interval = np.percentile(clt_data, [2.5, 97.5])

print(confi_interval)


fig, ax = plt.subplots(3, 3, figsize=(30,20))

all_clt = []



def calculate_clt(col):
    print(col)
    tt = []
    for i in range(10000):
        tt.append(np.mean(df[col].sample(50, replace=True)))
    ci = np.percentile(tt, [2.5, 97.5])
    return [tt, ci]



clmns = list(df.columns)

ind = 12

for i in range(0,3):
    for j in range(0, 3):
        
        if ind >= 19:
            break
        res = calculate_clt(clmns[ind])
        
        
        
        ax[i, j].hist(res[0])
        ax[i, j].set_title(clmns[ind] + ' = '+ str(res[1][0]) + ', ' + str(res[1][1]))
        ind += 1


import pingouin as pg


print(pg.ttest(x=df['price_rating'], y=df['wifi_rating']))
test = df.anova(dv="price_rating", between=["Gender", "Age"]).round(3)