#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression # linear regression
import matplotlib.pyplot as plt # Plotting graphs
import seaborn as sns


# # Information on the csv file

# In[2]:


csv = pd.read_csv("../input/economic-data-9-countries-19802020/Economic Data - 9 Countries (1980-2020).csv")


# **Finding number of null values per column**

# In[3]:


csv.info()


# **Replacing the Missing Values in the columns**

# In[4]:


# Replacing index prices with mean values
csv['index price'] = csv['index price'].fillna(csv['index price'].mean())

# Replacing inflation rates with mean values
csv['inflationrate'] = csv['inflationrate'].fillna(csv['inflationrate'].mean())

# Replacing excahnge rates with mean values
csv['exchange_rate'] = csv['exchange_rate'].fillna(csv['exchange_rate'].mean())

# Replacing GDP with mean values
csv['gdppercent'] = csv['gdppercent'].fillna(csv['gdppercent'].mean())

# Replacing per capita income with mean values
csv['percapitaincome'] = csv['percapitaincome'].fillna(csv['percapitaincome'].mean())

# Replacing unemployment rate with mean values
csv['unemploymentrate'] = csv['unemploymentrate'].fillna(csv['unemploymentrate'].mean())

# Replacing manifacturing output with mean values
csv['manufacturingoutput'] = csv['manufacturingoutput'].fillna(csv['manufacturingoutput'].mean())

# Replacing trade balance with mean values
csv['tradebalance'] = csv['tradebalance'].fillna(csv['tradebalance'].mean())


# In[5]:


# Checking if all values are filled
csv.info()


# Value distributions of the dataset

# In[6]:


csv.hist(figsize=(30, 40), bins=30)


# # Correlation Matrix and Heat Map

# In[7]:


corr = csv.corr('pearson')


# In[8]:


plt.figure(figsize = (15,15))
sns.heatmap(corr, vmax=1.0, vmin=-1.0, square=True, annot=True, annot_kws={"size": 9, "color": "black"}, 
            linewidths=0.1, cmap='rocket')


# **Correlations with Index Prices**

# In[9]:


corr1 = corr['log_indexprice'].drop(['log_indexprice'])
corr1.sort_values(ascending=False)


# **Regression Plots as per strengths of correlation with index prices**

# In[10]:


# Identifying the strong correlated features (corr > 0.6)
strong_corr = corr1[abs(corr1) >= 0.6].sort_values(ascending=False).index.tolist()
print('\n Strongly correlated features: ', strong_corr, '\n')

# Strong features
strong_fet = csv.loc[:, strong_corr + ['log_indexprice']]

# Plot of strong feature's regression
fig, ax = plt.subplots(1, 2, figsize = (20,10))

for i, ax in enumerate(ax):
    if i < len(strong_corr):
        sns.regplot(x=strong_corr[i], y='log_indexprice', data=strong_fet, ax=ax, line_kws={'color': 'red'})


# In[11]:


# moderate correlation features ( > 0.35 & < 0.6)

moderate_corr = corr1[(abs(corr1) >= 0.35) & (abs(corr1) < 0.6)].sort_values(ascending=False).index.tolist()
print('\n Moderate correlation features: ', moderate_corr, '\n')

moderate_fet = csv.loc[:, moderate_corr + ['log_indexprice']]

fig, ax = plt.subplots(2, 2, figsize=(30, 30))

for i, ax in enumerate(fig.axes):
    if i < len(moderate_corr):
        sns.regplot(x=moderate_corr[i], y='log_indexprice', data=moderate_fet, ax=ax, line_kws={'color': 'red'})


# In[12]:


weak_corr = corr1[(abs(corr1) < 0.35)].sort_values(ascending=False).index.tolist()
print('\n Weakly correlated features: ', weak_corr, '\n')

weak_fet = csv.loc[:, weak_corr + ['log_indexprice']]

fig, ax = plt.subplots(3, 2, figsize=(30, 30))

for i, ax in enumerate(fig.axes):
    if i < len(weak_corr):
        sns.regplot(x=weak_corr[i], y='log_indexprice', data=weak_fet, ax=ax, line_kws={'color': 'red'})


# # Linear Regression

# In[13]:


# Providing data in terms of array
x = np.array(csv['inflationrate']).reshape((-1, 1))
y = np.array(csv['log_indexprice'])

# Defining a model and fit it 
reg = LinearRegression().fit(x, y)

# Results of the regression
rsquared = reg.score(x, y)
r_sq = reg.score(x, y)
print(f"coefficient of determination: {r_sq}")

# Intercept of regression
print(f"intercept: {reg.intercept_}")

# Slope of the regressor
print(f"slope: {reg.coef_}")


# # Multivariate Distributions: Pairplot

# In[14]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(data = csv, hue = 'stock index')

