#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Analysis

# In[1]:


#import the required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# **Load the data file **

# In[3]:


telco_base_data = pd.read_csv('Customer_Churn.csv')


# Look at the top 5 records of data

# In[4]:


telco_base_data.head()


# Check the various attributes of data like shape (rows and cols), Columns, datatypes

# In[6]:


telco_base_data.shape


# In[7]:


telco_base_data.columns.values


# In[8]:


# Checking the data types of all the columns
telco_base_data.dtypes


# In[10]:


telco_base_data.describe()


# SeniorCitizen is actually a categorical hence the 25%-50%-75% distribution is not propoer
# 
# 75% customers have tenure less than 55 months
# 
# Average Monthly charges are USD 64.76 whereas 25% customers pay more than USD 89.85 per month

# In[11]:


telco_base_data['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);


# In[12]:


100*telco_base_data['Churn'].value_counts()/len(telco_base_data['Churn'])


# In[13]:


telco_base_data['Churn'].value_counts()


# * Data is highly imbalanced, ratio = 73:27<br>

# In[14]:


telco_base_data.info(verbose = True) 


# In[15]:


missing = pd.DataFrame((telco_base_data.isnull().sum())*100/telco_base_data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# ## Data Cleaning
# 

# In[17]:


telco_data = telco_base_data.copy()


# Total Charges should be numeric amount. Let's convert it to numerical data type

# In[18]:


telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
telco_data.isnull().sum()


# **3.** As we can see there are 11 missing values in TotalCharges column. Let's check these records 

# In[19]:


telco_data.loc[telco_data ['TotalCharges'].isnull() == True]


# **4. Missing Value Treatement**

# In[21]:


telco_data.dropna(how = 'any', inplace = True)
#telco_data.fillna(0)


# **5.** Divide customers into bins based on tenure e.g. for tenure < 12 months: assign a tenure group if 1-12, for tenure between 1 to 2 Yrs, tenure group of 13-24; so on...

# In[22]:


print(telco_data['tenure'].max()) #72


# In[23]:


labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)


# In[24]:


telco_data['tenure_group'].value_counts()


# **6.** Remove columns not required for processing

# In[25]:


telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
telco_data.head()


# ## Data Exploration

# ### Univariate Analysis

# In[26]:


for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telco_data, x=predictor, hue='Churn')


# **2.** Convert the target variable in a binary numeric variable i.e. Yes=1 ; No = 0

# In[27]:


telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)


# In[28]:


telco_data.head()


# **3.** Convert all the categorical variables into dummy variables

# In[29]:


telco_data_dummies = pd.get_dummies(telco_data)
telco_data_dummies.head()


# **9. ** Relationship between Monthly Charges and Total Charges

# In[30]:


sns.lmplot(data=telco_data_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# In[27]:


plt.figure(figsize=(20,8))
telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# ### Bivariate Analysis

# In[31]:


new_df1_target0=telco_data.loc[telco_data["Churn"]==0]
new_df1_target1=telco_data.loc[telco_data["Churn"]==1]


# In[32]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[33]:


uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[34]:


uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')


# In[35]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')


# In[36]:


uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')


# In[37]:


uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')


# In[38]:


uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


# In[55]:


telco_data_dummies.to_csv('tel_churn.csv')

