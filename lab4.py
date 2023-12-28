#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
from scipy import stats
df = pd.read_csv(r"C:\Users\busra\OneDrive\Masaüstü\Yapay zeka\bulut bilişim\Iris_Data.csv")


# In[63]:


df.groupby("species")['sepal_width'].describe()


# In[64]:


setosa = df[(df['species'] == 'Iris-setosa')]
setosa.reset_index(inplace= True)

versicolor = df[(df['species'] == 'Iris-versicolor')]
versicolor.reset_index(inplace= True)


# In[65]:


stats.levene(setosa['sepal_width'], versicolor['sepal_width'])


# In[66]:


diff = setosa['sepal_width'] - versicolor['sepal_width']


# In[67]:


from scipy import stats
import matplotlib.pyplot as plt

stats.probplot(diff, plot= plt)
plt.title('Sepal Width P-P Plot')
plt.savefig('Sepal Width Residuals.png')


# In[68]:


diff.plot(kind= "hist", title= "Sepal Width Residuals")
plt.xlabel("Length (cm)")
plt.savefig("Residuals Plot of Sepal Width.png")


# In[69]:


stats.ttest_ind(setosa['sepal_width'], versicolor['sepal_width'])


# In[70]:


get_ipython().system('pip install researchpy')


# In[71]:


import researchpy as rp


# In[72]:


df.groupby("species")['sepal_width'].describe()


# In[73]:


rp.summary_cont(df.groupby("species")['sepal_width'])


# In[85]:


descriptives, results = rp.ttest(setosa['sepal_width'], versicolor['sepal_width'])

descriptives


# In[84]:


results


# In[80]:


import pandas as pd

df = pd.read_csv(r"C:\Users\busra\OneDrive\Masaüstü\Yapay zeka\bulut bilişim\blood_pressure.csv")


df[['bp_before', 'bp_after']].describe()


# In[81]:


from scipy import stats
import matplotlib.pyplot as plt

df[['bp_before', 'bp_after']].plot(kind= 'box')

plt.savefig('boxplot_outliers.png')
plt.show()


# In[82]:


df['bp_difference'] = df['bp_before'] - df['bp_after']

df['bp_difference'].plot(kind='hist', title= 'Blood Pressure Difference Histogram')

plt.savefig('blood pressure difference histogram.png')
plt.show()


# In[83]:


stats.probplot(df['bp_difference'], plot= plt)
plt.title('Blood pressure Difference Q-Q Plot')
plt.savefig('blood pressure difference qq plot.png')
plt.show()


# In[40]:


stats.shapiro(df['bp_difference'])


# In[41]:


from scipy.stats import ttest_rel
ttest_rel(df['bp_before'],df['bp_after'])


# In[43]:


import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\busra\\OneDrive\\Masaüstü\\Yapay zeka\\bulut bilişim\\difficile.csv")
df.drop('person', axis=1, inplace=True)


# In[44]:


df['dose'].replace({1: 'placebo', 2: 'low', 3: 'high'}, inplace= True)

rp.summary_cont(df['libido'])


# In[45]:


rp.summary_cont(df['libido'].groupby(df['dose']))


# In[46]:


stats.f_oneway(df['libido'][df['dose'] == 'high'], 
              df['libido'][df['dose'] == 'low'],
              df['libido'][df['dose'] == 'placebo'])


# In[56]:


results = ols('libido ~ C(dose)', data=df).fit()
aov_table = sm.stats.anova_lm(results, typ=2)
print(aov_table)


# In[ ]:




