# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
import seaborn as sns

# %%
df = pd.read_csv('data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# %%
print("Checking for Missing values in dataset:\n",df.isnull().sum())

# %%
df1 = df.drop(['restecg', 'fbs', 'exang', 'slope', 'oldpeak', 'sex', 'target'], axis=1, inplace=False)
df_scaled = (df1-np.min(df1, axis=0))/(np.max(df1, axis=0)-np.min(df1, axis=0)).values
plt.figure(figsize=(10,6))
sns.boxplot(data=df_scaled)

# %%
import statsmodels.api as sm

plt.figure(figsize=(20,7))
sns.heatmap(df.corr(), annot = True, cmap="Blues")

# %%
from statsmodels.graphics.gofplots import qqplot

df2 = df[['age', 'oldpeak', 'thalach', 'cp', 'sex']]
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, figsize=(20,4))
axis = [ax1, ax2, ax3, ax4, ax5]
for idx, c in enumerate(df2.columns[:]):
  qqplot(df2[c], line='45', fit='True', ax=axis[idx])
  axis[idx].set_title("Q-Q plot of {}".format(c),fontsize=16)

plt.show()

# %%
yes_hd = df['age'][(df['target']==1)]
yes_hd_mean = df['age'][(df['target']==1)].mean()
print(yes_hd)
print("mean of patients with heart disease:",yes_hd_mean)

# %%
no_hd = df['age'][(df['target']==0)]
no_hd_mean = df['age'][(df['target']==0)].mean()
print(no_hd)
print("mean of patients with no heart disease:",no_hd_mean)

# %%
t_statistic, p_value = stats.ttest_ind(yes_hd, no_hd)

alpha = 0.05
# Compute the degrees of freedom (df) (n_A-1)+(n_b-1)
dof = len(no_hd)+len(yes_hd)-2

# Calculate the critical t-value
# ppf is used to find the critical t-value for a two-tailed test
critical_t = stats.t.ppf(1 - alpha/2, dof)
print("T-statistic:", t_statistic)
print("P-value:", p_value)
print("Critical t-value:", critical_t)

print('With T-value')
if np.abs(t_statistic) >critical_t:
    print('There is significant difference between two groups')
else:
    print('No significant difference found between two groups')

print('With P-value')
if p_value >alpha:
    print('No evidence to reject the null hypothesis that a significant difference between the two groups')
else:
    print('Evidence found to reject the null hypothesis, Hence there is a significant difference between the two age groups')

# %%
yes_hd = df['age'][(df['sex']==1) & (df['target']==1)]
print(yes_hd)

# %%
yes_hd = df['age'][(df['sex']==1) & (df['target']==0)]
print(yes_hd)

# %%
# chi-square test
contingency_table = pd.crosstab(df['sex'], df['target'])
chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2_stat)
print("P-value:", p_val_chi2)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)

alpha = 0.05
print('With P-value')
if p_value >alpha:
    print('No evidence to reject the null hypothesis that a significant difference between the two groups')
else:
    print('Evidence found to reject the null hypothesis, Hence there is a significant difference between the two sex groups')


