import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats
import seaborn
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import statsmodels.api as sm
import matplotlib.figure
from pylab import *


df = pd.read_csv('~/datasets/nesarc_pds.csv', low_memory=False)

#setting variables you will be working with to numeric
df['IDNUM'] =pd.to_numeric(df['IDNUM'], errors='coerce')
df['TAB12MDX'] = pd.to_numeric(df['TAB12MDX'], errors='coerce')
df['MAJORDEPLIFE'] = pd.to_numeric(df['MAJORDEPLIFE'], errors='coerce')
df['SOCPDLIFE'] = pd.to_numeric(df['SOCPDLIFE'], errors='coerce')
df['S3AQ3C1'] = pd.to_numeric(df['S3AQ3C1'], errors='coerce')
df['AGE'] =pd.to_numeric(df['AGE'], errors='coerce')
df['SEX'] = pd.to_numeric(df['SEX'], errors='coerce')

sub1=df[(df['AGE']<=25) & (df['CHECK321']==1) & (df['S3AQ3B1']==1)]
print("############################################################")
print(sub1.head())

# creating binary nicotine dependence variable :-
def NICOTINEDEP (row):
	if row["TAB12MDX"]==1:
		return 1
	else :
		return 0

sub1["NICOTINEDEP"]= sub1.apply(lambda x: NICOTINEDEP(x), axis=1)

# logistic regression on social phobia 
# NICOTINDEP is binary and SOCPDLIFE is the variable that indicates the presence or absence of social phobia 
lreg1= smf.logit(formula="NICOTINEDEP ~ SOCPDLIFE ", data=sub1).fit()
print(lreg1.summary())

# odds ratio:- 
print(np.exp(lreg1.params))