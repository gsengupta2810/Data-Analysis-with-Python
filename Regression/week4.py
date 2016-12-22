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

##########################################################################################
#   			Multiple regression for C->Q type with 3 or more catagories
##########################################################################################
df=pd.read_csv("~/datasets/nesarc_pds.csv", low_memory=False)

df['IDNUM'] =pd.to_numeric(df['IDNUM'], errors='coerce')
df['TAB12MDX'] = pd.to_numeric(df['TAB12MDX'], errors='coerce')
df['MAJORDEPLIFE'] = pd.to_numeric(df['MAJORDEPLIFE'], errors='coerce')
df['SOCPDLIFE'] = pd.to_numeric(df['SOCPDLIFE'], errors='coerce')
df['S3AQ3C1'] = pd.to_numeric(df['S3AQ3C1'], errors='coerce')
df['AGE'] =pd.to_numeric(df['AGE'], errors='coerce')
df['SEX'] = pd.to_numeric(df['SEX'], errors='coerce')

df['S3AQ3B1'] = pd.to_numeric(df['S3AQ3B1'], errors='coerce')
df['CHECK321'] =pd.to_numeric( df['CHECK321'], errors='coerce')
df['S3AQ8B11'] = pd.to_numeric(df['S3AQ8B11'], errors='coerce')
df['S3AQ8B12'] = pd.to_numeric(df['S3AQ8B12'], errors='coerce')
df['S3AQ8B13'] = pd.to_numeric(df['S3AQ8B13'], errors='coerce')
df['S3AQ8B7A'] = pd.to_numeric(df['S3AQ8B7A'], errors='coerce')
df['S3AQ8B7B'] = pd.to_numeric(df['S3AQ8B7B'], errors='coerce')
df['S3AQ8B7C'] = pd.to_numeric(df['S3AQ8B7C'], errors='coerce')
df['S3AQ8B7D'] = pd.to_numeric(df['S3AQ8B7D'], errors='coerce')
df['S3AQ8B7E'] = pd.to_numeric(df['S3AQ8B7E'], errors='coerce')
df['S3AQ8B7F'] = pd.to_numeric(df['S3AQ8B7F'], errors='coerce')
df['S3AQ8B7G'] = pd.to_numeric(df['S3AQ8B7G'], errors='coerce')
df['S3AQ8B7H'] = pd.to_numeric(df['S3AQ8B7H'], errors='coerce')
df['S3AQ8B7J'] = pd.to_numeric(df['S3AQ8B7J'], errors='coerce')

df['S6Q1'] = pd.to_numeric(df['S6Q1'], errors='coerce')
df['S6Q2'] = pd.to_numeric(df['S6Q2'], errors='coerce')
df['S6Q3'] = pd.to_numeric(df['S6Q3'], errors='coerce')
df['S6Q7'] = pd.to_numeric(df['S6Q7'], errors='coerce')
df['S6Q61'] = pd.to_numeric(df['S6Q61'], errors='coerce')
df['S6Q62'] = pd.to_numeric(df['S6Q62'], errors='coerce')
df['S6Q63'] = pd.to_numeric(df['S6Q63'], errors='coerce')
df['S6Q64'] = pd.to_numeric(df['S6Q64'], errors='coerce')
df['S6Q65'] = pd.to_numeric(df['S6Q65'], errors='coerce')
df['S6Q66'] = pd.to_numeric(df['S6Q66'], errors='coerce')
df['S6Q67'] = pd.to_numeric(df['S6Q67'], errors='coerce')
df['S6Q68'] = pd.to_numeric(df['S6Q68'], errors='coerce')
df['S6Q69'] = pd.to_numeric(df['S6Q69'], errors='coerce')
df['S6Q610'] = pd.to_numeric(df['S6Q610'], errors='coerce')
df['S6Q611'] = pd.to_numeric(df['S6Q611'], errors='coerce')
df['S6Q612'] = pd.to_numeric(df['S6Q612'], errors='coerce')
df['S6Q613'] = pd.to_numeric(df['S6Q613'], errors='coerce')

df['S3AQ3C1']=df['S3AQ3C1'].replace(99, np.nan)

sub1=df[(df['AGE']<=25) & (df['CHECK321']==1) & (df['S3AQ3B1']==1) & 
(df['IDNUM']!=20346) & (df['IDNUM']!=36471) & (df['IDNUM']!=28724)]

# rename variables
sub1.rename(columns={'S3AQ3C1': 'numbercigsmoked'}, inplace=True)

sub1['NDSymptoms'] = np.nansum([sub1['crit1'], sub1['crit2'], sub1['S3AQ8B13'], 
              sub1['crit4'], sub1['S3AQ8B5'], sub1['crit6'],
              sub1['crit7']], axis=0 )
df['NDSymptoms'] = pd.to_numeric(df['NDSymptoms'], errors='coerce')

# center quantitative IVs for regression analysis
sub1['numbercigsmoked_c'] = (sub1['numbercigsmoked'] - sub1['numbercigsmoked'].mean())
print (sub1['numbercigsmoked_c'].mean()) 
sub1['age_c']=(sub1['AGE'] - sub1['AGE'].mean())
print (sub1['age_c'].mean()) 

# adding 4 category ethnicity/race. Reference group coding is called "Treatment" coding in python
# and the default reference catergory is the group with a value = 0 (Hispanic)
reg6 = smf.ols('NDSymptoms ~ DYSLIFE + MAJORDEPLIFE + numbercigsmoked_c + age_c + SEX + C(ETHRACE)', 
               df=sub1).fit()
print (reg6.summary())
