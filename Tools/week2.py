import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn
import scipy.stats

df=pd.read_csv("~/datasets/nesarc_pds.csv")

df["S3AQ3B1"]=df["S3AQ3B1"].convert_objects(convert_numeric=True)
df["S3AQ3C1"]=df["S3AQ3C1"].convert_objects(convert_numeric=True)
df["CHECK321"]=df["CHECK321"].convert_objects(convert_numeric=True)

sub=df[(df["AGE"]>=18) & (df["AGE"]<=25) & (df["CHECK321"]==1)]

sub["S3AQ3B1"]=sub["S3AQ3B1"].replace(9, np.nan)
sub["S3AQ3C1"]=sub["S3AQ3C1"].replace(9, np.nan)

recode={1: 30, 2:22, 3:14, 4:5, 5:2, 6:1}
sub["USFREQMO"]=sub["S3AQ3B1"].map(recode)
sub["USFREQMO"]=sub["USFREQMO"].convert_objects(convert_numeric=True)

sub["NUMCIGMO_EST"]=sub["USFREQMO"]*sub["S3AQ3C1"]
sub["NUMCIGMO_EST"]=sub["NUMCIGMO_EST"].convert_objects(convert_numeric=True) 

# contingency table of observed counts
ct1=pd.crosstab(sub["TAB12MDX"],sub["USFREQMO"])
print(ct1)

# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)
# the first value in the printed table is c2 and the 2nd value is the p value
cs1=scipy.stats.chi2_contingency(ct1)
print(cs1)

# plotting the bar plot to see the relationship
sub["TAB12MDX"]=sub["TAB12MDX"].convert_objects(convert_numeric=True)
sub["USFREQMO"]=sub["USFREQMO"].astype("category")
seaborn.factorplot(x="USFREQMO",y="TAB12MDX",data=sub,kind='bar',ci=None)
plt.xlabel("User frequency")
plt.ylabel("Nicotine Dependence")
plt.show()