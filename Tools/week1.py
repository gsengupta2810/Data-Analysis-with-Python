import pandas as pd 
import numpy as np 
import seaborn 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

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

c1=sub.groupby("NUMCIGMO_EST").size()
print(c1)

# using the ols function to calculate the ANOVA F statistic..
# formula --- First is the numerical(quantitative) response variable, followed by ~ and by the catagorical explainatory variable
# catagorical variables have to be specified by putting a C in front of a paranthesis 
model1=smf.ols(formula="NUMCIGMO_EST ~C(MAJORDEPLIFE)", data=sub)
results1= model1.fit()
print(results1.summary())

# Now checking the standard deviation and mean of the catagories to actually compare them
sub1=sub[["NUMCIGMO_EST","MAJORDEPLIFE"]].dropna()
m1=sub1.groupby("MAJORDEPLIFE").mean()
print(m1)
sd1=sub1.groupby("MAJORDEPLIFE").std()
print(sd1)

# checking the amount of cigs consumed in different ethenic groups
sub2=sub[["NUMCIGMO_EST","ETHRACE2A"]].dropna()
model2=smf.ols(formula="NUMCIGMO_EST ~C(ETHRACE2A)", data=sub2).fit()
print(model2.summary())
m1=sub2.groupby("ETHRACE2A").mean()
print(m1)
sd1=sub2.groupby("ETHRACE2A").std()
print(sd1)

# Conducting the post hoc test :-
mc1=multi.MultiComparison(sub2["NUMCIGMO_EST"],sub2["ETHRACE2A"])
res1=mc1.tukeyhsd()
print(res1.summary())