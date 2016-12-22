import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats
import seaborn

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


def USQUAN (row):
	if row["S3AQ3B1"] !=1:
		return 0
	elif row["S3AQ3C1"] <=5:
		return 3
	elif row["S3AQ3C1"] <=10:
		return 8
	elif row["S3AQ3C1"] <=15:
		return 13
	elif row["S3AQ3C1"] <=20:
		return 18
	elif row["S3AQ3C1"] >20:
		return 37

sub["USQUAN"]= sub.apply(lambda row: USQUAN (row), axis=1)

# contingency table of observed counts
ct1=pd.crosstab(sub["TAB12MDX"],sub["USQUAN"])
print(ct1)

# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

# the first value in the printed table is c2 and the 2nd value is the p value
cs1=scipy.stats.chi2_contingency(ct1)
print(cs1)

#################################################################################################################
                          Considering the effect of Moderator in Chi-2 test
#################################################################################################################

# considering the effects of major depression disorder :- 
sub1=sub[(sub["MAJORDEPLIFE"]==0)]
sub2=sub[(sub["MAJORDEPLIFE"]==1)]

print("assosiation between Smoking quanitity and nicotine dependence in people without Depression \n")
# contingency table of observed counts
ct1=pd.crosstab(sub["TAB12MDX"],sub1["USQUAN"])
print(ct1)

# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

# the first value in the printed table is c2 and the 2nd value is the p value
cs1=scipy.stats.chi2_contingency(ct1)
print(cs1)

print("assosiation between Smoking quanitity and nicotine dependence in people with Depression \n")
# contingency table of observed counts
ct2=pd.crosstab(sub["TAB12MDX"],sub2["USQUAN"])
print(ct2)

# column percentages
colsum=ct2.sum(axis=0)
colpct=ct2/colsum
print(colpct)

# the first value in the printed table is c2 and the 2nd value is the p value
cs2=scipy.stats.chi2_contingency(ct2)
print(cs2)

# using a line graph to examine the rates of nicotine dependence in both the groups:-
sub1["USQUAN"]=sub1["USQUAN"].astype("category")
sub1["TAB12MDX"]=sub1["TAB12MDX"].convert_objects(convert_numeric=True)
seaborn.factorplot(x="USQUAN",y="TAB12MDX", data=sub1, king="point", ci=None)
plt.xlabel("USQUAN")
plt.ylabel("TAB12MDX")
plt.title("Without depression ")
plt.show()

sub2["USQUAN"]=sub2["USQUAN"].astype("category")
sub2["TAB12MDX"]=sub2["TAB12MDX"].convert_objects(convert_numeric=True)
seaborn.factorplot(x="USQUAN",y="TAB12MDX", data=sub2, king="point", ci=None)
plt.xlabel("USQUAN")
plt.ylabel("TAB12MDX")
plt.title("With depression ")
plt.show()?

##################################################################################################################
#                           Considering the effect of Moderator on Correlation
##################################################################################################################
df=pd.read_csv("~/datasets/gapminder.csv")

df["urbanrate"]=df["urbanrate"].convert_objects(convert_numeric=True)
df["incomeperperson"]=df["incomeperperson"].convert_objects(convert_numeric=True)
df["internetuserate"]=df["internetuserate"].convert_objects(convert_numeric=True)

df["incomeperperson"]=df["incomeperperson"].replace(r"\s+",np.nan)
df_clean=df.dropna()

def incomegrp (row):
	if row["incomeperperson"]<=744.239:
		return 1
	elif row["incomeperperson"]<=9425.326:
		return 2
	elif row["incomeperperson"]>9425.326:
		return 3

df_clean["incomegrp"]= df_clean.apply(lambda row: incomegrp (row), axis=1)
df_clean=df_clean.dropna()

print(df_clean.groupby("incomegrp").size())

sub1=df_clean[(df_clean["incomegrp"]==1)] 
sub2=df_clean[(df_clean["incomegrp"]==2)] 
sub3=df_clean[(df_clean["incomegrp"]==3)] 

print("group 1 :- low income per person")
print(scipy.stats.pearsonr(sub1["urbanrate"],sub1["internetuserate"]))
print("group 2 :- middle income per person")
print(scipy.stats.pearsonr(sub2["urbanrate"],sub2["internetuserate"]))
print("group 1 :- high income per person")
print(scipy.stats.pearsonr(sub3["urbanrate"],sub3["internetuserate"]))

seaborn.regplot(x="urbanrate",y="internetuserate",fit_reg=True,data=sub1)
plt.title("Relation between internet usage and urban population of a low income country")
plt.xlabel("urbanrate")
plt.ylabel("internetuserate")
plt.show()

seaborn.regplot(x="urbanrate",y="internetuserate",fit_reg=True,data=sub2)
plt.title("Relation between internet usage and urban population of a middle income country")
plt.xlabel("urbanrate")
plt.ylabel("internetuserate")
plt.show()

seaborn.regplot(x="urbanrate",y="internetuserate",fit_reg=True,data=sub3)
plt.title("Relation between internet usage and urban population of a high income country")
plt.xlabel("urbanrate")
plt.ylabel("internetuserate")
plt.show()
