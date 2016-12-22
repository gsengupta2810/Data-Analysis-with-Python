import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("~/datasets/gapminder.csv")
df=df.replace(r'\s+',np.nan,regex=True) 

df["incomeperperson"]=df["incomeperperson"].convert_objects(convert_numeric=True)
df["alcconsumption"]=df["alcconsumption"].convert_objects(convert_numeric=True)
df["lifeexpectancy"]=df["lifeexpectancy"].convert_objects(convert_numeric=True)
df["employrate"]=df["employrate"].convert_objects(convert_numeric=True)
df["suicideper100th"]=df["suicideper100th"].convert_objects(convert_numeric=True)

# creating a smaple from the dataset
# sub1= df[(df["incomeperperson"]>=500.00) & (df["alcconsumption"]>1.00)]
sub1= df[(df["alcconsumption"]>1.00)]
sub2=sub1.copy()

c1=df["alcconsumption"].value_counts(sort=True) #be default dropna=True
# print(c1)

sub2=sub2.replace(r'\s+',np.nan,regex=True)

c1=df["alcconsumption"].value_counts(sort=True,dropna=False)
# print(c1)

# recoding particular values in a variable or changing the mapping is sometimes necessary, in such cases use df[].map(recode)
# recode={1: 6, 2: 5,3: 4,4: 3,5: 2,2: 1}
# sub2["USFREQ"]=sub2["SQEREQ1"].map(recode)
sub2["unemployrate"]=100-sub2["employrate"]
sub2["suicideper100"]=sub2["suicideper100th"]/1000

# creating a secondary variable
# A variable derrived from a combination of two or more primary variables, for example- 
# sub2["probSuicide|unemployed"]=sub["unemployrate intersection drinkrate"]/sub2["unemployrate"]

sub3=sub2[["country","incomeperperson","alcconsumption","unemployrate","employrate","suicideper100"]]
print(sub3.head(20))

# rather that treating alcohol consumption quantitatively, if we want to treat it catagorically, 
# i decided that there will be 3 groups and then label those groups
sub3["alcconsumpCatagory1"]=pd.qcut(sub3.alcconsumption,3,labels=["low","high","very_high"])
c1=sub3["alcconsumpCatagory1"].value_counts(sort=True,dropna=True)
print(c1,sub3[["country","alcconsumpCatagory1"]].head(20))

# we can also create customised splits using pandas cut function
sub3["alcconsumpCatagory2"]=pd.cut(sub3.alcconsumption,[1,5,10,20])
c1=sub3["alcconsumpCatagory2"].value_counts(sort=True,dropna=True)
print(c1,sub3[["country","alcconsumpCatagory2"]].head(20))

# Crosstabs function allows us to cross two variables with one another..
print(pd.crosstab(sub3["alcconsumpCatagory1"],sub3["alcconsumpCatagory2"])) 
