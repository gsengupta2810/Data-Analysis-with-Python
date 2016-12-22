import pandas as pd
import numpy as np

df=pd.read_csv("~/datasets/gapminder.csv")

# replace a particular value from the data frame
df=df.replace(r'\s+',np.nan,regex=True) # can be replaced with anything of desire right here only

# converting the data from string to float
df["incomeperperson"]=df["incomeperperson"].convert_objects(convert_numeric=True)
df["alcconsumption"]=df["alcconsumption"].convert_objects(convert_numeric=True)
df["lifeexpectancy"]=df["lifeexpectancy"].convert_objects(convert_numeric=True)

# replace the nan values with the mean of the column 
df["incomeperperson"]=df["incomeperperson"].fillna(value=df["incomeperperson"].mean())
df["alcconsumption"]=df["alcconsumption"].fillna(value=df["alcconsumption"].mean())
df["lifeexpectancy"]=df["lifeexpectancy"].fillna(value=df["lifeexpectancy"].mean())

# print(df["country"])
# print(df["alcconsumption"])
# print(df["lifeexpectancy"])

# generating frequency distributions 
#df[].value_counts() is used to count the frequency of occurance of a particular value in a column
c1=df["alcconsumption"].value_counts(sort=True, normalize=True, dropna=False)
c2=df["lifeexpectancy"].value_counts(sort=True, dropna=False)
print(c1,c2)

#the same task can be achieved by using groupby method
ct1=df.groupby("alcconsumption").size() #*100/len(data) #for normalising it to percentages

# for selecting particular rows which satisfy some conditions, i.e. we are selecting a sample from the population :-
# for example in my research question, we can exclude the countries which have very low alcohol consumption rate or a very low per capita income, 
# because in such countries the mortality rate is dependant on many other factors.
sub1= df[(df["incomeperperson"]>=500.00) & (df["alcconsumption"]>1.00)]
# deep copy , sub2=sub1 is only shallow copy 
sub2=sub1.copy() 
