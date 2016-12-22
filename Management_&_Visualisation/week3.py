import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn

df=pd.read_csv("~/datasets/gapminder.csv")
df=df.replace(r'\s+',np.nan,regex=True) 

# Thiese instruct pandas to display maximum number of rows and columns that can be displayed
# pd.set_option('display.max_comlumns',None)
# pd.set_option('display.max_rows',None)

df["incomeperperson"]=df["incomeperperson"].convert_objects(convert_numeric=True)
df["alcconsumption"]=df["alcconsumption"].convert_objects(convert_numeric=True)
df["lifeexpectancy"]=df["lifeexpectancy"].convert_objects(convert_numeric=True)
df["employrate"]=df["employrate"].convert_objects(convert_numeric=True)
df["suicideper100th"]=df["suicideper100th"].convert_objects(convert_numeric=True)

sub1= df[(df["alcconsumption"]>1.00)]
sub2=sub1.copy()

sub2["unemployrate"]=100-sub2["employrate"]
sub2["suicideper100"]=sub2["suicideper100th"]/1000

sub3=sub2[["country","incomeperperson","alcconsumption","unemployrate","employrate","suicideper100"]]
# print(sub3.head(20))

# we can create customised splits using pandas cut function
sub3["alcconsumpCatagory2"]=pd.cut(sub3.alcconsumption,[1,3,5,7,9,12,15,18,20,22])
c1=sub3["alcconsumpCatagory2"].value_counts(sort=True,dropna=True)

# astype function is used to convert to a catagorical variable for plotting 
sub3["alcconsumpCatagory2"]=sub3["alcconsumpCatagory2"].astype('category')

# this is to plot a catagorical variable 
seaborn.countplot(x="alcconsumpCatagory2",data=sub3)
plt.xlabel("Alcohol consumption")
plt.ylabel("Number of Countries")
plt.show()

# to display a quantitative variable use distplot
# this generates a histogram
seaborn.distplot(sub3["unemployrate"].dropna(),kde=False)
plt.xlabel("Unemployment rate of countries")
plt.ylabel("freq")
plt.show()

# describe function is used to generate the statistical values of various statistical variables
desc1=sub3["unemployrate"].dropna().describe()
print("description of the unemployrate ---------------\n",desc1)


# **************************** Converting alcohol consumption to catagorical variable ******************************

# since generally one bottle of liquer contains 0.75 liters 
sub3["bottlesConsumed"]=sub3["alcconsumption"]/0.75
c1=sub3.groupby("bottlesConsumed").size() 
print("Number of bottles consumed per capita",c1)

seaborn.distplot(sub3["bottlesConsumed"].dropna(),kde=False)
plt.xlabel("Bottles Consumed per capita")
plt.ylabel("Number of countries")
plt.show()

desc1=sub3["bottlesConsumed"].dropna().describe()
print(desc1)

# bottles per month is still a quantitative variable
# But based on this I can divide it in catagories like <2, 2-5, 5-8, 8-10, 10-15, >15
sub3["bottleCatagory"]=pd.cut(sub3.bottlesConsumed,[0,1,2,5,8,10,15,20,25,31])
# subdivind them into two catagories for high and low 
sub3["bottleCatagory"]=sub3["bottleCatagory"].astype('category')

seaborn.countplot(x="bottleCatagory",data=sub3)
plt.xlabel("Bottles Consumed per capita")
plt.ylabel("Number of countries")
plt.show()

# trying to remap these catagories into two catagories 
recode={"(0, 1]": 0,"(1, 2]": 0,"(2, 5]": 0,"(5, 8]": 0,"(8, 10]": 0,"(10, 15]": 1,"(15, 20]": 1,"(20, 25]": 1,"(25, 31]": 1}
sub3["bottleCatagory"]=sub3["bottleCatagory"].map(recode)

# just trying to show the use of a lambda function:
def high_low(row):
	if row['bottleCatagory']==0:
		return "low consumption"
	else:
		return "high consumption"
sub3['alcConsumpType']=sub3.apply(lambda row: high_low (row), axis=1)
print(sub3["alcConsumpType"].value_counts(sort=False))
sub3["alcConsumpType"]=sub3["alcConsumpType"].astype('category')

# renaming the catagorical variables
sub3["alcConsumpType"]=sub3["alcConsumpType"].cat.rename_categories(["High","Low"])
seaborn.countplot(x="alcConsumpType",data=sub3)
plt.xlabel("Alcohol per capita")
plt.ylabel("Number of countries")
plt.show()

# ******************************************************************************************************************

# ***************************** Converting unemployment rate to catagorical variable *******************************

sub3["unempCatagory"]=pd.cut(sub3.unemployrate,[0,10,20,30,40,50,60,70,80,90,100])
sub3["unempCatagory"]=sub3["unempCatagory"].astype('category')
print(sub3["unempCatagory"].value_counts(sort=True))

seaborn.countplot(x="unempCatagory",data=sub3)
plt.xlabel("Unemployment rate catagories")
plt.ylabel("Number of countries")
plt.show()
# ******************************************************************************************************************

# To plot a c->c plot we use seaborn.factorplot()

# we need to set the catagoric response variable back to numeric
sub3['bottleCatagory']=sub3['bottleCatagory'].convert_objects(convert_numeric=True)
# print(sub3['bottleCatagory'].apply(pd.to_numeric, errors='ignore'))  
print("after to_numeric \n",sub3['bottleCatagory'].head(20))

seaborn.factorplot(x='unempCatagory',y='bottleCatagory',data=sub3,kind="bar",ci=None)
plt.xlabel("bottleCatagory")
plt.ylabel("unempCatagory")
plt.show()

# ************************************************** Quantitative to Quantitative scatter plot*******************************************************************

# When the response variable is quatitative
# for the plot of urban rate vs internet usage, both response and explainatory variables are quantitative
# A bar chart will not wok here, we require a scatter plot !! 
df["urbanrate"]=df["urbanrate"].convert_objects(convert_numeric=True)
df["internetuserate"]=df["internetuserate"].convert_objects(convert_numeric=True)

sub4=df[["urbanrate","internetuserate"]]

desc1=sub4["urbanrate"].describe()
desc2=sub4["internetuserate"].describe()
print(desc1,desc2)

# fit_reg=False for not plotting the line of best fit, make it True to show the line of best fit
seaborn.regplot(x="urbanrate",y="internetuserate",fit_reg=True,data=sub4)
plt.title("Relation between internet usage and urban population of a country")
plt.xlabel("urbanrate")
plt.ylabel("internetuserate")
plt.show()

# ****************************************Catagorical to quantitative bar chart *****************************************
df["hivrate"]=df["hivrate"].convert_objects(convert_numeric=True)
sub5=df[["hivrate","incomeperperson"]]
# dividing income per person into 4 catagories 
sub5["incomeCatagory"]=pd.qcut(sub5.incomeperperson,4,labels=["1-25th%tile","25-50th%tile","50-75th%tile","75-100th%tile"])
c1=sub5["incomeCatagory"].value_counts(sort=False,dropna=True)
print(c1)

seaborn.factorplot(x="incomeCatagory",y="hivrate",data=sub5,kind="bar",ci=None)
plt.xlabel("incomeCatagory")
plt.ylabel("hiv rate")
plt.show()

