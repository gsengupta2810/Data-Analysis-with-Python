import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats
import seaborn
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

# trying to find out the relation between oil consumption and urban rate 
# considering income per person as the confounder variable we'll try to access its role in our research question 
df=pd.read_csv("~/datasets/gapminder.csv")

df["incomeperperson"]=pd.to_numeric(df["incomeperperson"],errors="coerce")
df["oilperperson"]=pd.to_numeric(df["oilperperson"],errors="coerce")
df["urbanrate"]=pd.to_numeric(df["urbanrate"],errors="coerce")
df["employrate"]=pd.to_numeric(df["employrate"],errors="coerce")
df["relectricperperson"]=pd.to_numeric(df["relectricperperson"],errors="coerce")

# creating a subgroup of the data set of only the countries which are not majorly poor 
sub=df[(df["incomeperperson"]>=200)]

# analysing the type of data we have in our hand 
# all of the three are quantitative variables, so we need to do a correlation test between our explainatory and response variable 

# plotting the scatter plot of the data
seaborn.regplot(x="urbanrate",y="oilperperson",fit_reg=True,data=sub)
plt.title("urbanrate vs oilperperson")
plt.xlabel("urbanrate")
plt.ylabel("oilperperson")
plt.show()

# getting the statistics of the data using the ols function:- 
print("\n#################################################################")
reg1= smf.ols('oilperperson ~ urbanrate',data=sub).fit()
print(reg1.summary())

# now getting the dependence of oilperperson on incomeperperson
seaborn.regplot(x="incomeperperson",y="oilperperson",fit_reg=True,data=sub)
plt.title("incomeperperson vs oilperperson")
plt.xlabel("incomeperperson")
plt.ylabel("oilperperson")
plt.show()

reg2= smf.ols('oilperperson ~ incomeperperson',data=sub).fit()
print(reg2.summary())

# using both together 

print("\n#################################################################")
reg1= smf.ols('oilperperson ~ urbanrate + incomeperperson',data=sub).fit()
print(reg1.summary())

# The p values are below the level of alpha for both the cases and the t values are positive. 
# This indicates that both urbanrate and incomeperperson are sigificantly positively assosiated with the oil per person 
# So we have to control out the incomeperperson in order to get the pure effect of urbanrate on oil per person
# Now if the p-value of the explainatory variable would have risen above alpha, then we would have said that incomeperperson confounds 
# relationship between urbanrate and oil per person

# we can go adding variables to check for counfounders 
print("\n#################################################################")
reg1= smf.ols('oilperperson ~ urbanrate + incomeperperson + employrate + relectricperperson',data=sub).fit()
print(reg1.summary())
# it can be seen that on including the effects of electricity consumption and employment rate the p-value of income per person became insignificant, proving it to be a confounder variable 
