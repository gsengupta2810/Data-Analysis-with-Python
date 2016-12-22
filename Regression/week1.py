import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi


#########################################################################################
#                      			 For Q->Q data 
#########################################################################################
df=pd.read_csv("~/datasets/gapminder.csv")

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

# estimating the parameters of the best fit line 
reg1= smf.ols('urbanrate ~ internetuserate',data=sub4).fit()
print(reg1.summary())

#########################################################################################
#                      			 For C->C data 
#########################################################################################

# read the csv
# convert to numeric 
# use ols to find the fit()
# this time instead of a scatter plot there will be a bar chart 
# plot using seaborn factorplot function 
