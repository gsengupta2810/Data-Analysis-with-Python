import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats
import seaborn

df=pd.read_csv("~/datasets/gapminder.csv")
df["urbanrate"]=df["urbanrate"].convert_objects(convert_numeric=True)
df["internetuserate"]=df["internetuserate"].convert_objects(convert_numeric=True)
df["incomeperperson"]=df["incomeperperson"].convert_objects(convert_numeric=True)

sub4=df[["urbanrate","internetuserate","incomeperperson"]]
# correlation coefficient cannot be computed in the presence of NAs 
sub4=sub4.dropna()
desc1=sub4["urbanrate"].describe()
desc2=sub4["internetuserate"].describe()

# The first value is r and the 2nd value is p-value
print(scipy.stats.pearsonr(sub4["urbanrate"],sub4["internetuserate"]))
print(scipy.stats.pearsonr(sub4["incomeperperson"],sub4["internetuserate"]))

# fit_reg=False for not plotting the line of best fit, make it True to show the line of best fit
seaborn.regplot(x="urbanrate",y="internetuserate",fit_reg=True,data=sub4)
plt.title("Relation between internet usage and urban population of a country")
plt.xlabel("urbanrate")
plt.ylabel("internetuserate")
plt.show()

seaborn.regplot(x="incomeperperson",y="internetuserate",fit_reg=True,data=sub4)
plt.title("Relation between internet usage and urban population of a country")
plt.xlabel("incomeperperson")
plt.ylabel("internetuserate")
plt.show()
