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

df=pd.read_csv("~/datasets/gapminder.csv")

df["urbanrate"]=pd.to_numeric(df["urbanrate"],errors="coerce")
df["femaleemployrate"]=pd.to_numeric(df["femaleemployrate"],errors="coerce")
df["internetuserate"]=pd.to_numeric(df["internetuserate"],errors="coerce")

sub=df[["femaleemployrate","urbanrate","internetuserate"]].dropna()

# regular linear plot
seaborn.regplot(y="femaleemployrate",x="urbanrate", fit_reg=True, data=sub)
plt.xlabel("urbanrate")
plt.ylabel("femaleemployrate")


# quardratic plot 
seaborn.regplot(y="femaleemployrate",x="urbanrate", order=2, fit_reg=True, data=sub)
plt.xlabel("urbanrate")
plt.ylabel("femaleemployrate")
plt.show()

# testing the model with and without polynomial term
# first centering the urbanrate:- 
sub["urbanrate_c"]=(sub["urbanrate"]-sub["urbanrate"].mean())
sub["internetuserate_c"]=(sub["internetuserate"]-sub["internetuserate"].mean())

print("######################## without quardratic urbanrate  ######################")
reg1=smf.ols('femaleemployrate ~ urbanrate_c',data=sub).fit()
print(reg1.summary())
# we can see from the significant p value that urbanrate is negatively assisiated with the female employment rate
# but the r2 value is very small indicating that the linear assosiation is very weak in capturing the variability 

print("######################## with quardratic urbanrate  ######################")
# I is the identity in patsi which returns the value of whatever is in the paranthesis 
reg2=smf.ols('femaleemployrate ~ urbanrate_c + I(urbanrate_c**2)',data=sub).fit()
print(reg2.summary())

# Adding internet userrate to our model
sub["internetuserate_c"]=(sub["internetuserate"]-sub["internetuserate"].mean())
reg3= smf.ols('femaleemployrate ~ urbanrate_c + I(urbanrate_c**2) + internetuserate_c',data=sub).fit()
print(reg3.summary())

#####################################################
#				 generating the qq plot
#####################################################

# line =r generates a red linear regression line on the model
fig1=sm.qqplot(reg3.resid, line='r')
plt.show()

# plotting the standardised residuals against the multiples of Standard deviation about the mean 
stdres=pd.DataFrame(reg3.resid_pearson)
stdres=stdres.dropna()
fig2=plt.plot(stdres, 'o', ls='None')
l=plt.axhline(y=0,color='r')
plt.xlabel("standardised residuals")
plt.ylabel("Observation Number")
plt.show()

#  In order to improve the fit of this model we have to include more explainatory variables.
# we need to examine how some other specific explainatory variables contribute to the fit of our model.
fig3=plt.figure()
fig3.set_size_inches(15,10)
fig3=sm.graphics.plot_regress_exog(reg3, "internetuserate_c", fig=fig3)
plt.show() 
# It can be seen that the plot of residual against internet user rate, the residuals first decrease to a significant value and then again increase
# This shows that the model fails to predict properly for low and high values of internet userrate .
# In the partial regression residual plot, it can be seen that the residuals do not show any particular non-linear pattern.
# Also a lot many points are lying away from the line suggesting a high value of prediction error. 
# So incontrast to the statistical support in favor of assosiation of internet user rate to female employment, the plots suggest that this assosiation
# is pretty weak after controlling for the urbanisation rate .

##################################################
#  				Leverage Plot
##################################################
fig4=sm.graphics.influence_plot(reg3,size=8)
plt.show()