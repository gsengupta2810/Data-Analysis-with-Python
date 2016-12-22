import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn

df=pd.read_csv("~/datasets/gapminder.csv")
df["internetuserate"]=df["internetuserate"].convert_objects(convert_numeric=True)
df["urbanrate"]=df["urbanrate"].convert_objects(convert_numeric=True)
seaborn.regplot(x="urbanrate",y="internetuserate",fit_reg=False,data=df)
plt.xlabel("urbanrate")
plt.ylabel("interenuserate")
plt.show()