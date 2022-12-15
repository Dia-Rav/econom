import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from stargazer.stargazer import Stargazer
import yatg
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

dataset = pd.read_excel ("base.xlsx") 

# model = smf.ols("salary ~ gender + week + exp +I(exp**2) + gender + degree + marriage + v_age + gender:child + child + C(sphere)", data=dataset)
# model_est = model.fit()
# print(model_est.summary())

plt.clf()
sb.scatterplot(dataset, x = "exp", y = "salary")
plt.show()




