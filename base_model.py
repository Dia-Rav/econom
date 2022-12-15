import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from stargazer.stargazer import Stargazer
import yatg
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

dataset = pd.read_excel ("base.xlsx") 

model = smf.ols("salary ~ gender + week + exp +I(exp**2) + degree + marriage + v_age + child + C(sphere)", data=dataset)
model_est = model.fit()
print(model_est.summary())

model2 = smf.ols("salary ~ gender + week + exp +I(exp**2) + degree + marriage + v_age + child + gender:child + C(sphere)", data=dataset)
model2_est = model2.fit()
print(model2_est.summary())

model3 = smf.ols("salary ~ gender + week + exp +I(exp**2) + degree + marriage + v_age + child + gender:marriage + C(sphere)", data=dataset)
model3_est = model3.fit()
print(model3_est.summary())

model4 = smf.ols("salary ~ gender + week + exp +I(exp**2) + degree + marriage + v_age + child + gender:child + gender:marriage + C(sphere)", data=dataset)
model4_est = model4.fit()
print(model4_est.summary())


# plt.clf()
# sb.scatterplot(dataset, x = "child", y = "salary")
# plt.show()

# plt.clf()
# sb.scatterplot(dataset, x = "salary", y = model_est.resid**2)
# plt.show()

# print(sms.het_goldfeldquandt(y=model.endog, x=model.exog,split=.35, drop=.3))



 



