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


# критерий Акаике
# Таким образом, критерий не только вознаграждает за качество приближения, 
# но и штрафует за использование излишнего количества параметров модели. 
# Считается, что наилучшей будет модель с наименьшим значением критерия AIC.
# print(model_est.aic)
# print(model2_est.aic)
# print(model3_est.aic)
# print(model4_est.aic)
# 3631.2297869129725
# 3628.1231509833747
# 3626.4554268413963
# 3619.9477461035076


# информационный критерий шварца 
# print(model_est.bic)
# print(model2_est.bic)
# print(model3_est.bic)
# print(model4_est.bic)
# 3724.7696012854753
# 3724.780959168294
# 3723.1132350263156
# 3719.723548100844
# models with lower BIC are generally preferred


# print(sms.het_goldfeldquandt(y=model.endog, x=model.exog,split=.35, drop=.3))



# Тест Бройша-Пагана
# print(sms.het_breuschpagan(model_est.resid, model.exog))
# print(sms.het_breuschpagan(model2_est.resid, model2.exog))
# print(sms.het_breuschpagan(model3_est.resid, model3.exog))
# print(sms.het_breuschpagan(model4_est.resid, model4.exog))

# (40.46260269551623, 0.07665277838473605, 1.5106278480053188, 0.0609003187276317)   
# (42.85443167811152, 0.060349186001638896, 1.5648840811113025, 0.044745809247933456)
# (37.89817204378537, 0.152381559879282, 1.3307716034567847, 0.13811430776820818)    
# (41.34440629532407, 0.10142973690863587, 1.432870719521403, 0.08406713932862535)
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic

# Тест Уайта.
# print(sms.het_white(model_est.resid, model.exog))
# print(sms.het_white(model2_est.resid, model2.exog))
# print(sms.het_white(model3_est.resid, model3.exog))
# print(sms.het_white(model4_est.resid, model4.exog))
# (147.5031905491087, 0.2361029144651318, 1.6688611839509147, 0.05149282909355532)
# (151.6645146225534, 0.3355907412958376, 1.4323123804303264, 0.17024585333421094)
# (153.77410104351742, 0.3343585088835086, 1.502776041602281, 0.15221652670659727)
# (158.53002955568007, 0.38455452002003004, 1.4584453862304925, 0.23654095055772012)
# lm: float The lagrange multiplier statistic. lm_pvalue :float The p-value of lagrange multiplier test.
# fvalue: float The f-statistic of the hypothesis that the error variance does not depend on x. This is an alternative test variant not the original LM test.
# f_pvalue: float The p-value for the f-statistic.

plt.clf()
sb.scatterplot(dataset, x="v_age", y="salary")
sb.lmplot(dataset, x="v_age", y="salary", ci=None)
plt.show()
