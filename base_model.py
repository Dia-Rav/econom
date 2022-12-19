import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from stargazer.stargazer import Stargazer
import yatg
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor  as vif
# Проверьте чувствительность, качество подгонки данных моделью

dataset = pd.read_excel ("base.xlsx", engine='openpyxl')
# print(dataset.head())

model = smf.ols("np.log(salary) ~ gender + exp + I(exp**2) + degree + C (sphere) + boss", data=dataset)
model_est = model.fit()
print(model_est.summary())

model = smf.ols("np.log(salary) ~ gender + age + I(age**2) + degree + C (sphere) + boss", data=dataset)
model_est = model.fit()
print(model_est.summary())


#тест Зарембки
# model = smf.ols("salary_z ~ gender + exp + I(exp**2) + degree +C (sphere) + marriage + boss", data=dataset)
# model_est = model.fit()
# # print(model_est.ssr)

# model = smf.ols("np.log(salary_z)~ gender + exp + I(exp**2) + degree +C (sphere) + marriage + boss", data=dataset)
# model_est = model.fit()
# print(model_est.ssr)

# хи = 18.0251854829263...

# тест голдфелда квандта
# print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 2, split=.3, drop=.4))
# (1.1874302473304987, 0.32516711335819504, 'increasing')
# The Null hypothesis is that the variance in the two sub-samples are the same. 
# The alternative hypothesis, can be increasing, 
# i.e. the variance in the second sample is larger than in the first, or decreasing or two-sided.
# нет гетерскедостичности 


# тест глейзера 
# from scipy.stats import chi2
# model_aux = smf.ols("abs(model_est.resid) ~ exp", data=dataset)
# model_aux_est = model_aux.fit()
# stat_aux = model_aux_est.ess / ((1 - 2 / np.pi) * np.var(model_est.resid))
# print(f"Stat: {stat_aux:5.4f}, Critical value: {chi2.ppf(0.95, df=model_aux.df_model):5.4f}, \
# p-value: {1 - chi2.cdf(stat_aux, df=model_aux.df_model):5.4f}")

# Stat: 1.2490, Critical value: 3.8415, p-value: 0.2637 - exp
# Stat: 2.0498, Critical value: 3.8415, p-value: 0.1522 - degree

# коэффециенты незначимы, гетерскедостичности нет

# plt.clf()
# sb.scatterplot(dataset, x="age", y="exp")
# plt.show()

# Тест Бройша-Пагана.
# print(sms.het_breuschpagan(model_est.resid, model.exog))
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic
# (28.92924396308413, 0.3143096710720359, 1.128323745715039, 0.3182770191868023)
# гетереск нет

# for i in range(1, model.exog.shape[1]):
#     vif_est = vif(model.exog, i)
#     print(f"{model_est.params.index[i]:8}: {vif_est:5.4f}")

# Среди эконометристов существует убеждение, что если для одной из переменных >10
# то в регрессии есть мультиколлинеарность.
# C(sphere)[T.1]: 2.1030
# C(sphere)[T.2]: 1.7444
# C(sphere)[T.3]: 1.2402
# C(sphere)[T.4]: 1.1349
# C(sphere)[T.5]: 1.1384
# C(sphere)[T.6]: 2.4153
# C(sphere)[T.7]: 3.2101
# C(sphere)[T.9]: 1.3988
# C(sphere)[T.10]: 2.1662
# C(sphere)[T.11]: 1.5255
# C(sphere)[T.12]: 1.9704
# C(sphere)[T.13]: 1.7790
# C(sphere)[T.14]: 4.0627
# C(sphere)[T.15]: 1.3342
# C(sphere)[T.16]: 1.2222
# C(sphere)[T.17]: 2.6654
# C(sphere)[T.20]: 1.1260
# C(sphere)[T.21]: 1.2724
# C(sphere)[T.26]: 1.3484
# C(sphere)[T.27]: 1.2233
# C(sphere)[T.28]: 1.2297
# gender  : 1.4289
# exp     : 13.3546
# I(exp ** 2): 12.7708
# degree  : 1.3237
# boss    : 1.2125