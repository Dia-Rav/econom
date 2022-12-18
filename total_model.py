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

model = smf.ols("np.log(salary) ~ gender + exp + I(exp**2) + degree + age + C (sphere) + marriage + marriage:gender + child + child:gender\
+ ", data=dataset)
model_est = model.fit()
print(model_est.summary())

# тест голдфелда квандта
# print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 2, split=.3, drop=.4))
# print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 5, split=.3, drop=.4))
# (1.0592754435111993, 0.43999116334161426, 'increasing')
# (1.210007032674096, 0.30258478786068405, 'increasing')
# The Null hypothesis is that the variance in the two sub-samples are the same. 
# The alternative hypothesis, can be increasing, 
# i.e. the variance in the second sample is larger than in the first, or decreasing or two-sided.
# нет гетерскедостичности 


# тест глейзера 
# from scipy.stats import chi2
# model_aux = smf.ols("abs(model_est.resid) ~ degree", data=dataset)
# model_aux_est = model_aux.fit()
# stat_aux = model_aux_est.ess / ((1 - 2 / np.pi) * np.var(model_est.resid))
# print(f"Stat: {stat_aux:5.4f}, Critical value: {chi2.ppf(0.95, df=model_aux.df_model):5.4f}, \
# p-value: {1 - chi2.cdf(stat_aux, df=model_aux.df_model):5.4f}")

# Stat: 0.1068, Critical value: 3.8415,     p-value: 0.7438 - age
# Stat: 0.6419, Critical value: 3.8415,     p-value: 0.4230 - gender
# Stat: 0.9860, Critical value: 3.8415,     p-value: 0.3207 - Exception
# Stat: 0.6175, Critical value: 3.8415,     p-value: 0.4320 - degree
# коэффециенты незначимы, гетерскедостичности нет

plt.clf()
sb.scatterplot(dataset, x="salary", y=model_est.resid ** 2)
plt.show()

# Тест Бройша-Пагана.
# print(sms.het_breuschpagan(model_est.resid, model.exog))
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic
# (36.40091234383967, 0.08453871312506729, 1.5008137969287692, 0.07073892887533434)
# на 10 процентном уровне значимости гетереск нет

# for i in range(1, model.exog.shape[1]):
#     vif_est = vif(model.exog, i)
#     print(f"{model_est.params.index[i]:8}: {vif_est:5.4f}")

# Среди эконометристов существует убеждение, что если для одной из переменных >10
# то в регрессии есть мультиколлинеарность.
# C(sphere)[T.1]: 2.0648
# C(sphere)[T.2]: 1.7242
# C(sphere)[T.3]: 1.2364
# C(sphere)[T.4]: 1.1382
# C(sphere)[T.5]: 1.1468
# C(sphere)[T.6]: 2.3207
# C(sphere)[T.7]: 3.0952
# C(sphere)[T.9]: 1.3505
# C(sphere)[T.10]: 2.2264
# C(sphere)[T.11]: 1.4432
# C(sphere)[T.12]: 1.8827
# C(sphere)[T.13]: 1.7281
# C(sphere)[T.14]: 3.8372
# C(sphere)[T.15]: 1.3228
# C(sphere)[T.16]: 1.2264
# C(sphere)[T.17]: 2.5153
# C(sphere)[T.20]: 1.1346
# C(sphere)[T.21]: 1.2305
# C(sphere)[T.26]: 1.3353
# C(sphere)[T.27]: 1.2199
# C(sphere)[T.28]: 1.2080
# gender  : 1.4377
# exp     : 15.5426
# I(exp ** 2): 12.6149
# degree  : 1.3316
# age     : 3.8039