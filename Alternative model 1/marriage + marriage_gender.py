import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

dataset = pd.read_excel (r"D:\Уроки\3 курс\РАНХ\Эконометрика\Проект\base.xlsx")

# МОДЕЛЬ С НАЛИЧИЕМ БРАКА И ДЕТЕЙ

model = smf.ols("np.log(salary_z)~ gender + exp + I(exp**2) + boss + degree + C (sphere)  + marriage + marriage:gender", data=dataset)
model_est = model.fit()
print(model_est.summary())

 


# тест глейзера 
from scipy.stats import chi2
model_aux = smf.ols("abs(model_est.resid) ~ marriage", data=dataset)
model_aux_est = model_aux.fit()
stat_aux = model_aux_est.ess / ((1 - 2 / np.pi) * np.var(model_est.resid))
print(f"Stat: {stat_aux:5.4f}, Critical value: {chi2.ppf(0.95, df=model_aux.df_model):5.4f}, \
p-value: {1 - chi2.cdf(stat_aux, df=model_aux.df_model):5.4f}")

# Stat: 0.2061, Critical value: 3.8415,     p-value: 0.6498 - marriage
# Stat: 0.4839, Critical value: 3.8415,     p-value: 0.4867 - marriage:gender
# коэффециенты незначимы, гетерскедостичности нет


# Тест Бройша-Пагана.
print(sms.het_breuschpagan(model_est.resid, model.exog))
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic
# (25.540950728180956, 0.5982612725905296, 0.8897128654480757, 0.6281472445356965)
# на любом уровне значимости нет гетероскедастичности

from chow_test import chow_test
print(chow_test(y_series=pd.Series(model.endog), X_series=pd.DataFrame(model.exog), last_index=1499, first_index=1500, significance=0.05))

# VIF-тест на мультиколлинеарность
for i in range(1, model.exog.shape[1]):
    vif_est = vif(model.exog, i)
    print(f"{model_est.params.index[i]:8}: {vif_est:5.4f}")

#C(sphere)[T.1]: 2.1058
#C(sphere)[T.2]: 1.7704
#C(sphere)[T.3]: 1.2422
#C(sphere)[T.4]: 1.1387
#C(sphere)[T.5]: 1.1421
#C(sphere)[T.6]: 2.4179
#C(sphere)[T.7]: 3.2393
#C(sphere)[T.9]: 1.4530
#C(sphere)[T.10]: 2.1976
#C(sphere)[T.11]: 1.5333
#C(sphere)[T.12]: 1.9883
#C(sphere)[T.13]: 1.7847
#C(sphere)[T.14]: 4.1057
#C(sphere)[T.15]: 1.3460
#C(sphere)[T.16]: 1.2618
#C(sphere)[T.17]: 2.6797
#C(sphere)[T.20]: 1.1497
#C(sphere)[T.21]: 1.2758
#C(sphere)[T.26]: 1.3718
#C(sphere)[T.27]: 1.2364
#C(sphere)[T.28]: 1.2352
#gender  : 3.8925
#exp     : 13.5081
#I(exp ** 2): 13.0557
#boss    : 1.2308
#degree  : 1.3299
#marriage: 2.9669
#marriage:gender: 4.8580
# Вывод: Всё ок
