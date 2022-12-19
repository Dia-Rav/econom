import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

dataset = pd.read_excel (r"D:\Уроки\3 курс\РАНХ\Эконометрика\Проект\base.xlsx")



# МОДЕЛЬ С НАЛИЧИЕМ РЕБЁНКА

model = smf.ols("np.log(salary) ~ gender + exp + I(exp**2) + boss + degree + C (sphere) + child + child:gender", data=dataset)
model_est = model.fit()
print(model_est.summary())

 


# тест глейзера 
from scipy.stats import chi2
model_aux = smf.ols("abs(model_est.resid) ~ child", data=dataset)
model_aux_est = model_aux.fit()
stat_aux = model_aux_est.ess / ((1 - 2 / np.pi) * np.var(model_est.resid))
print(f"Stat: {stat_aux:5.4f}, Critical value: {chi2.ppf(0.95, df=model_aux.df_model):5.4f}, \
p-value: {1 - chi2.cdf(stat_aux, df=model_aux.df_model):5.4f}")

# Stat: 0.2075, Critical value: 3.8415,     p-value: 0.6487 - child
# Stat: 1.0760, Critical value: 3.8415,     p-value: 0.2996 - child:gender
# коэффециенты незначимы, гетерскедостичности нет


# Тест Бройша-Пагана.
print(sms.het_breuschpagan(model_est.resid, model.exog))
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic
# (30.29550309456054, 0.349232933376534, 1.0923113978747574, 0.3566280866903009)
# на любом уровне значимости нет гетероскедастичности

from chow_test import chow_test
print(chow_test(y_series=pd.Series(model.endog), X_series=pd.DataFrame(model.exog), last_index=1499, first_index=1500, significance=0.05))

# VIF-тест на мультиколлинеарность
for i in range(1, model.exog.shape[1]):
    vif_est = vif(model.exog, i)
    print(f"{model_est.params.index[i]:8}: {vif_est:5.4f}")

#C(sphere)[T.1]: 2.1220
#C(sphere)[T.2]: 1.7738
#C(sphere)[T.3]: 1.2470
#C(sphere)[T.4]: 1.1830
#C(sphere)[T.5]: 1.1442
#C(sphere)[T.6]: 2.4695
#C(sphere)[T.7]: 3.2800
#C(sphere)[T.9]: 1.4081
#C(sphere)[T.10]: 2.1673
#C(sphere)[T.11]: 1.5683
#C(sphere)[T.12]: 1.9726
#C(sphere)[T.13]: 1.8194
#C(sphere)[T.14]: 4.1540
#C(sphere)[T.15]: 1.3609
#C(sphere)[T.16]: 1.2525
#C(sphere)[T.17]: 2.6799
#C(sphere)[T.20]: 1.1329
#C(sphere)[T.21]: 1.2844
#C(sphere)[T.26]: 1.3564
#C(sphere)[T.27]: 1.2291
#C(sphere)[T.28]: 1.2303
#gender  : 3.9667
#exp     : 16.2871
#I(exp ** 2): 14.2631
#boss    : 1.2174
#degree  : 1.3670
#child   : 2.8280
#child:gender: 5.4309
# Вывод: Всё ок
