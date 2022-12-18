import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

dataset = pd.read_excel (r"D:\Уроки\3 курс\РАНХ\Эконометрика\Проект\base.xlsx")

# МОДЕЛЬ С НАЛИЧИЕМ РЕБЁНКА

model = smf.ols("np.log(salary) ~ gender + exp + I(exp**2) + degree + age + C (sphere) + child + child:gender", data=dataset)
model_est = model.fit()
print(model_est.summary())

# тест Зарембки (чтобы было на всякий случай)
model = smf.ols("salary_z ~ gender + exp + I(exp**2) + degree + age + C (sphere) +  + child + child:gender", data=dataset)
model_est = model.fit()
print(model_est.ssr)
# RSS_1 = 25.807713558771525

model = smf.ols("np.log(salary_z)~ gender + exp + I(exp**2) + degree + age +C (sphere) + child + child:gender", data=dataset)
model_est = model.fit()
print(model_est.ssr)
# RSS_2 = 20.39613511485961
# Xi = 8.12503113608

# тест голдфелда квандта
print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 7, split=.3, drop=.4))
# (0.6694885494068938, 0.8465318670460562, 'increasing') - сортировка по переменной child - вывод: гомоскедастичность
print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 8, split=.3, drop=.4))
# (0.7107453543858527, 0.8089508741527496, 'increasing') - сортировка по переменной child:gender - вывод: гомоскедастичность
 


# тест глейзера 
from scipy.stats import chi2
model_aux = smf.ols("abs(model_est.resid) ~ child", data=dataset)
model_aux_est = model_aux.fit()
stat_aux = model_aux_est.ess / ((1 - 2 / np.pi) * np.var(model_est.resid))
print(f"Stat: {stat_aux:5.4f}, Critical value: {chi2.ppf(0.95, df=model_aux.df_model):5.4f}, \
p-value: {1 - chi2.cdf(stat_aux, df=model_aux.df_model):5.4f}")

# Stat: 0.0057, Critical value: 3.8415,     p-value: 0.9396 - child
# Stat: 1.5163, Critical value: 3.8415,     p-value: 0.2182 - child:gender
# коэффециенты незначимы, гетерскедостичности нет


# Тест Бройша-Пагана.
print(sms.het_breuschpagan(model_est.resid, model.exog))
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic
# (35.297778981237656, 0.161364364621077, 1.320916410013872, 0.14926613639275754)
# # на любом уровне значимости нет гетероскедастичности

from chow_test import chow_test
print(chow_test(y_series=pd.Series(model.endog), X_series=pd.DataFrame(model.exog), last_index=1499, first_index=1500, significance=0.05))

# VIF-тест на мультиколлинеарность
for i in range(1, model.exog.shape[1]):
    vif_est = vif(model.exog, i)
    print(f"{model_est.params.index[i]:8}: {vif_est:5.4f}")

#C(sphere)[T.1]: 2.0655
#C(sphere)[T.2]: 1.7712
#C(sphere)[T.3]: 1.2395
#C(sphere)[T.4]: 1.1734
#C(sphere)[T.5]: 1.1569
#C(sphere)[T.6]: 2.3473
#C(sphere)[T.7]: 3.1299
#C(sphere)[T.9]: 1.3545
#C(sphere)[T.10]: 2.2369
#C(sphere)[T.11]: 1.4633
#C(sphere)[T.12]: 1.8998
#C(sphere)[T.13]: 1.7474
#C(sphere)[T.14]: 3.8789
#C(sphere)[T.15]: 1.3339
#C(sphere)[T.16]: 1.2421
#C(sphere)[T.17]: 2.5154
#C(sphere)[T.20]: 1.1525
#C(sphere)[T.21]: 1.2464
#C(sphere)[T.26]: 1.3478
#C(sphere)[T.27]: 1.2382
#C(sphere)[T.28]: 1.2091
#gender  : 3.9865
#exp     : 17.0881
#I(exp ** 2): 14.6682
#degree  : 1.3756
#age     : 4.4752
#child   : 3.0896
#child:gender: 5.4803
# Вывод: Всё ок
