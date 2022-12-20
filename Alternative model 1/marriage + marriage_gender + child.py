import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

dataset = pd.read_excel (r"D:\Уроки\3 курс\РАНХ\Эконометрика\Проект\base.xlsx")


# тестирование корреляции
from scipy.stats.stats import pearsonr
print(pearsonr(dataset['gender'], dataset['marriage']))
#(-0.16754183924129185, 0.03045077914551489) - корреляция значима на 5% уровне - добавляем
print(pearsonr(dataset['gender'], dataset['child']))
#(0.13319592310892073, 0.0861626202810414) - корреляция значима на 10% уровне - добавляем



# МОДЕЛЬ С НАЛИЧИЕМ БРАКА И ДЕТЕЙ

model = smf.ols("np.log(salary) ~ gender + exp + I(exp**2) + boss + degree + C (sphere) + marriage + marriage:gender + child + child:gender", data=dataset)
model_est = model.fit()
print(model_est.summary())

# тест Зарембки (чтобы было на всякий случай)
model = smf.ols("salary_z ~ gender + exp + I(exp**2) + degree + age + C (sphere)  + marriage + marriage:gender + child + child:gender", data=dataset)
model_est = model.fit()
print(model_est.ssr)
# RSS_1 = 24.478256610217617

model = smf.ols("np.log(salary_z)~ gender + exp + I(exp**2) + degree + age + C (sphere)  + marriage + marriage:gender + child + child:gender", data=dataset)
model_est = model.fit()
print(model_est.ssr)
# RSS_2 = 19.36324304422063
# Xi = 8.09328973312
 


# тест глейзера 
from scipy.stats import chi2
model_aux = smf.ols("abs(model_est.resid) ~ marriage", data=dataset)
model_aux_est = model_aux.fit()
stat_aux = model_aux_est.ess / ((1 - 2 / np.pi) * np.var(model_est.resid))
print(f"Stat: {stat_aux:5.4f}, Critical value: {chi2.ppf(0.95, df=model_aux.df_model):5.4f}, \
p-value: {1 - chi2.cdf(stat_aux, df=model_aux.df_model):5.4f}")

# Stat: 0.1566, Critical value: 3.8415,     p-value: 0.6923 - child
# Stat: 0.3046, Critical value: 3.8415,     p-value: 0.5810 - marriage
# Stat: 0.5100, Critical value: 3.8415,     p-value: 0.4752 - marriage:gender
# коэффециенты незначимы, гетерскедостичности нет


# Тест Бройша-Пагана.
print(sms.het_breuschpagan(model_est.resid, model.exog))
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic
# (26.90798139602361, 0.5766698479159604, 0.9072350477014984, 0.6060382897627022)
# на любом уровне значимости нет гетероскедастичности

from chow_test import chow_test
print(chow_test(y_series=pd.Series(model.endog), X_series=pd.DataFrame(model.exog), last_index=1499, first_index=1500, significance=0.05))

# VIF-тест на мультиколлинеарность
for i in range(1, model.exog.shape[1]):
    vif_est = vif(model.exog, i)
    print(f"{model_est.params.index[i]:8}: {vif_est:5.4f}")

#C(sphere)[T.1]: 2.1232
#C(sphere)[T.2]: 1.7705
#C(sphere)[T.3]: 1.2451
#C(sphere)[T.4]: 1.1807
#C(sphere)[T.5]: 1.1421
#C(sphere)[T.6]: 2.4690
#C(sphere)[T.7]: 3.2830
#C(sphere)[T.9]: 1.4576
#C(sphere)[T.10]: 2.1986
#C(sphere)[T.11]: 1.5747
#C(sphere)[T.12]: 1.9883
#C(sphere)[T.13]: 1.8170
#C(sphere)[T.14]: 4.1770
#C(sphere)[T.15]: 1.3631
#C(sphere)[T.16]: 1.2755
#C(sphere)[T.17]: 2.6904
#C(sphere)[T.20]: 1.1531
#C(sphere)[T.21]: 1.2826
#C(sphere)[T.26]: 1.3721
#C(sphere)[T.27]: 1.2412
#C(sphere)[T.28]: 1.2358
#gender  : 3.9778
#exp     : 16.4580
#I(exp ** 2): 14.5585
#boss    : 1.2325
#degree  : 1.3575
#marriage: 3.0382
#marriage:gender: 4.9286
#child   : 1.7163
# Вывод: Всё ок
