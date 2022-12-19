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

model = smf.ols("np.log(salary) ~ gender + exp + I(exp**2) + degree + age + C (sphere) + marriage + marriage:gender + child + child:gender", data=dataset)
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

# тест голдфелда квандта
print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 7, split=.3, drop=.4))
# (0.6178764084016665, 0.8801700534302372, 'increasing') - сортировка по переменной marriage - вывод: гомоскедастичность
print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 8, split=.3, drop=.4))
# (0.6304471816561239, 0.8715057033434903, 'increasing') - сортировка по переменной marriage:gender - вывод: гомоскедастичность
print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 9, split=.3, drop=.4))
# (0.6995635692948423, 0.811381994018188, 'increasing') - сортировка по переменной child - вывод: гомоскедастичность
print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 10, split=.3, drop=.4))
# (0.9098054982131474, 0.5922201442332113, 'increasing') - сортировка по переменной child:gender - вывод: гомоскедастичность
 


# тест глейзера 
from scipy.stats import chi2
model_aux = smf.ols("abs(model_est.resid) ~ marriage", data=dataset)
model_aux_est = model_aux.fit()
stat_aux = model_aux_est.ess / ((1 - 2 / np.pi) * np.var(model_est.resid))
print(f"Stat: {stat_aux:5.4f}, Critical value: {chi2.ppf(0.95, df=model_aux.df_model):5.4f}, \
p-value: {1 - chi2.cdf(stat_aux, df=model_aux.df_model):5.4f}")

# Stat: 0.1276, Critical value: 3.8415,     p-value: 0.7209 - child
# Stat: 2.6865, Critical value: 3.8415,     p-value: 0.1012 - child:gender
# Stat: 0.2176, Critical value: 3.8415,     p-value: 0.6409 - marriage
# Stat: 1.2175, Critical value: 3.8415,     p-value: 0.2699 - marriage:gender
# коэффециенты незначимы, гетерскедостичности нет


# Тест Бройша-Пагана.
print(sms.het_breuschpagan(model_est.resid, model.exog))
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic
# (31.648481652745886, 0.3840520571202679, 1.0600037485925404, 0.39526979473235924)
# на 10 процентном уровне значимости есть гетероскедастичность

from chow_test import chow_test
print(chow_test(y_series=pd.Series(model.endog), X_series=pd.DataFrame(model.exog), last_index=1499, first_index=1500, significance=0.05))

# VIF-тест на мультиколлинеарность
for i in range(1, model.exog.shape[1]):
    vif_est = vif(model.exog, i)
    print(f"{model_est.params.index[i]:8}: {vif_est:5.4f}")

#C(sphere)[T.1]: 2.0659
#C(sphere)[T.2]: 1.7884
#C(sphere)[T.3]: 1.2399
#C(sphere)[T.4]: 1.1869
#C(sphere)[T.5]: 1.1590
#C(sphere)[T.6]: 2.3556
#C(sphere)[T.7]: 3.1617
#C(sphere)[T.9]: 1.3572
#C(sphere)[T.10]: 2.2638
#C(sphere)[T.11]: 1.4910
#C(sphere)[T.12]: 1.9173
#C(sphere)[T.13]: 1.7580
#C(sphere)[T.14]: 3.9062
#C(sphere)[T.15]: 1.3512
#C(sphere)[T.16]: 1.2682
#C(sphere)[T.17]: 2.5250
#C(sphere)[T.20]: 1.1771
#C(sphere)[T.21]: 1.2471
#C(sphere)[T.26]: 1.3739
#C(sphere)[T.27]: 1.2507
#C(sphere)[T.28]: 1.2170
#gender  : 5.7111
#exp     : 17.3638          Отдельно без квадрата exp: 4.2517
#I(exp ** 2): 15.0540
#degree  : 1.3804
#age     : 4.5019
#marriage: 3.1828
#marriage:gender: 5.0382
#child   : 3.2755
#child:gender: 5.7181
# Вывод: Всё ок
