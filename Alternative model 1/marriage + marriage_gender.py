import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

dataset = pd.read_excel (r"D:\Уроки\3 курс\РАНХ\Эконометрика\Проект\base.xlsx")

# МОДЕЛЬ С НАЛИЧИЕМ БРАКА И ДЕТЕЙ

model = smf.ols("np.log(salary) ~ gender + exp + I(exp**2) + degree + age + C (sphere) + marriage + marriage:gender", data=dataset)
model_est = model.fit()
print(model_est.summary())

# тест Зарембки (чтобы было на всякий случай)
model = smf.ols("salary_z ~ gender + exp + I(exp**2) + degree + age + C (sphere)  + marriage + marriage:gender", data=dataset)
model_est = model.fit()
print(model_est.ssr)
# RSS_1 = 24.75150015829631

model = smf.ols("np.log(salary_z)~ gender + exp + I(exp**2) + degree + age + C (sphere)  + marriage + marriage:gender + child + child:gender", data=dataset)
model_est = model.fit()
print(model_est.ssr)
# RSS_2 = 19.605408881646753
# Xi = 8.04743694462

# тест голдфелда квандта
print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 7, split=.3, drop=.4))
# ((0.6052192965778995, 0.8992864047001389, 'increasing') - сортировка по переменной marriage - вывод: гомоскедастичность
print(sms.het_goldfeldquandt(y=model.endog, x=model.exog, idx = 8, split=.3, drop=.4))
# (0.6688080713125863, 0.8483666834326586, 'increasing') - сортировка по переменной marriage:gender - вывод: гомоскедастичность
 


# тест глейзера 
from scipy.stats import chi2
model_aux = smf.ols("abs(model_est.resid) ~ marriage", data=dataset)
model_aux_est = model_aux.fit()
stat_aux = model_aux_est.ess / ((1 - 2 / np.pi) * np.var(model_est.resid))
print(f"Stat: {stat_aux:5.4f}, Critical value: {chi2.ppf(0.95, df=model_aux.df_model):5.4f}, \
p-value: {1 - chi2.cdf(stat_aux, df=model_aux.df_model):5.4f}")

# Stat: 0.3616, Critical value: 3.8415,     p-value: 0.5476 - marriage
# Stat: 1.0211, Critical value: 3.8415,     p-value: 0.3123 - marriage:gender
# коэффециенты незначимы, гетерскедостичности нет


# Тест Бройша-Пагана.
print(sms.het_breuschpagan(model_est.resid, model.exog))
# lm: float lagrange multiplier statistic
# lm_pvalue: float p-value of lagrange multiplier test
# fvalue: float f-statistic of the hypothesis that the error variance does not depend on x
# f_pvalue: float p-value for the f-statistic
# (33.668322404684204, 0.2120071329761758, 1.244540943640595, 0.2043582142876271)
# # на любом уровне значимости нет гетероскедастичности

from chow_test import chow_test
print(chow_test(y_series=pd.Series(model.endog), X_series=pd.DataFrame(model.exog), last_index=1499, first_index=1500, significance=0.05))

# VIF-тест на мультиколлинеарность
for i in range(1, model.exog.shape[1]):
    vif_est = vif(model.exog, i)
    print(f"{model_est.params.index[i]:8}: {vif_est:5.4f}")

#C(sphere)[T.1]: 2.0653
#C(sphere)[T.2]: 1.7562
#C(sphere)[T.3]: 1.2372
#C(sphere)[T.4]: 1.1430
#C(sphere)[T.5]: 1.1516
#C(sphere)[T.6]: 2.3238
#C(sphere)[T.7]: 3.1211
#C(sphere)[T.9]: 1.3537
#C(sphere)[T.10]: 2.2548
#C(sphere)[T.11]: 1.4621
#C(sphere)[T.12]: 1.8994
#C(sphere)[T.13]: 1.7366
#C(sphere)[T.14]: 3.8664
#C(sphere)[T.15]: 1.3395
#C(sphere)[T.16]: 1.2601
#C(sphere)[T.17]: 2.5241
#C(sphere)[T.20]: 1.1574
#C(sphere)[T.21]: 1.2309
#C(sphere)[T.26]: 1.3633
#C(sphere)[T.27]: 1.2332
#C(sphere)[T.28]: 1.2155
#gender  : 3.8859
#exp     : 15.8358
#I(exp ** 2): 13.0518
#degree  : 1.3365
#age     : 3.8338
#marriage: 2.9882
#marriage:gender: 4.8282
# Вывод: Всё ок
