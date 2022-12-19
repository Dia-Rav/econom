import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

dataset = pd.read_excel (r"D:\Уроки\3 курс\РАНХ\Эконометрика\Проект\base.xlsx")


# ТЕСТ ХАУСМАНА
# 1 ШАГ
model_child = smf.ols("child ~ gender + exp + I(exp**2) + degree + boss + C (sphere) + marriage", data=dataset)
model_child_est = model_child.fit()

model_health = smf.ols("health_level ~ gender + exp + I(exp**2) + degree + boss + C (sphere) + marriage", data=dataset)
model_health_est = model_health.fit()

model_regular = smf.ols("regular ~ gender + exp + I(exp**2) + degree + boss + C (sphere) + marriage", data=dataset)
model_regular_est = model_regular.fit()

# 2 ШАГ
child = pd.DataFrame(model_child_est.resid, columns = ['child_resid'])
health = pd.DataFrame(model_health_est.resid, columns = ['health_resid'])
regular = pd.DataFrame(model_regular_est.resid, columns = ['regular_resid'])

new_data = dataset.join(child)
new_data = new_data.join(health)
new_data = new_data.join(regular)

model_test = smf.ols("np.log(salary) ~ child + health_level + regular + child_resid + health_resid + regular_resid", data=new_data)
model_test_est = model_test.fit()
print(model_test_est.summary())