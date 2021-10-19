import statsmodels.api as sm
import numpy as np
mtcars = sm.datasets.get_rdataset('mtcars')
mtcars_data = mtcars.data
liner_model = sm.formula.ols('np.log(wt) ~ np.log(mpg)',mtcars_data)
liner_result = liner_model.fit()
print(liner_result.rsquared)
