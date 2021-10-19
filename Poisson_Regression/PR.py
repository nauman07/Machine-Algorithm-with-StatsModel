import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
df_insurance=sm.datasets.get_rdataset("Insurance","MASS")
df_data=df_insurance.data
insurance_model=smf.poisson('Claims ~ np.log(Holders)',df_data)
insurance_model_result=insurance_model.fit()
res=(insurance_model_result.resid)
print(np.sum(res))
