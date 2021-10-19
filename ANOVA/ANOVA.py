import statsmodels.api as sm
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats import anova

mtcars = sm.datasets.get_rdataset("mtcars", "datasets").data
df = pd.DataFrame(mtcars)
model = smf.ols(formula='np.log(mpg) ~ np.log(wt)', data=mtcars).fit()
#print(anova.anova_lm(model))
print(anova.anova_lm(model).F["np.log(wt)"])
