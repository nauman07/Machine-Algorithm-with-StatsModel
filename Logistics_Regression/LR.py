import statsmodels.api as sa
import statsmodels.formula.api as sfa
biopsy = sa.datasets.get_rdataset("biopsy","MASS")
biopsy_data = biopsy.data
biopsy_data.rename(columns={"class":"Class"},inplace=True)
biopsy_data.Class = biopsy_data.Class.map({"benign":0,"malignant":1})
#biopsy_data["V1"] = np.divide(biopsy_data["V1"] - biopsy_data["V1"].min(), biopsy_data["V1"].max() - biopsy_data["V1"].min())
log_mod1 = sfa.logit("Class~V1",biopsy_data)
log_res1 = log_mod1.fit()
print(log_res1.prsquared)
