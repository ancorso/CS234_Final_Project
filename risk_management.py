import pandas as pd
import numpy as np
from utils import binary_reward,  wager_reward, severity_reward, severity_plus_wager, dataset_analyze_mistakes
from baseline_algs import fixed_dose, clinical_dosing_nan_to_medium, clinical_dosing_nan_to_zero
from linucb_alg import LinUCB, LinRegression
from matplotlib import pyplot as plt

filename = "data/warfarin.csv"
data = pd.read_csv(filename)

# baseline values:
fd_acc, fd_sev = dataset_analyze_mistakes(fixed_dose, data, binary_reward)
print("Fixed dose Accuracy: ", fd_acc, " Num severe mistakes", fd_sev)

cd_acc, cd_sev = dataset_analyze_mistakes(clinical_dosing_nan_to_zero, data, binary_reward)
print("clinical dosing (ignore nan) Accuracy: ", cd_acc, " Num severe mistakes", cd_sev)

cd_working_acc, cd_working_sev = dataset_analyze_mistakes(clinical_dosing_nan_to_medium, data, binary_reward)
print("clinical dosing (nan to medium) Accuracy: ", cd_working_acc, " Num severe mistakes", cd_working_sev)

p_low_data = LinUCB(data, columns_to_use=[
    "Height (cm)",
    "Weight (kg)",
    "Valve Replacement",
    "Subject Reached Stable Dose of Warfarin",
    "Current Smoker",
    "Cerivastatin (Baycol)",
    "Amiodarone (Cordarone)",
    'Simvastatin (Zocor)',
    'Aspirin',
    'Herbal Medications, Vitamins, Supplements',
    ])
lucb_acc10, lucb_sev10 = dataset_analyze_mistakes(p_low_data, data, binary_reward)
print("LinUCB (10 features) Accuracy: ", lucb_acc10, " Num severe mistakes", lucb_sev10)

wg_acc, wg_sev = dataset_analyze_mistakes(p_low_data, data, wager_reward)
print("LinUCB (10 features) Accuracy w/ wager: ", wg_acc, " Num severe mistakes", wg_sev)

sv_acc, sv_sev = dataset_analyze_mistakes(p_low_data, data, severity_reward)
print("LinUCB (10 features) Accuracy w/ severity: ", sv_acc, " Num severe mistakes", sv_sev)

wgsv_acc, wgsv_sev = dataset_analyze_mistakes(p_low_data, data, severity_plus_wager)
print("LinUCB (10 features) Accuracy w/ wager + severity: ", wgsv_acc, " Num severe mistakes", wgsv_sev)

p_all_data = LinUCB(data)
lucb_acc10, lucb_sev10 = dataset_analyze_mistakes(p_all_data, data, binary_reward)
print("LinUCB (all features) Accuracy: ", lucb_acc10, " Num severe mistakes", lucb_sev10)

wg_acc, wg_sev = dataset_analyze_mistakes(p_all_data, data, wager_reward)
print("LinUCB (all features) Accuracy w/ wager: ", wg_acc, " Num severe mistakes", wg_sev)

sv_acc, sv_sev = dataset_analyze_mistakes(p_all_data, data, severity_reward)
print("LinUCB (all features) Accuracy w/ severity: ", sv_acc, " Num severe mistakes", sv_sev)

wgsv_acc, wgsv_sev = dataset_analyze_mistakes(p_all_data, data, severity_plus_wager)
print("LinUCB (all features) Accuracy w/ wager + severity: ", wgsv_acc, " Num severe mistakes", wgsv_sev)

p_lin_reg = LinRegression(data, p_all_data)
lr_acc, lr_sev = dataset_analyze_mistakes(p_lin_reg, data, binary_reward)
print("LinRegression (all features) Accuracy: ", lr_acc, " Num severe mistakes", lr_sev)


