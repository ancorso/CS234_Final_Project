import pandas as pd
import numpy as np
from utils import binary_reward, apply_policy_multiple, dataset_accuracy
from baseline_algs import fixed_dose, clinical_dosing_nan_to_medium, clinical_dosing_nan_to_zero
from linucb_alg import LinUCB, LinRegression
from matplotlib import pyplot as plt

filename = "data/warfarin.csv"
data = pd.read_csv(filename)

# baseline values:
fd_acc = dataset_accuracy(fixed_dose, data, binary_reward)
print("accuracy of fixed dose: ", fd_acc)

cd_acc = dataset_accuracy(clinical_dosing_nan_to_zero, data, binary_reward)
print('accuracy of clinical dosing algorithm (ignoring nans): ', cd_acc)

cd_working_acc = dataset_accuracy(clinical_dosing_nan_to_medium, data, binary_reward)
print('accuracy of clinical dosing algorithm (nan to medium): ', cd_working_acc)

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
    ], verbose = True)
lucb_acc = dataset_accuracy(p, data, binary_reward)
print('accuracy of LinUCB dosing algorithm with limited features: ', lucb_acc)


p_all_data = LinUCB(data, verbose = True)
lucb_acc = dataset_accuracy(p_all_data, data, binary_reward)
p_all_data.print_thetas()
print('accuracy of LinUCB dosing algorithm with all features: ', lucb_acc)


p_lin_reg = LinRegression(data, p_all_data)
lr_acc = dataset_accuracy(p_lin_reg, data, binary_reward)
print("linear regression accuracy: ", lr_acc)
