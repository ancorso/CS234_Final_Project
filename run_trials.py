import pandas as pd
import numpy as np
from utils import binary_reward, apply_policy_multiple, dataset_accuracy
from baseline_algs import fixed_dose, clinical_dosing_nan_to_medium, clinical_dosing_nan_to_zero
from linucb_alg import LinUCB, LinRegression
from matplotlib import pyplot as plt

import random
from datetime import datetime
random.seed(datetime.now())

filename = "data/warfarin.csv"
data = pd.read_csv(filename)


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


p_all_data = LinUCB(data)

p_lin_reg = LinRegression(data, p_all_data)
lr_acc = dataset_accuracy(p_lin_reg, data, binary_reward)
print("linear regression accuracy: ", lr_acc)


# Plot the reward of teh fixed dose algorithm
# comparing regret and running accuracy
# actions, rewards, accuracy = apply_policy_multiple(fixed_dose, data, binary_reward, 10)
# regret = -np.cumsum(rewards, axis=1)
#
# mean_reg_fd = np.mean(regret, axis=0)
# std_reg = np.std(regret, axis=0)
# mean_acc = np.mean(accuracy, axis=0)
# std_acc = np.std(accuracy, axis=0)
#
# plt.figure(2)
# x = np.arange(0, mean_reg_fd.shape[0])
# plt.plot(x, mean_reg_fd)
# plt.fill_between(x, mean_reg_fd - std_reg, mean_reg_fd + std_reg, alpha=0.4)
# plt.xlabel("Patient Number")
# plt.ylabel("Regret")
# plt.title("Regret for Fixed Dose Algorithm")
# plt.savefig("regret_fd.png")

policies = [fixed_dose, clinical_dosing_nan_to_medium, p_lin_reg, p_low_data, p_all_data]
policies = [p_all_data]
labels = ["Fixed Medium Dose", "Clinical Dosing Algorithm", "Linear Regression Model", "LinUCB (Top 10 features)", "LinUCB (All features)"]
labels = ["LinUCB (All features)"]

for pi in range(len(policies)):
    policy = policies[pi]
    label = labels[pi]

    print("Running policy: ", label)

    # comparing regret and running accuracy
    actions, rewards, accuracy = apply_policy_multiple(policy, data, binary_reward, 10)
    regret = -np.cumsum(rewards, axis = 1)

    mean_reg =  np.mean(regret, axis = 0) # - mean_reg_fd
    std_reg = np.std(regret, axis = 0)
    mean_acc = np.mean(accuracy, axis=0)
    std_acc = np.std(accuracy, axis=0)

    if not (label == "Fixed Medium Dose"):
        plt.figure(0)
        x = np.arange(0,mean_reg.shape[0])
        plt.plot(x, mean_reg, label=labels[pi])
        plt.fill_between(x, mean_reg-std_reg, mean_reg+std_reg, alpha=0.4)


    plt.figure(1)
    x = np.arange(0,mean_acc.shape[0])
    plt.plot(x, mean_acc, label=labels[pi])
    plt.fill_between(x, mean_acc-std_acc, mean_acc+std_acc, alpha=0.4)


plt.figure(0)
plt.xlabel("Patient Number")
plt.ylabel("Regret Difference from Fixed")
plt.title("Regret Comparison of Algorithms")
plt.legend()
plt.savefig("regret.png")

plt.figure(1)
plt.xlabel("Patient Number")
plt.ylabel("Average Accuracy")
plt.title("Accuracy Comparison of Algorithms")
plt.legend()
plt.savefig("accuracy.png")
