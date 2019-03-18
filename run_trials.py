import pandas as pd
import numpy as np
from utils import apply_policy, vbin_dose, binary_reward, accuracy
from baseline_algs import fixed_dose, clinical_dosing_alg
from linucb_alg import LinUCB, get_unique

filename = "data/warfarin.csv"
data = pd.read_csv(filename)
yname = 'Therapeutic Dose of Warfarin'


ground_truth = vbin_dose(np.squeeze(data.loc[:, data.columns == yname].values))
print(ground_truth)


fd_actions, fd_rewards = apply_policy(fixed_dose, data, binary_reward)
fd_acc = accuracy(fd_actions, ground_truth)
print("accuracy of fixed dose: ", fd_acc)


cd_actions, cd_rewards = apply_policy(clinical_dosing_alg, data, binary_reward)
cd_acc = accuracy(cd_actions, ground_truth)
print('accuracy of clinical dosing algorithm: ', cd_acc)

linucb_alg = LinUCB(data)
lucb_actions, lucb_rewards = apply_policy(linucb_alg, data, binary_reward)
lucb_acc = accuracy(lucb_actions, ground_truth)
print('accuracy of LinUCB dosing algorithm: ', lucb_acc)