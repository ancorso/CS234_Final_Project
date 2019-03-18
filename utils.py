import numpy as np

def bin_dose(dosage):
    if np.isnan(dosage): return 0
    if 0 < dosage < 21: return 1
    if 21 <= dosage <= 49: return 2
    if dosage > 49: return 3
    return 0

vbin_dose = np.vectorize(bin_dose)


def binary_reward(pred_bin, data_row):
    ground_truth = bin_dose(data_row.loc['Therapeutic Dose of Warfarin'])
    return 1 if pred_bin == ground_truth else 0


def accuracy(predictions, ground_truth):
    not_nan = (ground_truth != 0) & (predictions != 0)
    num_correct = np.sum(predictions[not_nan] == ground_truth[not_nan])
    return num_correct / np.size(ground_truth[not_nan])


def apply_policy(policy, dataset, reward_fn):
    rows = dataset.values.shape[0]
    actions, rewards = np.zeros(rows),  np.zeros(rows)
    for t in range(rows):
        actions[t], rewards[t] = policy(dataset.loc[t, :], t, reward_fn)
    return actions, rewards


