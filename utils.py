import numpy as np
from linucb_alg import LinUCB

def bin_dose(dosage):
    if np.isnan(dosage): return 0
    if 0 < dosage < 21: return 1
    if 21 <= dosage <= 49: return 2
    if dosage > 49: return 3
    return 0

vbin_dose = np.vectorize(bin_dose)


def binary_reward(pred_bin, data_row):
    ground_truth = bin_dose(data_row.loc['Therapeutic Dose of Warfarin'])
    return 0 if pred_bin == ground_truth or np.isnan(ground_truth) else -1


def accuracy(predictions, ground_truth):
    not_nan = (ground_truth != 0) & (predictions != 0)
    num_correct = np.sum(predictions[not_nan] == ground_truth[not_nan])
    return num_correct / np.size(ground_truth[not_nan])

def severe_mistakes(predictions, ground_truth):
    not_nan = (ground_truth != 0) & (predictions != 0)
    predictions = predictions[not_nan]
    ground_truth = ground_truth[not_nan]
    np.sum(np.abs(predictions - ground_truth) > 1)

def dataset_severe_mistakes(policy, dataset, reward_fn):
    actions, rewards = apply_policy(policy, dataset, reward_fn)
    yname = 'Therapeutic Dose of Warfarin'
    ground_truth = vbin_dose(np.squeeze(dataset.loc[:, dataset.columns == yname].values))
    return severe_mistakes(actions, ground_truth)

def apply_policy(policy, dataset, reward_fn):
    rows = dataset.values.shape[0]
    actions, rewards = np.zeros(rows),  np.zeros(rows)
    for t in range(rows):
        actions[t], rewards[t] = policy(dataset.loc[t, :], t, reward_fn)
    return actions, rewards

def dataset_accuracy(policy, dataset, reward_fn):
    actions, rewards = apply_policy(policy, dataset, reward_fn)
    yname = 'Therapeutic Dose of Warfarin'
    ground_truth = vbin_dose(np.squeeze(dataset.loc[:, dataset.columns == yname].values))
    return accuracy(actions, ground_truth)

def apply_policy_multiple(policy, dataset, reward_fn, trials, accuracy_check = 200):
    m = len(dataset)
    mtrain = int(0.8*m)
    actions = np.zeros((trials, mtrain))
    rewards = np.zeros((trials, mtrain))

    macc = int(mtrain / accuracy_check)+1
    accuracy = np.zeros((trials, macc))

    for i in range(trials):
        if isinstance(policy, LinUCB):
            print("resetting LinUCB")
            policy.reset()

        print("trial: ", i)
        newdata = dataset.sample(frac=1).reset_index(drop=True)
        train = newdata[:mtrain]
        test = newdata[mtrain:].reset_index(drop=True)
        acc_count = 0
        for t in range(mtrain):
            actions[i, t], rewards[i, t] = policy(train.loc[t, :], t, reward_fn)
            if t < mtrain and t % accuracy_check == 0:
                if isinstance(policy, LinUCB):
                    policy.freeze_weights = True
                    accuracy[i, acc_count] = dataset_accuracy(policy, test, reward_fn)
                    policy.freeze_weights = False
                else:
                    accuracy[i, acc_count] = dataset_accuracy(policy, test, reward_fn)

                acc_count += 1

    return actions, rewards, accuracy






