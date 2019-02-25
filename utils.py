import numpy as np


def age_to_decade(age):
    if isinstance(age, str):
        return int(age[0])
    return float('nan')


def bin_dose(dosage):
    dosage = np.copy(dosage)
    dosage[np.isnan(dosage)] = -1
    bins = np.zeros_like(dosage, dtype=np.int)
    bins[(dosage < 21) & (dosage > 0)] = 1
    bins[(dosage <= 49) & (dosage >= 21)] = 2
    bins[dosage > 49] = 3
    return bins


def accuracy(predictions, ground_truth):
    not_nan = (ground_truth != 0) & (predictions != 0)
    print("sum of not nans: ", np.sum(not_nan))
    num_correct = np.sum(predictions[not_nan] == ground_truth[not_nan])
    return num_correct / np.size(ground_truth[not_nan])


def fixed_dose(state):
    return 35.0  # mg/week


def clinical_dosing_alg(state):
    sqrt_dose = 4.0376 \
                - 0.2546 * age_to_decade(state.loc['Age']) \
                + 0.0118 * state.loc['Height (cm)'] \
                + 0.0134 * state.loc['Weight (kg)'] \
                - 0.6752 * (state.loc['Race'] == 'Asian') \
                + 0.4060 * (state.loc['Race'] == 'Black or African American') \
                + 0.0443 * (state.loc['Race'] == 'Unknown') \
                + 1.2799 * ((state.loc['Carbamazepine (Tegretol)'] == 1) or
                            (state.loc['Phenytoin (Dilantin)'] == 1) or
                            (state.loc['Rifampin or Rifampicin'] == 1)) \
                - 0.5695 * state.loc['Amiodarone (Cordarone)']

    return sqrt_dose**2


def apply_policy(policy, data):
    rows = data.values.shape[0]
    pred = np.zeros(rows)
    for i in range(rows):
        pred[i] = policy(data.loc[i, :])
    return bin_dose(pred)


