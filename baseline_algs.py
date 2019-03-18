from utils import bin_dose

def fixed_dose(state, t, reward_fn):
    a = bin_dose(35.0)
    r = reward_fn(a, state)
    return a, r

def age_to_decade(age):
    if isinstance(age, str):
        return int(age[0])
    return float('nan')

def clinical_dosing_alg(state, t, reward_fn):
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

    a = bin_dose(sqrt_dose**2)
    r = reward_fn(a, state)
    return a, r