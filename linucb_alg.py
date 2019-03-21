import numpy as np
import pandas as pd

def bin_dose(dosage):
    if np.isnan(dosage): return 0
    if 0 < dosage < 21: return 1
    if 21 <= dosage <= 49: return 2
    if dosage > 49: return 3
    return 0


def get_unique(lst):
    arr = list(set(lst))
    new_arr = []
    for a in arr:
        if ~pd.isna([a])[0] and a != 'nan':
            new_arr.append(a)
    return new_arr


def to_onehot(feature, categories):
    oh = np.zeros(len(categories)+1)
    if feature in categories:
        oh[np.where(categories == feature)] = 1
    else:
        oh[-1] = 1
    return oh

def softmax(ps):
    exps = np.exp(ps)
    return exps / sum(exps)


class LinUCB:
    def __init__(self,
                 dataset, # The full dataframe
                 na=3, # Number of possible actions
                 delta=0.05, # 1-delta is the probability of sublinear regret
                 alpha=None, # Learning rate of the algorithm
                 verbose = False,
                 columns_to_use = None
                 ):
        # Skip certain features that are either useless or give away the answer
        self.columns_to_exclude = [
            'PharmGKB Subject ID',
            'Medications',
            'INR on Reported Therapeutic Dose of Warfarin',
            'Therapeutic Dose of Warfarin',
            'Indication for Warfarin Treatment',
            'Comorbidities',
            'Target INR',
            'Estimated Target INR Range Based on Indication'
        ]

        # Identify features that are actually real numbers
        self.columns_with_floats = [
            'Height (cm)',
            'Weight (kg)'
        ]

        self.verbose = verbose
        self.columns_to_use = dataset.columns if columns_to_use is None else columns_to_use

        # Prepare empty lists for keeping track of seen values
        self.past_vals = {}
        for e in self.columns_with_floats:
            self.past_vals[e] = np.array([])

        # Get unique values per column from the full dataset
        self.dataset_categories = {}
        for c in self.columns_to_use:
            if c not in self.columns_to_exclude and c not in self.columns_with_floats:
                self.dataset_categories[c] = get_unique(dataset.loc[:,c])

        # print(self.dataset_categories)

        self.state_mapping = {}

        # Read in the state to get the dimension
        x = self.convert_to_statevec(dataset.loc[0,:])
        self.nd = np.size(x)

        self.na = na # Action dimension
        self.delta = delta # 1 - prob of sublinear regret
        self.T = dataset.values.shape[0]
        # if alpha is None:
        #     self.alpha = np.sqrt(0.5*np.log(2*self.T*self.na/self.delta))
        # else:
        #     self.alpha = alpha
        self.alpha_override = alpha
        # print("nd: ", self.nd, "na: ", self.na, " delta: ", self.delta, " T: ", self.T, " alpha: ", self.alpha_override)

        # Create params for the linUCB algorithm
        self.reset()


    def reset(self):
        self.As = [np.eye(self.nd) for i in range(self.na)]
        self.Ainv = [np.eye(self.nd) for i in range(self.na)]
        self.bs = [np.zeros(self.nd) for i in range(self.na)]
        self.thetas = [np.zeros(self.nd) for i in range(self.na)]
        self.freeze_weights = False

    # Get the learning rate as a function of time to ensure sublinear regret with the specified probability
    def alpha(self, t):
        if self.alpha_override is None:
            return np.sqrt(0.5*np.log(2*(t+1)*self.na/self.delta))
        else:
            return self.alpha_override

    def print_thetas(self):
        d = {}
        for c in self.state_mapping:
            thetas = np.zeros(len(self.state_mapping[c]))
            for a in range(self.na):
                thetas += np.abs(self.thetas[a][self.state_mapping[c]])
            thetas = thetas / self.na
            d[np.mean(thetas)] = c

        for c in sorted(d):
            print(d[c], ": ", c)

    # convert a dataframe row to a state vector which concatenates one-hot vectors
    def convert_to_statevec(self, datarow):
        state = np.array([1])
        for c in self.columns_to_use:
            start_len = np.size(state)

            # Ignore any features that we have previously excluded
            if c in self.columns_to_exclude:
                continue

            # If the column contains real numbers, concatenate them directly
            if c in self.columns_with_floats:
                if np.isnan(datarow.loc[c]):
                    state = np.append(state, np.mean(self.past_vals[c]))
                else:
                    state = np.append(state, datarow.loc[c])
                    self.past_vals[c] = np.append(self.past_vals[c], datarow.loc[c])

            # Otherwise we have categories so convert the category to a one-hot vector
            else:
                oh = to_onehot(datarow.loc[c], self.dataset_categories[c])
                state = np.append(state, oh)

            end_len = np.size(state)
            if c not in self.state_mapping:
                self.state_mapping[c] = np.arange(start_len, end_len)

        norm = np.linalg.norm(state)
        if np.any(np.isnan(norm)):
            print(state)
            print(norm)
        return state / norm

    # Apply the LinUCB algorithm and make a decision
    def __call__(self, state, t, reward_fn):
        if self.verbose and t % 500 == 0:
            print("t: ", t)
        x = self.convert_to_statevec(state)
        ps = np.zeros(self.na)
        for a in range(self.na):
            ps[a] = np.dot(self.thetas[a], x) + self.alpha(t) * np.sqrt(np.dot(x, np.dot(self.Ainv[a], x)))

        ps = softmax(ps)
        a = np.argmax(ps)
        r = reward_fn(a + 1, state, ps[a])
        if not self.freeze_weights:
            self.As[a] = self.As[a] + np.outer(x, x)
            self.bs[a] = self.bs[a] + x*r
            self.Ainv[a] = np.linalg.inv(self.As[a])
            self.thetas[a] = np.dot(self.Ainv[a], self.bs[a])
        return a+1, r


class LinRegression:
    def __init__(self, data, p):
        y = data.loc[:, 'Therapeutic Dose of Warfarin']
        notnan = ~np.isnan(y)
        y = y[notnan]

        X = np.zeros((len(y), p.nd))
        index = 0
        for i in range(len(data)):
            if ~np.isnan(data.loc[i, 'Therapeutic Dose of Warfarin']):
                X[index, :] = p.convert_to_statevec(data.loc[i, :])
                index += 1

        self.theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
        self.p = p

    def __call__(self, state, t, reward_fn):
        x = self.p.convert_to_statevec(state)
        a = bin_dose(np.dot(self.theta, x))
        r = reward_fn(a, state)
        return a, r
