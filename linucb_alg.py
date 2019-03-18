import numpy as np
import pandas as pd

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


class LinUCB:
    def __init__(self,
                 dataset, # The full dataframe
                 na=3, # Number of possible actions
                 delta=0.05, # 1-delta is the probability of sublinear regret
                 alpha=None # Learning rate of the algorithm
                 ):
        # Skip certain features that are either useless or give away the answer
        self.columns_to_exclude = [
            'PharmGKB Subject ID',
            'Medications',
            'INR on Reported Therapeutic Dose of Warfarin',
            'Therapeutic Dose of Warfarin',
            'Indication for Warfarin Treatment',
            'Comorbidities'
        ]

        # Identify features that are actually real numbers
        self.columns_with_floats = [
            'Height (cm)',
            'Weight (kg)'
        ]

        # Prepare empty lists for keeping track of seen values
        self.past_vals = {}
        for e in self.columns_with_floats:
            self.past_vals[e] = np.array([])

        # Get unique values per column from the full dataset
        self.dataset_categories = {}
        for c in dataset.columns :
            if c not in self.columns_to_exclude and c not in self.columns_with_floats:
                self.dataset_categories[c] = get_unique(dataset.loc[:,c])

        print(self.dataset_categories)

        # Read in the state to get the dimension
        x = self.convert_to_statevec(dataset.loc[0,:])
        self.nd = np.size(x)

        self.na = na # Action dimension
        self.delta = delta # 1 - prob of sublinear regret
        self.T = dataset.values.shape[0]
        if alpha is None:
            self.alpha = np.sqrt(0.5*np.log(2*self.T*self.na/self.delta))
        else:
            self.alpha = alpha
        print("nd: ", self.nd, "na: ", self.na, " delta: ", self.delta, " T: ", self.T, " alpha: ", self.alpha)

        # Create params for the linUCB algorithm
        self.As = [np.eye(self.nd) for i in range(self.na)]
        self.bs = [np.zeros(self.nd) for i in range(self.na)]

    # Get the learning rate as a function of time to ensure sublinear regret with the specified probability
    def alpha(self):
        return

    # convert a dataframe row to a state vector which concatenates one-hot vectors
    def convert_to_statevec(self, datarow):
        state = np.array([1])
        for c in datarow.index:

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

        norm = np.linalg.norm(state)
        if np.any(np.isnan(norm)):
            print(state)
            print(norm)
        return state / norm

    # Apply the LinUCB algorithm and make a decision
    def __call__(self, state, t, reward_fn):
        x = self.convert_to_statevec(state)
        ps = np.zeros(self.na)
        for a in range(self.na):
            Ainv = np.linalg.inv(self.As[a])
            theta = np.dot(Ainv, self.bs[a])
            ps[a] = np.dot(theta, x) + self.alpha * np.sqrt(np.dot(x, np.dot(Ainv, x)))
        a = np.argmax(ps)
        r = reward_fn(a + 1, state)
        self.As[a] = self.As[a] + np.outer(x, x)
        self.bs[a] = self.bs[a] + x*r
        if t % 500 == 0:
            print("t: ", t, " ps: ", ps)
        return a+1, r
