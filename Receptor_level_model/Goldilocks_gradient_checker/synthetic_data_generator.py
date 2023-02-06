import numpy as np
import pandas as pd
from synapse_solver import SynapseModel
from SALib.sample import saltelli

# Generate dataframe

df = pd.DataFrame(columns=['A0', 'kon', 'koff', 'qon', 'qoff', 'rtot_t', 'rtot_f', 'delta2', 'delta4', 'delta7', 'fc'])

# Generate parameter samples

param_space = {
    'num_vars': 9,
    'names': ['kon', 'koff',  'qon', 'qoff', 'rtot_t', 'rtot_f', 'delta2', 'delta4', 'delta7'],
    'bounds': [[1e1, 1e7],
               [5e-8, 5e-1],
               [8e1, 1e7],
               [5e-8, 1e-1],
               [5e1, 1e8],
               [4e1, 3e8],
               [0.1, 200],
               [0.1, 200],
               [0.1, 200]]
}

paramaters = saltelli.sample(param_space, 2048)
print(len(paramaters))

A0s = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

tend = 100
t = np.geomspace(1e-10, tend, 64)

# Initialize parameter dictionary

param_dict = {'kon' : 1e5,
          'koff': 1e-4,
          'qon': 6.5e3,
          'qoff': 4.7e-3,
          'rtot_t': 1e5,
          'rtot_f': 2.2e5,
          'delta2': 1,
          'delta4': 1,
          'delta7': 1,
          'timesteps': t,
          'tend': tend}

# Create model solver

model_solver = SynapseModel(param_dict)

# Generate data


for i in range(len(paramaters)):

    paramater_array = paramaters[i]
    param_dict['kon'] = paramater_array[0]
    param_dict['koff'] = paramater_array[1]
    param_dict['qon'] = paramater_array[2]
    param_dict['qoff'] = paramater_array[3]
    param_dict['rtot_t'] = paramater_array[4]
    param_dict['rtot_f'] = paramater_array[5]
    param_dict['delta2'] = paramater_array[6]
    param_dict['delta4'] = paramater_array[7]
    param_dict['delta7'] = paramater_array[8]

    model_solver.set_params(param_dict)
    for A0 in A0s:
    
        fc = model_solver.calc_fc(A0)

        df = df.append({'A0': A0,
                        'kon': param_dict['kon'],
                        'koff': param_dict['koff'],
                        'qon': param_dict['qon'],
                        'qoff': param_dict['qoff'],
                        'rtot_t': param_dict['rtot_t'],
                        'rtot_f': param_dict['rtot_f'],
                        'delta2': param_dict['delta2'],
                        'delta4': param_dict['delta4'],
                        'delta7': param_dict['delta7'],
                        'fc': fc}, ignore_index = True)

# Write to file

df.to_csv('/Users/lukeheirene/ADCC-Project/Receptor_level_model/Goldilocks_gradient_checker/synthetic_synapse_fc_outputs_large_range.csv')