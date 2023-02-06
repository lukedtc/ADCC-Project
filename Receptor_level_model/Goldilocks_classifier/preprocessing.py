import numpy as np
import pandas as pd
import random

class PreProcessor():

    def __init__(self, model_data_path:str) -> None:
        self.model_output = model_data_path
        pass

    def get_data(self): 
    # Extract fc arrays
        df = pd.read_csv(self.model_output)
        df_fc = df['fc']
        model_vals = np.zeros((8, int(df_fc.shape[0]/8)))

        j = 0
        while j < df_fc.shape[0]:
            df1 = df_fc.loc[8*j:8*j+7]
            for i in range(len(df1)):
                model_vals[i, j] = df1.values[i]
            j+=1

        # Generate goldilocks data 

        def gaussian(x, mu, sig, a):
            return a*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        num_curves = int(0.5*model_vals.shape[1])
        print(num_curves)
        xarray = np.linspace(0, 10, 8)
        gaussian_vals = np.zeros((8, num_curves))

        for i in range(num_curves):
            mu = np.random.uniform(0, 10)
            sigma = np.random.uniform(0, max(mu-2, 1)) 
            a = random.random()


            data = gaussian(xarray, mu, sigma, a) 
            c = random.uniform(0, min(0.2, 1-np.max(data)))
            cs = np.ones_like(xarray)*c
    

            for j in range(len(data)):
                data[j] = data[j] + cs[j]
    
            for k in range(len(data)):
                gaussian_vals[k, i] = data[k]

        df = pd.DataFrame(columns=['arrays', 'label'])

        for i in range(model_vals.shape[1]):
            array = np.array(model_vals[:,i])
            dataframe = pd.DataFrame([[array, 0]], columns=['arrays', 'label'])
            df = pd.concat([df, dataframe], ignore_index=True)

        for i in range(gaussian_vals.shape[1]):
            array = np.array(gaussian_vals[:,i])
            dataframe = pd.DataFrame([[array, 1]], columns=['arrays', 'label'])
            df = pd.concat([df, dataframe], ignore_index=True)


        df.to_csv('/Users/lukeheirene/ADCC-Project/Receptor_level_model/Goldilocks_classifier/unshuffled_data.csv')

        return(df, gaussian_vals, model_vals)