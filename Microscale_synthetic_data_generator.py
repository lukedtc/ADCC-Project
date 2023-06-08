from scipy.integrate import solve_ivp
import numpy as np
from scipy.constants import N_A
import math

class DataGenerator():

    def __init__(self, param_dict):

        self.param_dict = param_dict
    
    def model_S(self, t, z, Ainit, rtot, kon, koff, k2, well_size = 150e-6, Target_cell_number = 2e5):

     k = Ainit*kon
     Atot = well_size*N_A*Ainit/Target_cell_number
     A0 = Atot - z[0] - z[1] 
     Ag = rtot - z[0] - 2*z[1]

     dA10 = 2*(k*Ag*A0/rtot) - koff*z[0] - (k2*Ag*z[0]/rtot) + 2*koff*z[1]
     dA11 = (k2*Ag*z[0]/rtot) - 2*koff*z[1]

     return [dA10, dA11]

    def model_R(self, t, z, Ainit, rtot, kon, koff, well_size = 150e-6, 
                             Target_cell_number = 2e5, tumour_cell_radius = 8e-6, r_ab = 1.25e-6):

        k = Ainit*kon
        reaction_volume = (2/3)*math.pi*((r_ab)**3)
        tumour_cell_surface_area = 4*math.pi*((tumour_cell_radius)**2)
        Atot = well_size*N_A*Ainit/Target_cell_number
        A0 = Atot - z[0] - z[1] 
        Ag = rtot - z[0] - 2*z[1]
        target_effective_conc1 =(rtot)*(tumour_cell_surface_area)*(1/N_A)/(reaction_volume)
        k2 = target_effective_conc1*kon

        dA10 = 2*(k*Ag*A0/rtot) - koff*z[0] - (k2*Ag*z[0]/rtot) + 2*koff*z[1]
        dA11 = (k2*Ag*z[0]/rtot) - 2*koff*z[1]

        return [dA10, dA11]

    def set_param_val(self, param_name: str, value: float) -> None:

        self.param_dict['param_name'] == value

    
    def generate_time_series(self, Ainit, model_name, noise_level):
       
       t_eval = self.param_dict['t_eval']
       t_span = self.param_dict['t_span']
       rtot = self.param_dict['rtot']
       kon = self.param_dict['kon']
       koff = self.param_dict['koff'] 

       
       if model_name == 'model S':
          
          k2 = self.param_dict['k2'] 
          sol = solve_ivp(self.model_S, t_span, [0, 0], method='Radau', 
                        t_eval=t_eval, args=(Ainit, rtot, kon, koff, k2))

       elif model_name == 'model R':
          
          sol = solve_ivp(self.model_R, t_span, [0, 0], method='Radau',
                        t_eval=t_eval, args=(Ainit, rtot, kon, koff))
        
       else:
          return ValueError('Incorrect model name')

       A1 = np.asarray(sol.y[0])
       A2 = np.asarray(sol.y[1])
       Ab1 = A1 + A2
       Ab = np.zeros_like(Ab1)
       noise = (np.random.normal(1, noise_level, len(Ab1))) 
       for i in range(len(Ab)):
           Ab[i] = Ab1[i]*noise[i]

       return Ab
    
    def generate_steady_states(self, Ainit_array, model_name, noise_level):
       
       t_eval = self.param_dict['t_eval']
       t_span = self.param_dict['t_span']
       rtot = self.param_dict['rtot']
       kon = self.param_dict['kon']
       koff = self.param_dict['koff'] 
       A1s = np.zeros_like(Ainit_array)
       A2s = np.zeros_like(Ainit_array) 

       for i, Ainit in enumerate(Ainit_array):
       
            if model_name == 'model S':
          
                k2 = self.param_dict['k2'] 
                z = solve_ivp(self.model_S, t_span, [0, 0], method='Radau', 
                              t_eval=t_eval, args=(Ainit, rtot, kon, koff, k2))

            elif model_name == 'model R':
          
                z = solve_ivp(self.model_R, t_span, [0, 0], method='Radau',
                              t_eval=t_eval, args=(Ainit, rtot, kon, koff))
        
            else:
                return ValueError('Incorrect model name')
            
            A1 = z.y[0]
            A2 = z.y[1]
            A1_stst = A1[-1]
            A2_stst = A2[-1]
            A1_stst = A1_stst*(1 + np.random.uniform(-noise_level, noise_level))
            A2_stst = A2_stst*(1 + np.random.uniform(-noise_level, noise_level))
            A1s[i] = A1_stst
            A2s[i] = A2_stst

        
       return [np.asarray(A1s), np.asarray(A2s)] 
       



