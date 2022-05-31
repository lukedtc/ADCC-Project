import numpy as np
from scipy.integrate import solve_ivp

class ADCCModelUtils:

    def __init__(self, params: dict) -> None:

        self.k_on = params['k_on']
        self.k_off = params['k_off']
        self.q_on = params['q_on']
        self.q_off = params['q_off']
        self.t_0 = params['t_0']
        self.T_0 = params['T_0']
        self.N_0 = params['N_0']
        self.beta_on = params['beta_on']
        self.beta_off = params['beta_off']
        self.fk_1 = params['fk_1']
        self.rho_T = params['rho_T']
        self.rho_N = params['rho_N']
        self.time_step = params['time_step'] 
        self.gamma = params['gamma']
        self.mu = params['mu']
        self.t_start = params['t_start']
        self.t_end = params['t_end']

        pass

    def microscale_model(self, t, z):

        alpha_1 = self.k_on*self.t_0*self.rho_T*self.T_0
        alpha_2 = self.k_off*self.t_0
        n_1 = self.q_on*self.t_0*self.rho_N*self.T_0
        n_2 = self.q_off*self.t_0
        delta = self.rho_N/self.rho_T

        dA = -alpha_1*z[0]*z[1] + alpha_2*self.gamma*z[2] - n_1*self.mu*z[0]*z[3] + n_2*self.gamma*z[4]
        dT = -(alpha_1/self.gamma)*z[1]*z[0] + alpha_2*z[2]
        dTr = (alpha_1/self.gamma)*z[1]*z[0] - alpha_2*z[2]
        dN = -(n_1/delta*self.gamma)*z[3]*z[0] + n_2*z[4]
        dNr = (n_1/delta*self.gamma)*z[3]*z[0] - n_2*z[4]
        dz = [dA, dT, dTr, dN, dNr]

        return dz
    
    def macroscale_model(self, t, z, a1, a2):


        v_1 = self.beta_on*self.t_0*self.T_0
        v_20 = self.beta_off*self.t_0
        v_21 = (self.beta_off)**2*self.t_0

        dS = -self.f_k1*(a2*z[1])
        dC = v_1*(self.mu - z[1])*(z[0] - z[1]) - v_20*(a1*z[1]) - v_21*(a2*z[1]) - self.f_k1*(z[1]/2)
        dz = [dS, dC]

        return dz


    def solve_microscale(self, t_begin, ICs: list):
        
        t_final = t_begin + self.time_step
        t = np.array([t_begin, t_final])
        t_span = [t_begin, t_final]

        z = solve_ivp(self.microscale_model, t_span, ICs, method = 'Radau', t_eval = t )
        
        A = z.y[0], T = z.y[1], Tr = z.y[2], N = z.y[3], Nr = z.y[4]

        solution_dict =  {'A': A, 'T': T, 'Tr': Tr, 'N': N, 'N_r': Nr}

        return solution_dict

    def calc_lambda(self, Nr: float, Tr: float):

        lmda = Nr*(1 - Tr) + Tr*(1 - Nr)

        return lmda
    
    def adjust_complex_coeffs(self, lmda: float):

        if lmda < 0.2:
            a1 = 1
            a2 = 0
        elif 0.2 <= lmda < 0.4:
            a1 = 3/4
            a2 = 1/4
        elif 0.4 <= lmda < 0.6:
            a1 = 1/3
            a2 = 1/2
        elif 0.6 <= lmda < 0.8:
            a1 = 1/4
            a2 = 3/4
        else:
            a1 = 0
            a2 = 1
        
        return [a1, a2]

    def solve_model(self):

        microscale_ICs = [1, 1, 0, 1, 0]
        macroscale_ICs = [1, 0]

        t = self.t_start
        A = [1]
        T = [1]
        Tr = [0]
        N = [1]
        Nr = [0]
        S = [1]
        C = [0] 

        while t < self.t_end:

            microscale_dict = self.solve_microscale(t, microscale_ICs)
            A.append(microscale_dict['A'])
            T.append(microscale_dict['T']), Tr.append(microscale_dict['Tr'])
            N.append(microscale_dict['N']), Nr.append(microscale_dict['Nr'])

            lmda = self.calc_lambda(microscale_dict['Nr'], microscale_dict['Tr'])
            coeffs = self.adjust_complex_coeffs(lmda)
            a1 = coeffs[0], a2 = coeffs[1]

            t_dt = t + self.time_step
            t_evals = [t, t_dt]
            t_span = [t, t_dt]
            z = solve_ivp(self.macroscale_model, t_span, macroscale_ICs, method = 'Radau', t_eval = t_evals )
            S_val = z.y[0], C_val = z.y[1]
            S.append(S_val)
            C.append(C_val)

            # reset sim params
            microscale_ICs = [microscale_dict['A'], microscale_dict['T'],
                              microscale_dict['Tr'], microscale_dict['N'], microscale_dict['Nr']]
            macroscale_ICs = [S_val, C_val]

            t+=self.time_step
        
        return [A, T, Tr, N, Nr, S, C]









