import numpy as np
from scipy.integrate import solve_ivp

class ADCCModelUtils:

    def __init__(self, params: dict) -> None:

        self.k_on1 = params['k_on1']
        self.k_off = params['k_off']
        self.q_on1 = params['q_on1']
        self.q_off = params['q_off']
        self.t_0 = params['t_0']
        self.T_01 = params['T_01']
        self.beta_on = params['beta_on']
        self.beta_off = params['beta_off']
        self.fk_1 = params['fk_1']
        self.rho_T = params['rho_T']
        self.rho_N = params['rho_N']
        self.time_step = params['time_step'] 
        self.mu = params['mu']
        self.A_01 = params['A_01']
        self.t_start = params['t_start']
        self.t_end = params['t_end']

        pass

    def microscale_model(self, t, z):

        A_0 = (172635*self.A_01*1e-7)/1.66e-24
        T_0 = (6e14*self.T_01*1e-7)/1.66e-9
        N_0 = self.mu*T_0
        k_on = self.k_on1*self.T_01
        q_on = self.q_on1*self.T_01


        # params
        alpha_1 = k_on*self.rho_T*T_0*self.t_0
        alpha_2 = self.k_off*self.t_0
        n_1 = q_on*self.t_0*self.rho_N*N_0
        n_2 = self.q_off*self.t_0
        gamma_T = (self.rho_T*T_0/A_0)
        gamma_N = (self.rho_N*N_0/A_0)

        dA = -alpha_1*z[0]*z[1] + gamma_T*alpha_2*z[2] - n_1*z[0]*z[3] + gamma_N*n_2*z[4]
        dTr = (-alpha_1/gamma_T)*z[0]*z[1] + alpha_2*z[2]
        dTrA = (alpha_1/gamma_T)*z[0]*z[1] - alpha_2*z[2]
        dNr = (-n_1/gamma_N)*z[0]*z[3] + n_2*z[4]
        dNrA = (n_1/gamma_N)*z[0]*z[3] - n_2*z[4]
        dz = [dA, dTr, dTrA, dNr, dNrA]

        return dz
    
    def macroscale_model(self, t, z, a1, a2):


        v_1 = self.beta_on*self.t_0*self.T_01
        v_20 = self.beta_off*self.t_0
        v_21 = (self.beta_off)**2*self.t_0

        dS = -self.fk_1*(a2*z[1])
        dC = v_1*(self.mu - z[1])*(z[0] - z[1]) - v_20*(a1*z[1]) - v_21*(a2*z[1]) - self.fk_1*(z[1]/2)
        dz = [dS, dC]

        return dz


    def solve_microscale(self, t_begin, ICs):
        
        t_final = t_begin + 2*self.time_step
        t_evals = np.array([t_begin, t_begin + self.time_step, t_final], dtype=object)
        t_span = [t_begin, t_final]

        z = solve_ivp(self.microscale_model, t_span, ICs, method = 'Radau', t_eval = t_evals )
        
        A = z.y[0][1]
        Tr = z.y[1][1]
        TrA = z.y[2][1]
        Nr = z.y[3][1]
        NrA = z.y[4][1]

        solution_dict =  {'A': A, 'Tr': Tr, 'TrA': TrA, 'Nr': Nr, 'NrA': NrA}

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
        
        return a1, a2

    def solve_model(self):

        microscale_ICs = np.array([1, 1, 0, 1, 0], dtype=object)
        macroscale_ICs = np.array([1, 0], dtype=object)

        t = float(self.t_start)
        A = [1]
        Tr = [1]
        TrA = [0]
        Nr = [1]
        NrA = [0]
        S = [1]
        C = [0] 
        macro_time_points = [float(self.t_start)]

        while t < self.t_end:

            microscale_dict = self.solve_microscale(t, microscale_ICs)
            A.append(microscale_dict['A'])
            Tr.append(microscale_dict['Tr'])
            TrA.append(microscale_dict['TrA'])
            Nr.append(microscale_dict['Nr']) 
            NrA.append(microscale_dict['NrA'])

            lmda = self.calc_lambda(microscale_dict['Nr'], microscale_dict['Tr'])
            coeffs = self.adjust_complex_coeffs(lmda)

            t_dt = t + 2*float(self.time_step)
            t_evals = np.array([t, t + float(self.time_step), t_dt], dtype=object)
            t_span = [t, t_dt]
            z = solve_ivp(self.macroscale_model, t_span, macroscale_ICs, args = (coeffs[0], coeffs[1]), method = 'Radau', t_eval = t_evals )
            S_val = z.y[0][1]
            C_val = z.y[1][1]
            S.append(S_val)
            C.append(C_val)

            # reset sim params
            microscale_ICs = [microscale_dict['A'], microscale_dict['Tr'],
                              microscale_dict['TrA'], microscale_dict['Nr'], microscale_dict['NrA']]
            macroscale_ICs = [S_val, C_val]

            t+=self.time_step
            macro_time_points.append(t)
        
        return [A, Tr, TrA, Nr, NrA, S, C, macro_time_points]









