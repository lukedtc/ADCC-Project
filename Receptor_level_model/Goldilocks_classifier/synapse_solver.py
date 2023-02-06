from scipy.integrate import solve_ivp


class SynapseModel():

    def __init__(self, params:dict):

        self.kon = params['kon']
        self.koff = params['koff']
        self.qon = params['qon']
        self.qoff = params['qoff']
        self.rtot_t = params['rtot_t']
        self.rtot_f = params['rtot_f']
        self.delta2 = params['delta2']
        self.delta4 = params['delta4']
        self.delta7 = params['delta7']
        self.timesteps = params['timesteps']
        self.tend = params['tend']
        self.target_cell_number = 5e3
    

    def tumour_cell_binding(self, t, z, A0, delta, rtot, kon, koff):

        Ainit = A0
        k = Ainit*kon
        k1 = 2*k
        alpha1 = k1/koff
        Atot = 1e16*Ainit/self.target_cell_number
        beta = Atot/rtot

        dA1 = alpha1*(1-z[0]-2*z[1])*(beta-z[0]-z[1]) - z[0] - delta*alpha1*(1-z[0]-2*z[1])*z[0] + 2*z[1]
        dA2 = delta*alpha1*(1-z[0]-2*z[1])*z[0] - 2*z[1]

        return [dA1, dA2]
    
    def set_params(self, params:dict):

        self.kon = params['kon']
        self.koff = params['koff']
        self.qon = params['qon']
        self.qoff = params['qoff']
        self.rtot_t = params['rtot_t']
        self.rtot_f = params['rtot_f']
        self.delta2 = params['delta2']
        self.delta4 = params['delta4']
        self.delta7 = params['delta7']
        self.timesteps = params['timesteps']
        self.tend = params['tend']
    
    def tumour_cell_stst(self, A0, delta, rtot, kon, koff):

        tspan = [1e-10, self.tend]
        z0 = [0, 0]

        # solve single cell model to obtain equilibrium values
        # to use as ics for A10 and A20 in two cell model
        z = solve_ivp(self.tumour_cell_binding, tspan, z0, method='Radau', args=(delta, A0, rtot, kon, koff), t_eval=self.timesteps)
        A1 = z.y[0]
        A2 = z.y[1]
        A1_stst = A1[-1]
        A2_stst = A2[-1]
    
        return [A1_stst, A2_stst]
    
    def synapse_model(self, t, z, A0, delta2, rtot_t, kon, koff, rtot_f, qon, qoff, delta4, delta7, A10_0, A20_0):

        Ainit = A0
        k = Ainit*kon
        q = qon*Ainit
        delta5 = delta4*delta7/delta2
        delta6 = delta4
        k1 = 2*k
        k1off = koff
        k2 = delta2*k
        k2off = 2*koff
        k3 = q
        k3off = qoff
        k4 = delta4*q
        k4off = qoff
        k5 = delta5*q
        k5off = qoff
        k6 = 2*delta6*k
        k6off = koff
        k7 = delta7*k
        k7off = 2*koff

        alpha1 = k1/k1off
        alpha2 = k2/k1off
        gamma2 = k2off/k1off
        alpha3 = k3/k1off
        gamma3 = k3off/k1off
        alpha4 = k4/k1off
        gamma4 = k4off/k1off
        alpha5 = k5/k1off
        gamma5 = k5off/k1off
        alpha6 = k6/k1off
        gamma6 = k6off/k1off
        alpha7 = k7/k1off
        gamma7 = k7off/k1off

        beta_synapse = 1e1*Ainit
        beta_t = beta_synapse + A10_0 + A20_0
        phi = rtot_t/rtot_f
        beta_f = beta_t*phi

        A00 = (1 - (1/beta_t)*(z[0] + z[1] + z[3] + z[4] + (1/phi)*z[2]))
        rt = 1- z[0] - z[3] - 2*(z[1] + z[4])
        rf = 1 - z[2] - phi*(z[3] + z[4])
    
        dA10 = alpha1*beta_t*(A00)*(rt) - z[0] - alpha2*z[0]*(rt) + gamma2*z[1] - alpha4*z[0]*(rf) + gamma4*z[3]
        dA20 = alpha2*z[0]*(rt) - gamma2*z[1] - alpha5*z[1]*(rf) + gamma5*z[4]
        dA01 = alpha3*beta_f*(A00)*(rf) - gamma3*z[2] - alpha6*z[2]*(rt) + gamma6*phi*z[3]
        dA11 = alpha4*z[0]*(rf) - gamma4*z[3] + (alpha6/phi)*z[2]*(rt) - gamma6*z[3] - alpha7*z[3]*(rt) + gamma7*z[4]
        dA21 = alpha5*z[1]*(rf) - gamma5*z[4] + alpha7*z[3]*(rt) - gamma7*z[4]
        dz = [dA10, dA20, dA01, dA11, dA21]

        return dz

    def calc_fc(self, A0):

        ICS = self.tumour_cell_stst(A0, self.delta2, self.rtot_t, self.kon, self.koff)
        z01 = [ICS[0], ICS[1], 0, 0, 0]
        t_span1 = [1e-10, self.tend]
    
        z = solve_ivp(self.synapse_model, t_span1, z01, method='Radau', t_eval=self.timesteps,
                      args=(A0, self.delta2, self.rtot_t, self.kon, self.koff,
                            self.rtot_f, self.qon, self.qoff, self.delta4, self.delta7, ICS[0], ICS[1]))

        A11 = z.y[3]
        A21 = z.y[4]
        fc = A11[-1] + A21[-1]

        return fc
    
    def solve_model(self, A0):

        ICS = self.tumour_cell_stst(A0, self.delta2, self.rtot_t, self.kon, self.koff)
        z01 = [ICS[0], ICS[1], 0, 0, 0]
        t_span1 = [1e-10, self.tend]
    
        z = solve_ivp(self.synapse_model, t_span1, z01, method='Radau', t_eval=self.timesteps,
                      args=(A0, self.delta2, self.rtot_t, self.kon, self.koff,
                            self.rtot_f, self.qon, self.qoff, self.delta4, self.delta7, ICS[0], ICS[1]))

        A10 = z.y[0]
        A20 = z.y[1]
        A01 = z.y[2]
        A11 = z.y[3]
        A21 = z.y[4]

        return [A10, A20, A01, A11, A21]