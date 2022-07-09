import numpy as np
import copy


class SE3_DMP():
    
    def __init__(self,N_bf=20,alphaz=25.0,betaz=6.25,dt=0.1):
        
        self.alphax = 1.0
        self.alphaz = alphaz
        self.betaz = betaz
        self.N_bf = N_bf # number of basis functions
        self.dt = dt
        self.T = 1.0

        self.phase = 1.0 # initialize phase variable

        # Centers of basis functions 
        self.c = np.ones(self.N_bf) 
        c_ = np.linspace(0,self.T,self.N_bf)
        for i in range(self.N_bf):
            self.c[i] = np.exp(-self.alphax *c_[i])

        # Widths of basis functions 
        # (as in https://github.com/studywolf/pydmps/blob/80b0a4518edf756773582cc5c40fdeee7e332169/pydmps/dmp_discrete.py#L37)
        self.h = np.ones(self.N_bf) * self.N_bf**1.5 / self.c / self.alphax

    def imitate(self, H_demo):

        self.N = H_demo.shape[0]
        
        # Demo trajectories of homogeneous transformation H, twist V and "wrench" W
        self.H_des = copy.deepcopy(H_demo) 
        self.V_des = self.SE3_gradient(self.H_des) 
        self.W_des = np.gradient(self.V_des,axis=0)/self.dt
                
        self.H0 = self.H_des[0,:,:]
        self.V0 = self.V_des[0,:]
        self.W0 = self.W_des[0,:]
        self.HT = self.H_des[-1,:,:]
        
        # Initialize the DMP
        self.H = copy.deepcopy(self.H0)
        self.V = copy.deepcopy(self.V0)
        self.W = copy.deepcopy(self.W0)
        
        # Evaluate the forcing term
        forcing_target = np.zeros([self.N,6]) 
        for n in range(self.N-1):
            forcing_target[n,:] = self.W_des[n,:] - self.alphaz*(self.betaz*self.SE3_logarithmic_map(
                                          self.SE3_error(self.HT,self.H_des[n,:,:])) - self.V_des[n,:])
        
        self.fit_dmp(forcing_target)
        
        return self.H_des
    
    def RBF(self, phase):

        if type(phase) is np.ndarray:
            return np.exp(-self.h*(phase[:,np.newaxis]-self.c)**2)
        else:
            return np.exp(-self.h*(phase-self.c)**2)

    def forcing_function_approx(self,weights,phase):

        BF = self.RBF(phase)
        if type(phase) is np.ndarray:
            return np.dot(BF,weights)*phase/np.sum(BF,axis=1)
        else:
            return np.dot(BF,weights)*phase/np.sum(BF)
    
    def fit_dmp(self,forcing_target):

        phase = np.exp(-self.alphax*np.linspace(0.0,self.T,self.N))
        BF = self.RBF(phase)
        X = BF*phase[:,np.newaxis]/np.sum(BF,axis=1)[:,np.newaxis]
        
        regcoef = 0.01 # regularization coefficient
        
        self.weights = np.zeros([self.N_bf,6])
        
        for d in range(6):         
            self.weights[:,d] = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,(X)) + \
                                regcoef * np.eye(X.shape[1])),X.T),forcing_target[:,d].T)

    def SE3_gradient(self,H):

        V = np.zeros([H.shape[0], 6])

        V[0,:] = self.SE3_logarithmic_map(self.SE3_error(H[1,:,:], H[0,:,:])) / self.dt
        for n in range(1, H.shape[0]-1):
            V[n,:] = self.SE3_logarithmic_map(self.SE3_error(H[n+1,:,:], H[n-1,:,:])) / (2.0*self.dt)
        V[-1,:] = self.SE3_logarithmic_map(self.SE3_error(H[-1,:,:], H[-2,:,:])) / self.dt

        return V

    def SE3_error(self,H2,H1):

        # invert H1
        Rotmat = H1[:3,:3]
        p = H1[:3,3]

        H1_inv = np.zeros([4,4])
        H1_inv[:3,:3] = Rotmat.T
        H1_inv[:3,3] = - np.dot(Rotmat.T,p)
        H1_inv[3,3] = 1.0

        return np.matmul(H2,H1_inv) 

    def skew(self,w):
        return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

    def SE3_logarithmic_map(self,H):

        Rotmat = H[:3,:3]
        u = H[:3,3]

        theta = np.arccos(0.5*np.trace(Rotmat) - 0.5)
        A = np.sin(theta)/theta
        B = (1.0 - np.cos(theta))/(theta**2)

        w = (theta/(2.0*np.sin(theta))) * np.array([Rotmat[2,1]-Rotmat[1,2],Rotmat[0,2]-Rotmat[2,0],Rotmat[1,0]-Rotmat[0,1]])
        
        # Taylor series expansion
        # w = (0.5 + ((theta**2)/12) + (7.0*(theta**4)/720.0) + (31.0*(theta**6)/30240.0)) * np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])

        U_inv = np.eye(3) - 0.5*self.skew(w) + (1.0/(theta**2))*(1.0-(A/(2.0*B)))*(np.dot(self.skew(w),self.skew(w)))

        V = np.hstack((w,np.dot(U_inv,u)))

        return V

    def SE3_exponential_map(self,V):

        w = V[:3] # rotation
        u = V[3:] # translation

        theta = np.linalg.norm(w) # screw angle

        # Taylor series expansion (for small theta)
        # A = 1.0 - ((theta**2)/6.0) + ((theta**4)/120.0)
        # B = 0.5 - ((theta**2)/24.0) + ((theta**4)/720.0)
        # C = ((theta**2)/6.0) - ((theta**4)/120.0) + ((theta**6)/5040)

        A = np.sin(theta)/theta
        B = (1.0 - np.cos(theta))/(theta**2)
        C = (1 - A)/(theta**2) 

        Rotmat = np.eye(3) + A*self.skew(w) + B*(np.dot(self.skew(w),self.skew(w)))
        V = np.eye(3) + B*self.skew(w) + C*(np.dot(self.skew(w),self.skew(w)))

        H = np.zeros([4,4])
        H[:3,:3] = Rotmat
        H[:3,3] = np.dot(V,u)
        H[3,3] = 1.0

        return H

    def reset(self):
        
        self.phase = 1.0
        
        self.H = copy.deepcopy(self.H0)
        self.V = copy.deepcopy(self.V0)
        self.W = copy.deepcopy(self.W0)

    def step(self, external_wrench=None):
        
        if external_wrench is None:
            external_wrench = np.zeros(6)
        
        self.phase += (-self.alphax * self.phase) * (self.T/self.N)
        forcing_term = self.forcing_function_approx(self.weights,self.phase)

        self.W = self.alphaz*(self.betaz*self.SE3_logarithmic_map(
            self.SE3_error(self.HT,self.H)) - self.V) + forcing_term + external_wrench        
        self.V += self.W * self.dt
        self.H = np.matmul(self.SE3_exponential_map(self.V*self.dt),self.H)

        return copy.deepcopy(self.H), copy.deepcopy(self.V), copy.deepcopy(self.W)

    def rollout(self):

        H_rollout = np.zeros([self.N,4,4])
        V_rollout = np.zeros([self.N,6])
        W_rollout = np.zeros([self.N,6])

        H_rollout[0,:,:] = self.H0
        V_rollout[0,:] = self.V0
        W_rollout[0,:] = self.W0
        
        phase = np.exp(-self.alphax*np.linspace(0.0,self.T,self.N))
        
        for n in range(1,self.N):

            forcing_term = self.forcing_function_approx(self.weights,phase[n-1])
            W_rollout[n,:] = self.alphaz*(self.betaz*self.SE3_logarithmic_map(
                self.SE3_error(self.HT,H_rollout[n-1,:,:])) - V_rollout[n-1,:]) + \
                forcing_term
            
            V_rollout[n,:] = V_rollout[n-1,:] + W_rollout[n,:]*self.dt
            H_rollout[n,:,:] = np.matmul(self.SE3_exponential_map(V_rollout[n,:]*self.dt),H_rollout[n-1,:,:])
        
        return H_rollout, V_rollout, W_rollout


# Test

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R

    N = 1000
    dt = 0.001

    # Translation component
    x_demo = (np.sin(1*np.pi*np.linspace(0,1,N)))**2
    y_demo = (np.sin(0.5*np.pi*np.linspace(0,1,N)))**3
    z_demo = np.zeros(x_demo.shape[0])

    # Rotation component
    yaw_demo =  np.arctan2(y_demo,x_demo)
    roll_demo = (np.sin(1*np.pi*np.linspace(0,1,N)))**2
    pitch_demo = -(np.sin(1*np.pi*np.linspace(0,1,N)))**2

    # Demo transformation matrices 
    H_demo = np.zeros([x_demo.shape[0],4,4])

    for i in range(N):
        H_demo[i,:3,:3] = R.from_euler('xyz',[roll_demo[i],pitch_demo[i],yaw_demo[i]]).as_matrix()
        H_demo[i,:3,3] = [x_demo[i],y_demo[i],z_demo[i]]
        H_demo[i,3,3] = 1.0


    # Fit the DMP
    dmp = SE3_DMP(N_bf=100,dt=dt)
    _ = dmp.imitate(H_demo)


    # Rollout mode
    H_dmp, _, _ = dmp.rollout()

    fig = plt.figure(figsize=(9,3))
    plt.subplot(121)
    plt.plot(H_demo[:,:3,3],label='demo')
    plt.plot(H_dmp[:,:3,3],'--',label='rollout')
    plt.title('Translation')
    # plt.legend()
    
    plt.subplot(122)
    plt.plot(H_demo[:,:3,:3].reshape(H_demo.shape[0],-1),label='demo')
    plt.plot(H_dmp[:,:3,:3].reshape(H_demo.shape[0],-1),'--',label='rollout')
    plt.title('Rotation')
    # plt.legend()
    plt.suptitle('Rollout mode')
    plt.tight_layout()
    plt.show()


    # Feedback mode (w/o external wrench)
    dmp.reset()
    H_list = []
    for i in range(N):
        H_step, _, _ = dmp.step()
        H_list.append(H_step)

    fig = plt.figure(figsize=(9,3))
    plt.subplot(121)
    plt.plot(H_demo[:,:3,3],label='demo')
    plt.plot(np.array(H_list)[:,:3,3],'--',label='rollout')
    plt.title('Translation')
    # plt.legend()
    
    plt.subplot(122)
    plt.plot(H_demo[:,:3,:3].reshape(H_demo.shape[0],-1),label='demo')
    plt.plot(np.array(H_list)[:,:3,:3].reshape(np.array(H_list).shape[0],-1),'--',label='rollout')
    plt.title('Rotation')
    # plt.legend()
    plt.suptitle('Feedback mode (no external wrench)')
    plt.tight_layout()
    plt.show()


    # Feedback mode (w/ external wrench)
    dmp.reset()
    H_list = []
    for i in range(N):
        if i in range(10,15):
            H_step, _, _ = dmp.step(external_wrench=[1000,-100,800,1000,0,400])
        else:
            H_step, _, _ = dmp.step()
        H_list.append(H_step)

    fig = plt.figure(figsize=(9,3))
    plt.subplot(121)
    plt.plot(H_demo[:,:3,3],label='demo')
    plt.plot(np.array(H_list)[:,:3,3],'--',label='rollout')
    plt.title('Translation')
    # plt.legend()
    
    plt.subplot(122)
    plt.plot(H_demo[:,:3,:3].reshape(H_demo.shape[0],-1),label='demo')
    plt.plot(np.array(H_list)[:,:3,:3].reshape(np.array(H_list).shape[0],-1),'--',label='rollout')
    plt.title('Rotation')
    # plt.legend()
    plt.suptitle('Feedback mode w/ external wrench')
    plt.tight_layout()
    plt.show()