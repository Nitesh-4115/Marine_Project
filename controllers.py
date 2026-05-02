import numpy as np

class RBFPDSMCController:
    """ RBF Neural Network PD Sliding Mode Control """
    def __init__(self, M, num_hidden=9):
        self.M = M
        self.kp = np.diag([0.2, 0.2, 0.2, 0.2])
        self.kd = np.diag([0.2, 0.2, 0.2, 0.2])
        self.K = np.diag([1.0, 1.0, 1.0, 1.0])
        
        self.prev_e = np.zeros(4)
        
        self.num_hidden = num_hidden
        self.W = np.zeros((num_hidden, 4))
        self.centers = np.linspace(-3, 3, num_hidden)
        self.width = 1.5
        self.Gamma = 0.5
        
    def get_h(self, x):
        h = np.zeros(self.num_hidden)
        norm_x = np.linalg.norm(x)
        for i in range(self.num_hidden):
            h[i] = np.exp(- (norm_x - self.centers[i])**2 / (2 * self.width**2))
        return h

    def compute_control(self, eta, v, eta_d, eta_d_dot, eta_d_ddot, dt):
        e = eta_d - eta
        e[3] = (e[3] + np.pi) % (2 * np.pi) - np.pi
        
        e_dot = (e - self.prev_e) / dt
        self.prev_e = e
        
        # PD sliding surface
        s = self.kp @ e + self.kd @ e_dot
        
        h = self.get_h(v)
        f_hat = self.W.T @ h
        
        self.W += self.Gamma * np.outer(h, s) * dt
        
        psi = eta[3]
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi),  np.cos(psi), 0, 0],
            [0,            0,           1, 0],
            [0,            0,           0, 1]
        ])
        R_inv = np.linalg.inv(R)
        
        tau_eq = self.M @ R_inv @ (eta_d_ddot + self.kp @ e_dot)
        tau_robust = self.M @ R_inv @ (self.K @ np.tanh(s / 0.1))
        
        return tau_eq + tau_robust + f_hat, s, f_hat

class RBFPIDSMCController:
    """ RBF Neural Network PID Sliding Mode Control """
    def __init__(self, M, num_hidden=9):
        self.M = M
        self.kp = np.diag([0.2, 0.2, 0.2, 0.2])
        self.ki = np.diag([0.01, 0.01, 0.01, 0.01])
        self.kd = np.diag([0.2, 0.2, 0.2, 0.2])
        self.K = np.diag([1.0, 1.0, 1.0, 1.0])
        
        self.int_e = np.zeros(4)
        self.prev_e = np.zeros(4)
        
        self.num_hidden = num_hidden
        self.W = np.zeros((num_hidden, 4))
        self.centers = np.linspace(-3, 3, num_hidden)
        self.width = 1.5
        self.Gamma = 0.5
        
    def get_h(self, x):
        h = np.zeros(self.num_hidden)
        norm_x = np.linalg.norm(x)
        for i in range(self.num_hidden):
            h[i] = np.exp(- (norm_x - self.centers[i])**2 / (2 * self.width**2))
        return h

    def compute_control(self, eta, v, eta_d, eta_d_dot, eta_d_ddot, dt):
        e = eta_d - eta
        e[3] = (e[3] + np.pi) % (2 * np.pi) - np.pi
        
        self.int_e += e * dt
        e_dot = (e - self.prev_e) / dt
        self.prev_e = e
        
        # PID sliding surface
        s = self.kp @ e + self.ki @ self.int_e + self.kd @ e_dot
        
        h = self.get_h(v)
        f_hat = self.W.T @ h
        
        self.W += self.Gamma * np.outer(h, s) * dt
        
        psi = eta[3]
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi),  np.cos(psi), 0, 0],
            [0,            0,           1, 0],
            [0,            0,           0, 1]
        ])
        R_inv = np.linalg.inv(R)
        
        tau_eq = self.M @ R_inv @ (eta_d_ddot + self.kp @ e_dot + self.ki @ e)
        tau_robust = self.M @ R_inv @ (self.K @ np.tanh(s / 0.1))
        
        return tau_eq + tau_robust + f_hat, s, f_hat

class DLNNSMCController:
    """ Double-Loop PID-Type Neural Network Sliding Mode Control """
    def __init__(self, M, num_hidden=15):
        self.M = M
        # Outer loop (kinematic)
        self.kp_outer = np.diag([0.4, 0.4, 0.4, 0.4])
        self.ki_outer = np.diag([0.02, 0.02, 0.02, 0.02])
        self.int_e_eta = np.zeros(4)
        
        # Inner loop (dynamic)
        self.c_inner = np.diag([0.8, 0.8, 0.8, 0.8])
        self.int_e_v = np.zeros(4)
        self.K = np.diag([1.5, 1.5, 1.5, 1.5])
        
        # RBF Neural Network
        self.num_hidden = num_hidden
        self.W = np.zeros((num_hidden, 4))
        self.centers = np.linspace(-5, 5, num_hidden)
        self.width = 2.0
        self.Gamma = 2.0
        
        self.prev_v_d = np.zeros(4)
        
    def get_h(self, x):
        h = np.zeros(self.num_hidden)
        norm_x = np.linalg.norm(x)
        for i in range(self.num_hidden):
            h[i] = np.exp(- (norm_x - self.centers[i])**2 / (2 * self.width**2))
        return h

    def compute_control(self, eta, v, eta_d, eta_d_dot, eta_d_ddot, dt):
        # 1. Outer Kinematic Loop
        e_eta = eta_d - eta
        e_eta[3] = (e_eta[3] + np.pi) % (2 * np.pi) - np.pi
        self.int_e_eta += e_eta * dt
        
        # Virtual velocity in earth frame
        eta_dot_d = eta_d_dot + self.kp_outer @ e_eta + self.ki_outer @ self.int_e_eta
        
        psi = eta[3]
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi),  np.cos(psi), 0, 0],
            [0,            0,           1, 0],
            [0,            0,           0, 1]
        ])
        R_inv = np.linalg.inv(R)
        
        # Desired body velocity
        v_d = R_inv @ eta_dot_d
        
        # 2. Inner Dynamic Loop
        e_v = v_d - v
        self.int_e_v += e_v * dt
        
        # Double-loop Sliding surface
        s = e_v + self.c_inner @ self.int_e_v
        
        h = self.get_h(v)
        f_hat = self.W.T @ h
        
        # Adaptive law
        self.W += self.Gamma * np.outer(h, s) * dt
        
        if np.linalg.norm(self.prev_v_d) == 0 and np.linalg.norm(v) == 0:
            self.prev_v_d = v_d.copy()
            
        v_d_dot = (v_d - self.prev_v_d) / dt
        self.prev_v_d = v_d
        
        # Control Law
        tau_eq = self.M @ (v_d_dot + self.c_inner @ e_v)
        tau_robust = self.M @ (self.K @ np.tanh(s / 0.1))
        
        tau = tau_eq + tau_robust + f_hat
        
        return tau, s, f_hat
