import numpy as np

class AUVModel:
    def __init__(self):
        # AUV physical parameters (mass and inertia)
        self.m = 115.0  # kg
        self.Iz = 1.5   # kg*m^2
        
        # Added mass coefficients
        self.X_udot = -20.0
        self.Y_vdot = -30.0
        self.Z_wdot = -20.0
        self.N_rdot = -0.5
        
        # Rigid body + added mass matrix (4-DOF: surge, sway, heave, yaw)
        self.M11 = self.m - self.X_udot
        self.M22 = self.m - self.Y_vdot
        self.M33 = self.m - self.Z_wdot
        self.M44 = self.Iz - self.N_rdot
        self.M = np.diag([self.M11, self.M22, self.M33, self.M44])
        self.M_inv = np.linalg.inv(self.M)
        
        # Linear hydrodynamic damping
        self.Xu = -5.0
        self.Yv = -10.0
        self.Zw = -5.0
        self.Nr = -2.0
        
        # Quadratic hydrodynamic damping
        self.Xuu = -1.0
        self.Yvv = -5.0
        self.Zww = -1.0
        self.Nrr = -0.5
        
    def get_C_matrix(self, v):
        """Coriolis and centripetal matrix including added mass"""
        u, v_sway, w, r = v[0], v[1], v[2], v[3]
        C = np.zeros((4, 4))
        C[0, 3] = -self.M22 * v_sway
        C[1, 3] = self.M11 * u
        C[3, 0] = self.M22 * v_sway
        C[3, 1] = -self.M11 * u
        return C
        
    def get_D_matrix(self, v):
        """Hydrodynamic damping matrix (linear + quadratic)"""
        u, v_sway, w, r = v[0], v[1], v[2], v[3]
        D = np.diag([-self.Xu - self.Xuu * abs(u),
                     -self.Yv - self.Yvv * abs(v_sway),
                     -self.Zw - self.Zww * abs(w),
                     -self.Nr - self.Nrr * abs(r)])
        return D
        
    def step(self, eta, v, tau, tau_dist, dt):
        """
        Step the AUV model forward by dt.
        eta: [x, y, z, psi] in earth frame
        v: [u, v_sway, w, r] in body frame
        """
        C = self.get_C_matrix(v)
        D = self.get_D_matrix(v)
        
        # Dynamics: M*v_dot + C*v + D*v = tau + tau_dist
        v_dot = self.M_inv @ (tau + tau_dist - C @ v - D @ v)
        v_next = v + v_dot * dt
        v_next = np.clip(v_next, -15.0, 15.0)
        
        # Kinematics: eta_dot = R(psi) * v
        psi = eta[3]
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi),  np.cos(psi), 0, 0],
            [0,            0,           1, 0],
            [0,            0,           0, 1]
        ])
        
        eta_dot = R @ v
        eta_next = eta + eta_dot * dt
        
        # Wrap heading to [-pi, pi]
        eta_next[3] = (eta_next[3] + np.pi) % (2 * np.pi) - np.pi
        
        return eta_next, v_next
