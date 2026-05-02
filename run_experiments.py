import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from auv_environment import AUVModel
from controllers import RBFPDSMCController, RBFPIDSMCController, DLNNSMCController

def deriv(f, t, dt=1e-3):
    return (f(t + dt) - f(t - dt)) / (2 * dt)

def deriv2(f, t, dt=1e-3):
    return (f(t + dt) - 2 * f(t) + f(t - dt)) / (dt ** 2)

def generate_trajectory(t, exp_type):
    if exp_type == 1:
        # Helix (as from paper Eq 70)
        def f(t):
            x = np.sin(0.025 * t) + 25.1
            y = np.cos(0.025 * t) - 1.05
            z = -0.0184 * t - 93.7
            psi = np.arctan2(0.025 * np.cos(0.025 * t), -0.025 * np.sin(0.025 * t))
            return np.array([x, y, z, psi])
    elif exp_type == 2:
        # Sinusoidal (as from paper Eq 72)
        def f(t):
            x = 0.0184 * t + 26.0
            y = np.sin(0.025 * t) - 0.3
            z = 0.0167 * t - 93.7
            psi = 0.1 * np.sin(0.025 * t)
            return np.array([x, y, z, psi])

    eta_d = f(t)
    eta_d_dot = deriv(f, t)
    eta_d_ddot = deriv2(f, t)
    return eta_d, eta_d_dot, eta_d_ddot

def run_experiment_for_controller(exp_id, ctrl_class, initial_eta, duration=300, dt=0.5):
    auv = AUVModel()
    eta = initial_eta.copy()
    v = np.array([0.0, 0.0, 0.0, 0.0])
    
    time_steps = int(duration / dt)
    
    # Induce 15% parameter uncertainty: the controller doesn't know the exact physics
    M_hat = auv.M * 0.85 
    ctrl = ctrl_class(M_hat)
    
    history_eta = []
    history_eta_d = []
    
    for i in range(time_steps):
        t = i * dt
        eta_d, eta_d_dot, eta_d_ddot = generate_trajectory(t, exp_id)
        
        # Realistic ocean currents (low frequency + random noise)
        tau_dist = np.array([
            8.0 * np.sin(0.05*t) + 3.0 * np.cos(0.02*t) + 1.0 * np.random.randn(),
            4.0 * np.cos(0.04*t) + 2.0 * np.sin(0.01*t) + 1.0 * np.random.randn(),
            5.0 * np.sin(0.03*t) + 1.0 * np.random.randn(),
            1.0 * np.sin(0.08*t) + 0.2 * np.random.randn()
        ])
        
        # Add realistic sensor noise (IMU / DVL approximation)
        eta_measured = eta + np.random.randn(4) * np.array([0.02, 0.02, 0.02, 0.005])
        v_measured = v + np.random.randn(4) * np.array([0.01, 0.01, 0.01, 0.002])
        
        tau, _, _ = ctrl.compute_control(eta_measured, v_measured, eta_d, eta_d_dot, eta_d_ddot, dt)
        tau = np.clip(tau, -800, 800) # Realistic physical torque limits
        
        eta, v = auv.step(eta, v, tau, tau_dist, dt)
        
        history_eta.append(eta.copy())
        history_eta_d.append(eta_d.copy())
        
    return np.array(history_eta), np.array(history_eta_d)

def run_and_plot(exp_id):
    print(f"\n--- Running Experiment {exp_id} ---")
    
    # Paper initial conditions (adjusted to start with max error ~ 0.4m to match [-0.5, 0.5] bounds)
    if exp_id == 1:
        # Helix start
        eta_init = np.array([24.7, 0.3, -93.3, -0.1])
    else:
        # Sinusoidal start
        eta_init = np.array([25.6, 0.1, -93.3, -0.1])
        
    duration = 300
    dt = 0.05
    
    print("Running RBFPDSMC...")
    eta_pd, eta_d = run_experiment_for_controller(exp_id, RBFPDSMCController, eta_init, duration, dt)
    print("Running RBFPIDSMC...")
    eta_pid, _ = run_experiment_for_controller(exp_id, RBFPIDSMCController, eta_init, duration, dt)
    print("Running DLNNSMC...")
    eta_our, _ = run_experiment_for_controller(exp_id, DLNNSMCController, eta_init, duration, dt)
    
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. 3D Trajectory (Matching Figure 6 style)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(eta_d[:, 0], eta_d[:, 1], eta_d[:, 2], 'm--', linewidth=2, label='Desired')
    ax.plot(eta_pd[:, 0], eta_pd[:, 1], eta_pd[:, 2], 'r-', linewidth=1.5, label='RBFPDSMC')
    ax.plot(eta_pid[:, 0], eta_pid[:, 1], eta_pid[:, 2], 'g-', linewidth=1.5, label='RBFPIDSMC')
    ax.plot(eta_our[:, 0], eta_our[:, 1], eta_our[:, 2], 'b-', linewidth=2, label='DLNNSMC')
    
    ax.plot([eta_init[0]], [eta_init[1]], [eta_init[2]], 'o', color='orange', markersize=10)
    ax.text(eta_init[0], eta_init[1] + 0.5, eta_init[2], 'AUV start point', fontweight='bold')
    
    ax.set_xlabel('X[m]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y[m]', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z[m]', fontsize=14, fontweight='bold')
    ax.view_init(elev=20, azim=-45)
    
    plt.legend(prop={'size': 14, 'weight': 'bold'})
    plt.grid(True)
    plt.savefig(f'{out_dir}/exp{exp_id}_3d_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Position Errors (2x2 Subplots like Figure 12/17)
    # ---------------------------------------------------------
    time = np.arange(len(eta_d)) * dt
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # (a) x_e
    axs[0, 0].plot(time, eta_d[:, 0] - eta_pd[:, 0], 'r-', linewidth=1.5, label='RBFPDSMC')
    axs[0, 0].plot(time, eta_d[:, 0] - eta_pid[:, 0], 'g-', linewidth=1.5, label='RBFPIDSMC')
    axs[0, 0].plot(time, eta_d[:, 0] - eta_our[:, 0], 'b-', linewidth=2, label='DLNNSMC')
    axs[0, 0].set_title(r'(a) $x_e$ [m]', fontweight='bold')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Error X')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # (b) y_e
    axs[0, 1].plot(time, eta_d[:, 1] - eta_pd[:, 1], 'r-', linewidth=1.5)
    axs[0, 1].plot(time, eta_d[:, 1] - eta_pid[:, 1], 'g-', linewidth=1.5)
    axs[0, 1].plot(time, eta_d[:, 1] - eta_our[:, 1], 'b-', linewidth=2)
    axs[0, 1].set_title(r'(b) $y_e$ [m]', fontweight='bold')
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Error Y')
    axs[0, 1].grid(True)
    
    # (c) z_e
    axs[1, 0].plot(time, eta_d[:, 2] - eta_pd[:, 2], 'r-', linewidth=1.5)
    axs[1, 0].plot(time, eta_d[:, 2] - eta_pid[:, 2], 'g-', linewidth=1.5)
    axs[1, 0].plot(time, eta_d[:, 2] - eta_our[:, 2], 'b-', linewidth=2)
    axs[1, 0].set_title(r'(c) $z_e$ [m]', fontweight='bold')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Error Z')
    axs[1, 0].grid(True)
    
    # (d) psi_e
    # Error in psi (wrapped)
    e_psi_pd = (eta_d[:, 3] - eta_pd[:, 3] + np.pi) % (2 * np.pi) - np.pi
    e_psi_pid = (eta_d[:, 3] - eta_pid[:, 3] + np.pi) % (2 * np.pi) - np.pi
    e_psi_our = (eta_d[:, 3] - eta_our[:, 3] + np.pi) % (2 * np.pi) - np.pi
    
    axs[1, 1].plot(time, e_psi_pd, 'r-', linewidth=1.5)
    axs[1, 1].plot(time, e_psi_pid, 'g-', linewidth=1.5)
    axs[1, 1].plot(time, e_psi_our, 'b-', linewidth=2)
    axs[1, 1].set_title(r'(d) $\psi_e$ [rad]', fontweight='bold')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel(r'Error $\psi$')
    axs[1, 1].grid(True)
    
    # --- RMSE Calculation ---
    rmse_pd = np.sqrt(np.mean((eta_d[:, :3] - eta_pd[:, :3])**2, axis=0))
    rmse_pid = np.sqrt(np.mean((eta_d[:, :3] - eta_pid[:, :3])**2, axis=0))
    rmse_our = np.sqrt(np.mean((eta_d[:, :3] - eta_our[:, :3])**2, axis=0))
    
    print(f"\n[Experiment {exp_id} RMSE Metrics]")
    print(f"RBFPDSMC  -> X: {rmse_pd[0]:.4f}m, Y: {rmse_pd[1]:.4f}m, Z: {rmse_pd[2]:.4f}m")
    print(f"RBFPIDSMC -> X: {rmse_pid[0]:.4f}m, Y: {rmse_pid[1]:.4f}m, Z: {rmse_pid[2]:.4f}m")
    print(f"DLNNSMC   -> X: {rmse_our[0]:.4f}m, Y: {rmse_our[1]:.4f}m, Z: {rmse_our[2]:.4f}m")

    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/exp{exp_id}_position_errors.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 3. Combined 3D Trajectory GIF Generation (Enlarged)
    # ---------------------------------------------------------
    print("Generating 3D animated GIF...")
    # Vastly increased figure size for visibility
    fig_ani = plt.figure(figsize=(16, 14))
    ax_ani = fig_ani.add_subplot(111, projection='3d')
    
    ax_ani.set_xlim(np.min(eta_d[:, 0])-1, np.max(eta_d[:, 0])+1)
    ax_ani.set_ylim(np.min(eta_d[:, 1])-1, np.max(eta_d[:, 1])+1)
    ax_ani.set_zlim(np.min(eta_d[:, 2])-1, np.max(eta_d[:, 2])+1)
    
    ax_ani.set_xlabel('X[m]', fontsize=14, fontweight='bold')
    ax_ani.set_ylabel('Y[m]', fontsize=14, fontweight='bold')
    ax_ani.set_zlabel('Z[m]', fontsize=14, fontweight='bold')
    ax_ani.view_init(elev=25, azim=-40)
    
    # Static desired trajectory line
    ax_ani.plot(eta_d[:, 0], eta_d[:, 1], eta_d[:, 2], 'm--', alpha=0.6, linewidth=2, label='Desired')
    
    line_pd, = ax_ani.plot([], [], [], 'r-', linewidth=1.5, label='RBFPDSMC')
    pt_pd, = ax_ani.plot([], [], [], 'ro', markersize=6)
    
    line_pid, = ax_ani.plot([], [], [], 'g-', linewidth=1.5, label='RBFPIDSMC')
    pt_pid, = ax_ani.plot([], [], [], 'go', markersize=6)
    
    line_our, = ax_ani.plot([], [], [], 'b-', linewidth=2.5, label='DLNNSMC')
    pt_our, = ax_ani.plot([], [], [], 'bo', markersize=8)
    
    ax_ani.legend(prop={'size': 16, 'weight': 'bold'})
    
    def update(frame):
        idx = frame * 40
        if idx >= len(eta_d):
            idx = len(eta_d) - 1
            
        line_pd.set_data(eta_pd[:idx, 0], eta_pd[:idx, 1])
        line_pd.set_3d_properties(eta_pd[:idx, 2])
        pt_pd.set_data([eta_pd[idx, 0]], [eta_pd[idx, 1]])
        pt_pd.set_3d_properties([eta_pd[idx, 2]])
        
        line_pid.set_data(eta_pid[:idx, 0], eta_pid[:idx, 1])
        line_pid.set_3d_properties(eta_pid[:idx, 2])
        pt_pid.set_data([eta_pid[idx, 0]], [eta_pid[idx, 1]])
        pt_pid.set_3d_properties([eta_pid[idx, 2]])
        
        line_our.set_data(eta_our[:idx, 0], eta_our[:idx, 1])
        line_our.set_3d_properties(eta_our[:idx, 2])
        pt_our.set_data([eta_our[idx, 0]], [eta_our[idx, 1]])
        pt_our.set_3d_properties([eta_our[idx, 2]])
        
        return line_pd, pt_pd, line_pid, pt_pid, line_our, pt_our

    ani = FuncAnimation(fig_ani, update, frames=len(eta_d)//40, blit=False)
    # Generate high quality GIF
    ani.save(f'{out_dir}/exp{exp_id}_combined_trajectory.gif', writer=PillowWriter(fps=15))
    plt.close(fig_ani)

if __name__ == '__main__':
    for i in [1, 2]:
        run_and_plot(i)
    print("Replication finished and GIFs generated.")
