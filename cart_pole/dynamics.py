"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains functions required for propagating dynamics of cart pole.

"""

import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp


class Dynamics:
    def setupDynamics(
        self,
        blending_bool=False,
    ) -> None:
        
        # constants
        self.mp = 0.1 # kg
        self.mc = 1.0 # kg
        self.g = 9.81 # m/s^2
        self.l = 0.5 # m
        self.F_max = 10 # N

        mt = self.mc + self.mp
        self.tf = 3.0

        # Integration options
        self.int_options = {"rtol": 1e-9, "atol": 1e-9}
        self.blending_bool = blending_bool

        # Simulation data
        self.del_t = 0.02  # [sec]
        self.total_steps = int(self.tf / self.del_t) + 2
        self.curr_step = 0

        # Initial conditions
        self.x0 = np.array([0, 0, 0, 0])
        self.theta_max = np.deg2rad(12)
        self.gamma = 2

        # Disturbances
        self.dw_max = 0.2
        self.omega = 0.25
        self.dv_max = self.omega * self.dw_max

        # Constant
        self.theta_ddot_max = -self.F_max/(mt*self.l*(4.0/3.0 - self.mp/mt))
        self.theta_dot_max = 2*self.theta_ddot_max*self.tf
        self.acc_max = (self.mp*self.l*-self.theta_ddot_max + self.F_max)/mt
        self.vel_max = np.sqrt(self.acc_max**2*self.tf**2)

        self.sup_fcl = np.sqrt(self.vel_max**2 + self.acc_max**2 + self.theta_dot_max**2 + self.theta_ddot_max**2)

    def propMain(self, t, x, u, dist, args):
        """
        Propagation function for dynamics with disturbance and STM if applicable.
        Could be optimized (linear system).

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) @ u + dist
        if len(x) > lenx:
            # Construct F
            F = self.computeJacobianSTM(x[:lenx])

            # Extract STM & reshape
            STM = x[lenx:].reshape(lenx, lenx)
            dSTM = F @ STM

            # Reshape back to column
            dSTM = dSTM.reshape(lenx**2)
            dx[lenx:] = dSTM

        return dx

    def computeJacobianSTM(self, x):
        """
        Compute Jacobian of dynamics.
        """
        jac = self.A
        return jac

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.
        x[0] - x
        x[1] - x_dot
        x[2] - theta (rad)
        x[3] - theta_dot (rad/s)

        """

        pos = x[0]
        vel = x[1]
        theta = x[2]
        theta_dot = x[3]

        mt = self.mc + self.mp

        den = self.l*(4.0/3.0 - self.mp*np.cos(theta)**2/mt)

        theta_ddot = (self.g*np.sin(theta) - (self.mp*self.l*theta_dot**2*np.sin(theta)*np.cos(theta)))/den

        x_ddot = (self.mp*self.l/mt)*theta_dot**2*np.sin(theta) - (self.mp*self.l/mt)*np.cos(theta)*theta_ddot

        f = np.array([x[1], x_ddot, x[3], theta_ddot])
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        pos = x[0]
        vel = x[1]
        theta = x[2]
        theta_dot = x[3]

        mt = self.mc + self.mp

        den = self.l*(4.0/3.0 - self.mp*np.cos(theta)**2/mt)

        x_ctrl = (1/mt) - (self.mp*self.l/mt)*np.cos(theta)*(-(np.cos(theta)/mt)/den)
        theta_ctrl = (-(np.cos(theta)/mt)/den)

        g = np.array([[0.0], [x_ctrl], [0.0], [theta_ctrl]])
        
        return g

    def integrateState(self, x, u, t_step, dist, options):
        """
        State integrator using propagation function.

        """
        t_step = (0.0, t_step)
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMain(t, x, u, dist, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1]
        return x

    def propMainBackup(self, t, x, args):
        """
        Propagation function for backup dynamics and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f(x[:lenx]) + self.g(x[:lenx]) * self.backupControl(x[:lenx])

        # Construct F
        F = self.f(x[:lenx])

        # Extract STM & reshape
        STM = x[lenx:].reshape(lenx, lenx)
        dSTM = F @ STM

        # Reshape back to column
        dSTM = dSTM.reshape(lenx**2)
        dx[lenx:] = dSTM

        return dx

    def propMainBackupBlending(self, t, x, args):
        """Propagation function for nominal backup dynamics

        Args:
            t (_type_): _description_
            x (_type_): _description_
            args (_type_): _description_
        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx = self.f_x(x) + self.g_x(x) @ self.backupControl(x)

        return dx

    def integrateStateBackup(self, x, tspan_b, options):
        """
        Propagate backup flow over the backup horizon. Evaluate at discrete points.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        if self.blending_bool is False:
            soltn = solve_ivp(
                lambda t, x: self.propMainBackup(t, x, args),
                t_step,
                x,
                method="RK45",
                rtol=options["rtol"],
                atol=options["atol"],
                t_eval=tspan_b,
            )
        else:
            soltn = solve_ivp(
                lambda t, x: self.propMainBackupBlending(t, x, args),
                t_step,
                x,
                method="RK45",
                rtol=options["rtol"],
                atol=options["atol"],
                t_eval=tspan_b,
            )
        x = soltn.y[:, :]
        return x

    def disturbanceFun(self, t, x, u, args):
        """
        Process disturbance function, norm bounded by dw_max.

        """
        # dist_t = np.array([1, 1])
        # dist_t = np.random.uniform([-1, -1], [1, 1])
        dist_t = np.array(
            [0, 0, 0, 0]
        )
        return (
            (dist_t / (np.linalg.norm(dist_t))) * self.dw_max
            if np.linalg.norm(dist_t) != 0
            else dist_t
        )
