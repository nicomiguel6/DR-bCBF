"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains functions required for propagating dynamics of double integrator.

"""

import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp


class Dynamics:
    def setupDynamics(
        self,
        blending_bool=False,
    ) -> None:

        # Integration options
        self.int_options = {"rtol": 1e-3, "atol": 1e-3}
        self.blending_bool = blending_bool

        # Simulation data
        self.del_t = 0.02  # [sec]
        self.total_steps = int(5 / self.del_t) + 2
        self.curr_step = 0

        # Initial conditions
        self.x0 = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])

        # Disturbances
        self.dw_max = 0.0
        self.omega = 0.25
        self.dv_max = self.omega * self.dw_max

        # Constant
        max_vel = 2  # based on starting x
        self.sup_fcl = np.sqrt(max_vel**2 + 1)

        # Scaling
        self.s = np.array([5, 5, 180, 5, 5, 3])

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

    def propMain_scaled(self, t, x_tilde, u, dist, args):
        """
        Dimensionless ODE callback:
          x_tilde_dot = ( f(x_phys) + g(x_phys) @ u + dist ) / s
        """
        # 1) recover physical state
        x_phys = x_tilde * self.s

        # 2) compute f and g in physical units
        f_phys = self.f_x(x_phys)  # shape (6,)
        g_phys = self.g_x(x_phys)  # shape (6,2)

        # 3) compute RHS in physical units
        rhs_phys = f_phys + g_phys.dot(u) + dist  # shape (6,)

        # 4) return dimensionless derivative
        return rhs_phys / self.s  # shape (6,)

    def computeJacobianSTM(self, x):
        """
        Compute Jacobian of dynamics.
        """
        jac = self.A
        return jac

    def fScaled(self, t, Xtilde, u, dist):
        # 1) Recover the real state
        X = self.s * Xtilde  # element‐wise multiply

        # 2) Call your original f_x and g_x on the real state
        f_real = self.f_x(X)  # shape (6,)
        g_real = self.g_x(X)  # shape (6,2)

        # 3) Build u‐term and disturbance in real units
        gu_real = g_real @ u  # shape (6,)

        # 4) Combine to get Xdot in real units
        Xdot_real = f_real + gu_real + dist

        # 5) Scale back to get d(Xtilde)/dt
        return Xdot_real / self.s  # shape (6,)

    def integrateStateScaled(self, x, u, t_step, dist, options):
        """
        State integrator using propagation function.

        """
        t_step = (0.0, t_step)
        soltn = solve_ivp(
            lambda t, x: self.fScaled(t, x, u, dist),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1]
        return x

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        f = np.array([*x[-3:], 0.0, -9.81, 0.0])
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        g = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [np.sin(x[2] * np.pi / 180), 0.0],
                [np.cos(x[2] * np.pi / 180), 0.0],
                [0.0, -1 / 0.25],
            ]
        )
        return g

    def integrateState(self, x, u, t_step, dist, options):
        """
        State integrator using propagation function.

        """
        t_step = (0.0, t_step)
        args = {}

        x_tilde = x / self.s

        soltn = solve_ivp(
            lambda t, x: self.propMain(t, x, u, dist, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1] * self.s
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

    def propMainBackupBlending_scaled(self, t, xtilde, args):
        """
        Scaled version of propMainBackupBlending for integrateStateBackup.
        xtilde is the dimensionless state; we recover x_phys, run the blending
        ODE, then return dx_tilde/dt = dx_phys/dt / s.
        """
        # 1) recover physical state
        x_phys = xtilde * self.s

        # 2) compute the physical‐state derivative under your existing backup‐blend ODE
        dx_phys = self.propMainBackupBlending(t, x_phys, args)
        #    (propMainBackupBlending should return a length‐6 dx_phys vector)

        # 3) deflate back to dimensionless derivative
        return dx_phys / self.s

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
            xt0 = x / self.s

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
        # return (x[:,].T * self.s).T
        return x

    def disturbanceFun(self, t, x, u, args):
        """
        Process disturbance function, norm bounded by dw_max.

        """
        dist_t = np.array([1, 1, 1, 1, 1, 1])
        # dist_t = np.random.uniform(-np.ones((1, 6)), np.ones((1, 6)))
        # dist_t = np.array(
        #     [np.sin(self.omega * t + np.pi / 4), np.cos(self.omega * t + np.pi / 4)]
        # )
        return (
            (dist_t / (np.linalg.norm(dist_t))) * self.dw_max
            if np.linalg.norm(dist_t) != 0
            else dist_t
        )
