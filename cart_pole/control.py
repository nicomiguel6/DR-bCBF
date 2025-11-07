"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module containing control laws.

"""

import numpy as np
from scipy.linalg import solve_continuous_are


class Control:
    def setupControl(
        self,
    ) -> None:

        # Control limits
        self.u_max = 5 # N
        self.u_bounds = [-self.u_max, self.u_max]
        self.beta = 1

        mp = self.mp
        mk = self.mc
        g = self.g
        lp = self.l
        mt = self.mp + self.mc

        a = g/(lp*(4.0/3 - mp/(mp+mk)))
        self.A = np.array([[0, 1, 0, 0],
                    [0, 0, a, 0],
                    [0, 0, 0, 1],
                    [0, 0, a, 0]])

        # input matrix
        b = -1/(lp*(4.0/3 - mp/(mp+mk)))
        self.B = np.array([[0], [1/mt], [0], [b]])

        self.Q = np.diag([1, 10, 1, 1])
        self.R = np.eye(1)

        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = -np.linalg.inv(self.B.T @ self.P @ self.B + self.R) @ (self.B.T @ self.P @ self.A)


    def primaryControl(self, x_curr, t):
        """
        Primary controller producing desired control at each step.

        """
        u_des = self.u_max
        return u_des

    def backupControl(self, x):
        """
        Safe backup controller. Feedback Control using linearized model.

        """
        u_b = self.K @ x
        return u_b

    def lambdaScore(self, x, hi_min):

        # assume flow using backup controller has been propagated already

        # assume array of h and hb evaluated across flow has been evaluated

        # assume self.hi_max has been calculated already for state x

        return 1 - np.exp(-self.beta * hi_min)  # / np.max([x[1], 0.0001]))

    def blendInputs(self, x, u_des, u_backup, hi_min):

        # Calculate lambda_score
        lambda_score = self.lambdaScore(x, hi_min)

        return (
            np.clip(
                lambda_score * u_des + (1 - lambda_score) * u_backup,
                self.u_bounds[0],
                self.u_bounds[1],
            ),
            lambda_score,
        )
