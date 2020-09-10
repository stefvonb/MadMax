import nlopt

NLOPT_VERSION = nlopt.__version__

NLOPT_EXIT_CODES = {
     1: "SUCCESS",
     2: "STOPVAL_REACHED",
     3: "FTOL_REACHED",
     4: "XTOL_REACHED",
     5: "MAXEVAL_REACHED",
     6: "MAXTIME_REACHED",
    -1: "FAILURE",
    -2: "INVALID_ARGS",
    -3: "OUT_OF_MEMORY",
    -4: "ROUNDOFF_LIMITED",
    -5: "FORCED_STOP"
}

NLOPT_ALGORITHMS = {
    0:  {'name': "GN_DIRECT", 'desc': "DIRECT (global, no-derivative)"},
    1:  {'name': "GN_DIRECT_L", 'desc': "DIRECT-L (global, no-derivative)"},
    2:  {'name': "GN_DIRECT_L_RAND", 'desc': "Randomized DIRECT-L (global, no-derivative)"},
    3:  {'name': "GN_DIRECT_NOSCAL", 'desc': "Unscaled DIRECT (global, no-derivative)"},
    4:  {'name': "GN_DIRECT_L_NOSCAL", 'desc': "Unscaled DIRECT-L (global, no-derivative)"},
    5:  {'name': "GN_DIRECT_L_RAND_NOSCAL", 'desc': "Unscaled Randomized DIRECT-L (global, no-derivative)"},
    6:  {'name': "GN_ORIG_DIRECT", 'desc': "Original DIRECT version (global, no-derivative)"},
    7:  {'name': "GN_ORIG_DIRECT_L", 'desc': "Original DIRECT-L version (global, no-derivative)"},
    8:  {'name': "GD_STOGO", 'desc': "StoGO (global, derivative-based)"},
    9:  {'name': "GD_STOGO_RAND", 'desc': "StoGO with randomized search (global, derivative-based)"},
    10: {'name': "LD_LBFGS_NOCEDAL", 'desc': "original L-BFGS code by Nocedal et al. (NOT COMPILED)"},
    11: {'name': "LD_LBFGS", 'desc': "Limited-memory BFGS (L-BFGS) (local, derivative-based)"},
    12: {'name': "LN_PRAXIS", 'desc': "Principal-axis, praxis (local, no-derivative)"},
    13: {'name': "LD_VAR1", 'desc': "Limited-memory variable-metric, rank 1 (local, derivative-based)"},
    14: {'name': "LD_VAR2", 'desc': "Limited-memory variable-metric, rank 2 (local, derivative-based)"},
    15: {'name': "LD_TNEWTON", 'desc': "Truncated Newton (local, derivative-based)"},
    16: {'name': "LD_TNEWTON_RESTART", 'desc': "Truncated Newton with restarting (local, derivative-based)"},
    17: {'name': "LD_TNEWTON_PRECOND", 'desc': "Preconditioned truncated Newton (local, derivative-based)"},
    18: {'name': "LD_TNEWTON_PRECOND_RESTART", 'desc': "Preconditioned truncated Newton with restarting (local, derivative-based)"},
    19: {'name': "GN_CRS2_LM", 'desc': "Controlled random search (CRS2) with local mutation (global, no-derivative)"},
    20: {'name': "GN_MLSL", 'desc': "Multi-level single-linkage (MLSL), random (global, no-derivative)"},
    21: {'name': "GD_MLSL", 'desc': "Multi-level single-linkage (MLSL), random (global, derivative)"},
    22: {'name': "GN_MLSL_LDS", 'desc': "Multi-level single-linkage (MLSL), quasi-random (global, no-derivative)"},
    23: {'name': "GD_MLSL_LDS", 'desc': "Multi-level single-linkage (MLSL), quasi-random (global, derivative)"},
    24: {'name': "LD_MMA", 'desc': "Method of Moving Asymptotes (MMA) (local, derivative)"},
    25: {'name': "LN_COBYLA", 'desc': "COBYLA (Constrained Optimization BY Linear Approximations) (local, no-derivative)"},
    26: {'name': "LN_NEWUOA", 'desc': "NEWUOA unconstrained optimization via quadratic models (local, no-derivative)"},
    27: {'name': "LN_NEWUOA_BOUND", 'desc': "Bound-constrained optimization via NEWUOA-based quadratic models (local, no-derivative)"},
    28: {'name': "LN_NELDERMEAD", 'desc': "Nelder-Mead simplex algorithm (local, no-derivative)"},
    29: {'name': "LN_SBPLX", 'desc': "Sbplx variant of Nelder-Mead (re-implementation of Rowan's Subplex) (local, no-derivative)"},
    30: {'name': "LN_AUGLAG", 'desc': "Augmented Lagrangian method (local, no-derivative)"},
    31: {'name': "LD_AUGLAG", 'desc': "Augmented Lagrangian method (local, derivative)"},
    32: {'name': "LN_AUGLAG_EQ", 'desc': "Augmented Lagrangian method for equality constraints (local, no-derivative)"},
    33: {'name': "LD_AUGLAG_EQ", 'desc': "Augmented Lagrangian method for equality constraints (local, derivative)"},
    34: {'name': "LN_BOBYQA", 'desc': "BOBYQA bound-constrained optimization via quadratic models (local, no-derivative)"},
    35: {'name': "GN_ISRES", 'desc': "ISRES evolutionary constrained optimization (global, no-derivative)"},
    36: {'name': "AUGLAG", 'desc': "Augmented Lagrangian method (needs sub-algorithm)"},
    37: {'name': "AUGLAG_EQ", 'desc': "Augmented Lagrangian method for equality constraints (needs sub-algorithm)"},
    38: {'name': "G_MLSL", 'desc': "Multi-level single-linkage (MLSL), random (global, needs sub-algorithm)"},
    39: {'name': "G_MLSL_LDS", 'desc': "Multi-level single-linkage (MLSL), quasi-random (global, needs sub-algorithm)"},
    40: {'name': "LD_SLSQP", 'desc': "Sequential Quadratic Programming (SQP) (local, derivative)"},
    41: {'name': "LD_CCSAQ", 'desc': "CCSA (Conservative Convex Separable Approximations) with simple quadratic approximations (local, derivative)"},
    42: {'name': "GN_ESCH", 'desc': "ESCH evolutionary strategy"}
}

class NLOPTError(Exception):
    pass

class optimiser(nlopt.opt):
    """
    An optimiser with extra features and a default set up
    Bounds are fixed to a unit hypercube
    """
    def __init__(self, algorithm, dimension, xtol_rel=1e-3, ftol_rel=1e-3, maxtime=60):
        try:
            if isinstance(algorithm, int):
                int_id = algorithm
            elif isinstance(algorithm, str):
                # Search for the integer using the name
                int_id = next(key for key, value in NLOPT_ALGORITHMS.items() if value['name'] == algorithm)
            else:
                raise NLOPTError("Please input either a string or integer as the algorithm name")
            super().__init__(int_id, dimension)
        except (MemoryError, StopIteration):
            raise NLOPTError("Did not understand algorithm name '{}'".format(algorithm))
        except (NotImplementedError, TypeError):
            raise NLOPTError("Problem with inputs for optimiser, please check the number of dimensions")
        self.name = NLOPT_ALGORITHMS[int_id]['name']
        self.set_lower_bounds([0.0 for _ in range(dimension)])
        self.set_upper_bounds([1.0 for _ in range(dimension)])
        self.set_xtol_rel(xtol_rel)
        self.set_ftol_rel(ftol_rel)
        self.set_maxtime(maxtime)
        self.optimised_parameters = None

    @property
    def description(self):
        return self.get_algorithm_name()

    @property
    def dimension(self):
        return self.get_dimension()

    @property
    def scope(self):
        if "AUGLAG" in self.name or "MLSL" in self.name:
            return 'global/local'
        prefix = self.name.split("_")[0]
        if len(prefix) == 2 or len(prefix) == 1:
            if "G" in prefix:
                return 'global'
            elif "L" in prefix:
                return 'local'
    
    @property
    def xopt(self):
        return self.optimised_parameters
        
    def run(self, starting_values):
        try:
            self.optimised_parameters = self.optimize(starting_values)
        except nlopt.RoundoffLimited:
            print("WARNING:\tMaximisation limited by round-off. Returning starting values.")
            self.optimised_parameters = starting_values
