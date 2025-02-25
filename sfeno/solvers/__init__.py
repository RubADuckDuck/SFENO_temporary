

from sfeno.solvers.heun import HeunSolver
from sfeno.solvers.torchdiffeq import TorchdiffeqSolver


def get_ode_solver(solver_name):
    """Get the ODE solver based on the given solver name"""
    if solver_name.lower() == 'heun':
        return HeunSolver()

    elif solver_name.lower() == 'torchdiffeq':
        return TorchdiffeqSolver()

    raise Exception("Illegal ODE solver. Use [heun, euler, rk4, midpoint]")

    # if args.ode_solver == 'euler':
    #     return euler_solver
    # if args.ode_solver == 'rk4':
    #     return rk4_solver
    # if args.ode_solver == 'midpoint':
    #     return midpoint_solver
