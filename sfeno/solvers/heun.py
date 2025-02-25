import torch
import torch.nn as nn
import torch.nn.functional as F


# def heun_solver(x, u_t, dt, nt, ode, n_activity_nodes=None):

class HeunSolver:
    """Heun's ODE solver"""
    def __init__(self):
        pass

    def general_solve(
        self, 
        ode,
        dt,
        nt,
        t0,
        x,
        arg_dict):
        batch_s = x.shape[0]
        
        xs = []
        
        for i in range(nt):
            t = i*dt
            dxdt_current = ode(t, x, arg_dict)
            # print(x.type()) # debug
            # print(dxdt_current.type())
            # print((x + dt * dxdt_current).type()) # debug
            dxdt_next = ode(t + dt, x + dt * dxdt_current, arg_dict)
            x = x + dt * 0.5 * (dxdt_current + dxdt_next)
            xs.append(x)
            # print(f'Shape of x: {x.shape}') # debug

        # print(f'dxdt: {dxdt_current.shape}')
        xs = torch.stack(xs, dim=0)
        return xs

    def solve(self, x, u_t, dt, nt, ode, n_activity_nodes=None):
        batch_s, _ , _ = x.shape
        # print(f'Shape of x: {x.shape}')

        # print(f'u_t: {u_t.shape}')
        xs = []
        n_x = u_t.shape[1]
        n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes

        # state of none active nodes do not change
        dxdt_mask = torch.zeros((batch_s, n_x, 1))
        dxdt_mask[:, :n_activity_nodes] = 1
        # print(dxdt_mask.shape)

        # set u on ode
        ode.perturb(u_t)

        for i in range(nt):
            t = i*dt
            dxdt_current = ode(t, x)
            # print(x.type()) # debug
            # print(dxdt_current.type())
            # print((x + dt * dxdt_current).type()) # debug
            dxdt_next = ode(t + dt, x + dt * dxdt_current)
            x = x + dt * 0.5 * (dxdt_current + dxdt_next) * dxdt_mask
            xs.append(x)
            # print(f'Shape of x: {x.shape}') # debug

        # print(f'dxdt: {dxdt_current.shape}')
        xs = torch.stack(xs, dim=0)
        return xs


# not implemented for now

# def euler_solver(x, u_t, dt, nt, ode, n_activity_nodes=None):
#     """Euler's method"""
#     xs = []
#     n_x = u_t.shape[0]
#     n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
#     dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
#     for _ in range(nt):
#         dxdt_current = ode(x, u_t)
#         x = x + dt * dxdt_current * dxdt_mask
#         xs.append(x)
#     xs = tf.stack(xs, axis=0)
#     return xs
#
#
# def midpoint_solver(x, u_t, dt, nt, ode, n_activity_nodes=None):
#     """Midpoint method"""
#     xs = []
#     n_x = u_t.shape[0]
#     n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
#     dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
#     for _ in range(nt):
#         dxdt_current = ode(x, u_t)
#         dxdt_midpoint = ode(x + 0.5 * dt * dxdt_current, u_t)
#         x = x + dt * dxdt_midpoint * dxdt_mask
#         xs.append(x)
#     xs = tf.stack(xs, axis=0)
#     return xs
#
#
# def rk4_solver(x, u_t, dt, nt, ode, n_activity_nodes=None):
#     """Runge-Kutta method"""
#     xs = []
#     n_x = u_t.shape[0]
#     n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
#     dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
#     for _ in range(nt):
#         k1 = ode(x, u_t)
#         k2 = ode(x + 0.5*dt*k1, u_t)
#         k3 = ode(x + 0.5*dt*k2, u_t)
#         k4 = ode(x + dt*k3, u_t)
#         x = x + dt * (1/6*k1+1/3*k2+1/3*k3+1/6*k4) * dxdt_mask
#         xs.append(x)
#     xs = tf.stack(xs, axis=0)
#     return xs