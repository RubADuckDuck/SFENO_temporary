from torchdiffeq import odeint
import torch


class TorchdiffeqSolver:
    """Torchdiffeq's ODE solver"""
    def __init__(self):
        pass

    def solve(self, x, u_t, dt, nt, ode, n_activity_nodes=None):
        # batch_s, _ , _ = x.shape
        # # print(f'Shape of x: {x.shape}')
        #
        # # print(f'u_t: {u_t.shape}')
        # xs = []
        # n_x = u_t.shape[1]
        # n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
        #
        # # state of none active nodes do not change
        # dxdt_mask = torch.zeros((batch_s,n_x,1))
        # dxdt_mask[:,:n_activity_nodes] = 1
        # # print(dxdt_mask.shape)
        #
        # for _ in range(nt):
        #     dxdt_current = ode(x, u_t)
        #     # print(x.type()) # debug
        #     # print(dxdt_current.type())
        #     # print((x + dt * dxdt_current).type()) # debug
        #     dxdt_next = ode(x + dt * dxdt_current, u_t)
        #     x = x + dt * 0.5 * (dxdt_current + dxdt_next) * dxdt_mask
        #     xs.append(x)
        #     # print(f'Shape of x: {x.shape}')
        #
        # # print(f'dxdt: {dxdt_current.shape}')
        # xs = torch.stack(xs, dim=0)
        #
        # # ======================='''
        '''
        got to convert dt, nt to a 1d array explicity writing all the time stamp
        dt = 0.5, nt = 10
        t = tensor([0,0.5,1,1.5,2,2.5])
        '''
        end_time = dt * nt
        # set u on ode
        ode.perturb(u_t)

        t = torch.range(start=0, end=end_time, step=dt, requires_grad=False)

        xs = odeint(ode, x, t)
        return xs