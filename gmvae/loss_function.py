import torch
import torch.nn.functional as F
import numpy as np

class Criterion:
    def __init__(self):
        pass

    def gmvae_loss(self, params, weights, reduction='none'):
        # get parameters from model
        x = params['x']
        x_z = params['x_z']
        w_x_mean, w_x_var = params['w_x_mean'], params['w_x_var']
        y_wz = params['y_wz']
        z_x = params['z_x']
        z_x_mean, z_x_var = params['z_x_mean'], params['z_x_var'],
        z_wy_means, z_wy_vars = params['z_wy_means'], params['z_wy_vars']

        # get loss weights from arguments
        rec_wei = weights.get('rec_wei') or 1.
        cond_wei = weights.get('cond_wei') or 1.
        w_wei = weights.get('w_wei') or 1.
        y_wei = weights.get('y_wei') or 1.
        y_thres = weights.get('y_thres') or 0.

        rec_loss = rec_wei * self.binary_cross_entropy(x, x_z)
        conditional_negative_kl = \
            cond_wei * self.conditional_negative_kl(z_x, z_x_mean, z_x_var,
                                                    z_wy_means, z_wy_vars, y_wz)
        w_prior_negative_kl = w_wei * self.gaussian_negative_kl(w_x_mean, w_x_var)
        y_prior_negative_kl = y_wei * self.y_prior_negative_kl(y_wz, thres=y_thres)

        total = torch.cat([rec_loss.view(-1),
                           conditional_negative_kl.view(-1),
                           w_prior_negative_kl.view(-1),
                           y_prior_negative_kl.view(-1)])

        if reduction is 'sum':
            return total.sum()
        return total

    def cross_entropy(self, output, target):
        loss = F.cross_entropy(output, target, reduction='sum')
        loss /= input.shape[0]
        return loss

    def binary_cross_entropy(self, x, x_):
        # Reconstruction loss
        # https://arxiv.org/pdf/1312.6114.pdf -> C.1
        # E_q[log p(x^(i)|z^(i))]=1/LΣ(log p(x_m^(i)|z_m^(i,l)))
        # x, p ~ β(p, x)=p^x+(1-p)^(1-x)
        # log p(x_m^(i)|z_m^(i,l) = log(p_i^x_i+(1-p_i)^(1-x_i))
        #                         = x_i log(p_i)+(1-x_i) log(1-p_i)
        loss = F.binary_cross_entropy(x_, x, reduction='sum')
        loss /= x.shape[0]
        return loss


    def conditional_negative_kl(self, z_x, z_x_mean, z_x_var,
                                z_wy_means, z_wy_vars, y_wz):
        # Conditional loss
        # q(z|x)=N(μ_x,σ_x)
        # logp = −0.5 * { log(det(σ_x^2)) + (z − μ_x)^2 / σ_x^2 }
        # q(z|w,y=1)=N(μ_w,σ_w)
        # logq = −0.5 * { Σπlog(det(σ_w^2)) + Σπ(z − μ_w)^2 / σ_w^2 }
        eps = 1e-10
        logq = -0.5 * (torch.log(z_x_var + eps).sum(1)
                      + (torch.pow(z_x - z_x_mean, 2) / z_x_var).sum(1))
        K = y_wz.shape[-1]
        z_wy = z_x.repeat(1, K).view(z_x.shape[0], K, -1).transpose(1,2)  # (batch_size, z_dim, K)
        logp = -0.5 * (y_wz * torch.log(z_wy_vars + eps).sum(1)
                       + y_wz * (torch.pow(z_wy - z_wy_means, 2) / z_wy_vars).sum(1)).sum(1)
        kl = (logq - logp).mean()
        return kl

    def gaussian_negative_kl(self, mean, var):
        # input: μ_θ(w), (batch_size, w_dim)
        # input: σ_θ(w), (batch_size, w_dim)
        eps = 1e-10
        kl = 0.5 * (var - 1 - torch.log(var) + torch.pow(mean, 2)).sum(-1)
        kl = kl.mean()
        return kl

    def y_prior_negative_kl(self, y_wz, pi=None, thres=0.):
        eps = 1e-10
        k = y_wz.shape[-1]
        kl = -np.log(k) - 1 / k * torch.log(y_wz + eps).sum(-1)
        thres = torch.ones_like(kl) * thres
        kl = torch.max(kl, thres)
        kl = kl.mean() # negative value minimize(kl)
        return kl
