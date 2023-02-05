import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

"""
Origin sourse of GP code is FedCor paper.
"""

class GPR(nn.Module):
    def __init__(self,
        num_users,
        loss_type='MML',
        init_noise=0.01) -> None:
        super(GPR, self).__init__()

        self.num_users = num_users
        self.loss_type = loss_type
        self.init_noise = init_noise

        self.noise = Parameter(torch.Tensor([init_noise]))
        self.mu = torch.zeros(num_users).detach()
        self.loss_stat = torch.ones(num_users).detach()
        self.discount = torch.ones(num_users).detach()

        self.device = self.noise.device

    def predict_loss(self, index, value, priori_idx, posteriori_idx, noisy=None):
        index = torch.Tensor(index)
        value = torch.Tensor(value)

        mu, sigma = self.get_posteriori(index[priori_idx], value[priori_idx])
        noise = 0.1
        
        while True:
            try:
                cov_matrix = sigma[posteriori_idx,:][:,posteriori_idx] + noise*torch.eye(len(posteriori_idx))
                pdist = MultivariateNormal(
                    loc=mu[posteriori_idx], 
                    covariance_matrix=cov_matrix)
                break
            except ValueError:
                noise *= 10
                if noise > 100:
                    raise Exception("Cannot satisfy positive definiteness property")
        
        proxy_loss = -pdist.log_prob(value[posteriori_idx])
        proxy_loss = proxy_loss.detach().item()
        
        return proxy_loss, mu, sigma

    def update_loss(self, index, value):
        self.loss_stat[index] = torch.Tensor(value).to(self.device)

    def update_discount(self, index, gamma=0.9):
        self.discount[index] *= gamma

    def get_losstype(self):
        return self.loss_type

    def get_noise(self, noisy):
        return torch.diag(noisy).to(self.device)*(self.noise**2)

    def get_posteriori(self, index, value, noisy=None):
        index = index.to(self.device).long()
        value = value.to(self.device)
        if noisy is None:
            noisy = torch.ones(len(index)).to(self.device)
        else:
            noisy = noisy.to(self.device)
        covariance = self.Covariance()

        sigma_inv = torch.inverse(covariance[index,:][:,index]+self.get_noise(noisy))
        mu = self.mu.to(self.device)+((covariance[:,index]@(sigma_inv))@((value-self.mu[index].to(self.device)).unsqueeze(1))).squeeze()
        sigma = covariance

        return mu.detach(), sigma.detach()

    def reset_discount(self):
        self.discount = torch.ones(self.num_users).detach()

    def get_logmarginallikelihood(self, index, value, noisy=None):
        #Calculate the log marginal likelihood of the given data
        #data: given in the form [index,loss,noisy = {0,1}]
        #return log(p(loss|mu,sigma,relation,sigma_n))
        
        index = index.to(self.device).long()
        value = value.to(self.device)
        if noisy is None:
            noisy = torch.ones(len(index)).to(self.device)
        else:
            noisy = noisy.to(self.device)

        mu = self.mu[index].to(self.device)
        Sigma = self.Covariance(index)+self.get_noise(noisy)
        distribution = MultivariateNormal(loc = mu,covariance_matrix = Sigma)
        
        ret = distribution.log_prob(value)

        return ret

    def select_clients(self,
        number=10,
        loss_power=0.5,
        epsilon=0.0,
        discount_method='loss',
        weights=None,
        Dynamic=False,
        Dynamic_TH=0.0):

        def max_loss_decrease_client(client_group,
            Sigma,
            power,
            discount_method,
            weights):
            #Calculate the loss decrease of each client in the client group
            Sigma_valid = Sigma[:,client_group]
            Diag_valid = 1.0/(torch.diagonal(Sigma[:,client_group][client_group,:])+self.noise**2)
            Diag_valid = -Diag_valid*torch.sqrt(torch.diagonal(Sigma[:,client_group][client_group,:]))
            if discount_method=='loss':
                Diag_valid = Diag_valid*torch.pow(self.loss_stat[client_group],power)
            elif discount_method=='time':
                Diag_valid = Diag_valid*self.discount[client_group]

            if weights is None:
                total_loss_decrease = torch.sum(Sigma_valid,dim=0)*Diag_valid
            else:
                total_loss_decrease = torch.sum(torch.Tensor(weights).reshape([self.num_users,1])*Sigma_valid,dim=0)*Diag_valid

            mld,idx = torch.min(total_loss_decrease,0)
            idx = idx.item()
            selected_idx = client_group[idx]
            p_Sigma = Sigma-Sigma[:,selected_idx:selected_idx+1].mm(Sigma[selected_idx:selected_idx+1,:])/(Sigma[selected_idx,selected_idx]+self.noise**2)

            return selected_idx, p_Sigma, mld.item()

        # mu = self.mu
        prob = np.random.rand(1)[0]
        if prob<epsilon:
            # use epsilon-greedy
            return None
        else:
            Sigma = self.Covariance()
            remain_clients = list(range(self.num_users))
            selected_clients = []
            for _ in range(number):
                idx, Sigma, total_loss_decrease = max_loss_decrease_client(remain_clients,Sigma,loss_power,discount_method,weights)
                if Dynamic and -total_loss_decrease<Dynamic_TH:
                    break
                selected_clients.append(idx)
                remain_clients.remove(idx)

            return selected_clients


class PolyKernel(nn.Module):
    def __init__(self,
        order = 1,
        normal = False) -> None:
        super(PolyKernel, self).__init__()

        self.order = order
        self.normal = normal

        # variance parameter
        self.sigma_f = Parameter(torch.Tensor([1.0]))

    def forward(self, xs):
        k = xs.transpose(0,1) @ xs
        if self.normal:
            x_size = xs.size(1)
            one = torch.ones(x_size, x_size).to(self.sigma_f)
            A = torch.sum(xs**2, dim=0, keepdim=True) * one
            norm = torch.sqrt(A) * torch.sqrt(A.transpose(0,1))
            k = k / norm
            return torch.pow(k, self.order) * self.sigma_f ** 2
        else:
            return torch.pow(k, self.order)


class Kernel_GPR(GPR):
    """
    A GPR class with covariance defined by a kernel function

    Parameters:
        Projection.PMatrix: A Matrix that projects index (in a one-hot vector form)
                            into a low-dimension space. 
                            In fact each column of this matrix corresponds to the location 
                            of that user in the low-dimension space.
        Kernel.sigma_f: Diagonal of covariance matrix, which reveals the uncertainty 
                        priori on each user.We assume the same uncertainty before sampling.
         
        noise: Standard Deviation of sample noise (sigma_n). 
               This noise is caused by averaging weights, 
               and we assume the same noise for all clients.
        
        Total number of parameters is num_users x dimension + 2
    """
    def __init__(self,
        num_users,
        dimension = 10,
        init_noise=0.01,
        loss_type = 'MML',
        kernel=PolyKernel,
        **Kernel_Arg) -> None:

        class IndexProjection(nn.Module):
            def __init__(self, 
                num_users,
                dimension = 10):
                super(IndexProjection, self).__init__()
                self.PMatrix = Parameter(torch.randn(dimension, num_users)/np.sqrt(dimension))

            def forward(self, index):
                return self.PMatrix[:, index]

        super(Kernel_GPR, self).__init__(num_users, loss_type, init_noise)

        self.Projection = IndexProjection(num_users, dimension)
        self.Kernel = kernel(**Kernel_Arg)

    def set_parameters(self,
        mu=None,
        proj=None,
        sigma=None,
        noise=None):
        if mu is not None:
            self.mu = mu
        if proj is not None:
            self.Projection.PMatrix.data = proj
        if sigma is not None:
            self.Kernel.sigma_f.data = sigma
        if noise is not None:
            self.noise.data = noise
        
    def Covariance(self, idx=None):
        if idx is None:
            idx = list(range(self.num_users))
        xs = self.Projection(idx)
        return self.Kernel(xs)

    def Parameters(self):
        proj_params = [self.Projection.PMatrix,]
        sigma_params = [self.Kernel.sigma_f, self.noise] if hasattr(self.Kernel,'sigma_f') else [self.noise,]
        return proj_params, sigma_params


def TrainGPR(gpr,
    data,
    method = None,
    matrix_lr=1e-4,
    sigma_lr=1e-4,
    gamma=0.9,
    max_epoches=100, 
    schedule_lr=False, 
    schedule_t=None, 
    schedule_gamma=0.1,
    verbose=True):
    """
    Train hyperparameters(Covariance,noise) of GPR
    data : In shape as [Group,index,value,noise]
    method : {'MML','LOO','NNP'}
        MML:maximize log marginal likelihood
        LOO:maximize Leave-One-Out cross-validation predictive probability 
    """
    #if method is not None:
    #    gpr.loss_type = method
    #method = gpr.get_losstype()

    matrix_params, sigma_params = gpr.Parameters()

    optimizer = torch.optim.Adam([{'params':matrix_params,'lr':matrix_lr},
                                  {'params':sigma_params,'lr':sigma_lr}], lr=matrix_lr, weight_decay=0.0)
    if schedule_lr:
        lr_scd = torch.optim.lr_scheduler.MultiStepLR(optimizer,schedule_t,gamma = schedule_gamma)

    for epoch in range(max_epoches):
        gpr.zero_grad()
        loss = 0.0
        for group in range(len(data)):
            indexs = torch.Tensor(data[group][:,0])
            values = torch.Tensor(data[group][:,1])
            loss = loss*gamma - gpr.get_logmarginallikelihood(indexs, values)
            """
            if method == 'LOO':
                loss = loss*gamma - gpr.Log_LOO_Predictive_Probability(data[group])
            elif method == 'MML': # this is default
                loss = loss*gamma - gpr.Log_Marginal_Likelihood(data[group])
            elif method == 'NNP':
                loss = loss*gamma + gpr.Log_NonNoise_Predictive_Error(data[group])
            else:
                raise RuntimeError("Not supported training method!!")
            """
        loss.backward()
        optimizer.step()

        if epoch%10==0 and verbose:
            desc = "Train_Epoch:{}\t|Noise:{:.4f}\t|Sigma:{:.4f}\t|Loss:{:.4f}".format(
                epoch,
                gpr.noise.detach().item(),
                torch.mean(torch.diagonal(gpr.Covariance())).detach().item(),
                loss.item())
            print(desc)
        if schedule_lr:
            lr_scd.step()
            
    return loss.item() 