import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

"""
Origin sourse of GP code is FedCor paper.
"""

"""
np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                    np.ones([args.num_users,1])],1)
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

        self.noise = Parameter(torch.tensor(init_noise))
        self.mu = torch.zeros(num_users).detach()
        self.loss_stat = torch.ones(num_users).detach()
        self.discount = torch.ones(num_users).detach()

    def Posteriori(self, data):
        pass

    def predict_loss(self, data, priori_idx, posteriori_idx):
        mu, sigma = self.Posteriori(data[priori_idx,:])
        noise = 0.1
        while True:
            try:
                cov_matrix = sigma[posteriori_idx,:][:,posteriori_idx] + noise*torch.eye(len(posteriori_idx))
                pdist = MultivariateNormal(loc=mu[posteriori_idx], 
                    covariance_matrix=cov_matrix)
                break
            except ValueError:
                noise *= 10
                if noise > 100:
                    raise Exception("Cannot satisfy positive definiteness property")
        proxy_loss = -pdist.log_prob(torch.tensor(data[posteriori_idx,1]).to(mu_p))
        proxy_loss = proxy_loss.detach().item()
        return proxy_loss, mu,sigma

    def select_clients(self):
        pass

    def update_loss(self):
        pass

    def update_discount(self, index, gamma=0.9):
        self.discount[index] *= gamma

    def get_losstype(self):
        return self.loss_type

    def get_posteriori(self):
        data = torch.tensor(data).to(self.noise)
        indexes = data[:,0].long()
        values = data[:,1]
        noisy = data[:,2]
        Cov = self.Covariance()

        sigma_inv = torch.inverse(Cov[indexes,:][:,indexes]+torch.diag(noisy).to(self.noise)*(self.noise**2))
        mu = self.mu.to(self.noise)+((Cov[:,indexes] @ (sigma_inv)) @ ((values-self.mu[indexes].to(self.noise)).unsqueeze(1))).squeeze()
        Sigma = Cov
        return mu.detach(), Sigma.detach()

    def Reset_Discount(self):
        self.discount = torch.ones(self.num_users).detach()

    """
    def Log_Marginal_Likelihood(self, data):
        #Calculate the log marginal likelihood of the given data
        #data: given in the form [index,loss,noisy = {0,1}]
        #return log(p(loss|mu,sigma,relation,sigma_n))
        
        data = torch.tensor(data).to(self.noise)
        indexes = data[:,0].long()
        values = data[:,1]
        noisy = data[:,2]
        mu = self.mu[indexes].to(self.noise)
        Sigma = self.Covariance(indexes)+torch.diag(noisy).to(self.noise)*(self.noise**2)
        distribution = MultivariateNormal(loc = mu,covariance_matrix = Sigma)
        res = distribution.log_prob(values)

        return res

    def Select_Clients(self,number=10,loss_power = 0.5,epsilon = 0.0,discount_method = 'loss',weights = None,Dynamic=False,Dynamic_TH=0.0,verbose = False):
        def max_loss_decrease_client(client_group,Sigma,power = 0.3,discount_method = 'loss',weights = None):
            # print(mu)
            Sigma_valid = Sigma[:,client_group]
            Diag_valid = 1.0/(torch.diagonal(Sigma[:,client_group][client_group,:])+self.noise**2)
            # Diag_valid = Diag_valid*(self.loss_decrease_estimation[client_group].to(self.noise)-mu[client_group])
            Diag_valid = -Diag_valid*torch.sqrt(torch.diagonal(Sigma[:,client_group][client_group,:]))
            if discount_method=='loss':
                Diag_valid = Diag_valid*torch.pow(self.loss_stat[client_group],power)
            elif discount_method=='time':
                Diag_valid = Diag_valid*self.discount[client_group]

            # loss_decrease = Sigma_valid*Diag_valid
            if weights is None:
                total_loss_decrease = torch.sum(Sigma_valid,dim=0)*Diag_valid
            else:
                total_loss_decrease = torch.sum(torch.tensor(weights).reshape([self.num_users,1])*Sigma_valid,dim=0)*Diag_valid
            # total_loss_decrease = torch.sum(loss_decrease,dim = 0)# Add across row
            mld,idx = torch.min(total_loss_decrease,0)
            idx = idx.item()
            selected_idx = client_group[idx]
            p_Sigma = Sigma-Sigma[:,selected_idx:selected_idx+1].mm(Sigma[selected_idx:selected_idx+1,:])/(Sigma[selected_idx,selected_idx]+self.noise**2)
            d_mu = Sigma_valid[:,idx]*Diag_valid[idx]
            # return selected_idx,mu+loss_decrease[:,idx],p_Sigma
            return selected_idx,p_Sigma,mld.item(),d_mu.detach()


        # mu = self.mu
        prob = np.random.rand(1)[0]
        if prob<epsilon:
            # use epsilon-greedy
            return None
        else:
            Sigma = self.Covariance()
            remain_clients = list(range(self.num_users))
            selected_clients = []
            for i in range(number):
                idx,Sigma,total_loss_decrease,loss_decrease = max_loss_decrease_client(remain_clients,Sigma,loss_power,discount_method,weights)
                if Dynamic and -total_loss_decrease<Dynamic_TH:
                    break
                selected_clients.append(idx)
                remain_clients.remove(idx)
    """


class PolyKernel(nn.Module):
    def __init__(self,
        order = 1,
        normal = False) -> None:
        super(PolyKernel, self).__init__()

        self.order = order
        self.normal = normal

        # variance parameter
        self.sigma_f = Parameter(torch.tensor(1.0))

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

"""
gpr = Kernel_GPR(args.num_users,
                 dimension = args.dimension,
                 init_noise=0.01,
                 order = 1, 
                 Normalize = args.poly_norm,
                 kernel=GPR.Poly_Kernel,
                 loss_type= args.train_method)
"""

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
                self.PMatrix = Parameter(torch.randn(dimension, num_users)/torch.sqrt(dimension))

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
            ids = list(range(self.num_users))
        xs = self.Projection(ids)
        return self.Kernel(xs)

    def Parameters(self):
        proj_params = [self.Projection.PMatrix,]
        sigma_params = [self.Kernel.sigma_f, self.noise] if hasattr(self.Kernel,'sigma_f') else [self.noise,]
        return proj_params, sigma_params


def TrainGPR(gpr,
    data,
    method = None,
    lr = 1e-4,
    llr = 1e-4,
    gamma = 0.9,
    max_epoches = 100, 
    schedule_lr = False, 
    schedule_t = None, 
    schedule_gamma = 0.1,
    verbose=True):
    """
    Train hyperparameters(Covariance,noise) of GPR
    data : In shape as [Group,index,value,noise]
    method : {'MML','LOO','NNP'}
        MML:maximize log marginal likelihood
        LOO:maximize Leave-One-Out cross-validation predictive probability 
    """
    if method is not None:
        gpr.loss_type = method
    method = gpr.get_losstype()
    matrix_params,sigma_params = gpr.Parameter_Groups()
    optimizer = torch.optim.Adam([{'params':matrix_params,'lr':lr},
                                  {'params':sigma_params,'lr':llr}], lr=lr,weight_decay=0.0)
    if schedule_lr:
        lr_scd = torch.optim.lr_scheduler.MultiStepLR(optimizer,schedule_t,gamma = schedule_gamma)

    for epoch in range(max_epoches):
        gpr.zero_grad()
        loss = 0.0
        for group in range(len(data)):
            if method == 'LOO':
                loss = loss*gamma - gpr.Log_LOO_Predictive_Probability(data[group])
            elif method == 'MML': # this is default
                loss = loss*gamma - gpr.Log_Marginal_Likelihood(data[group])
            elif method == 'NNP':
                loss = loss*gamma + gpr.Log_NonNoise_Predictive_Error(data[group])
            else:
                raise RuntimeError("Not supported training method!!")

        loss.backward()
        optimizer.step()
        if epoch%10==0 and verbose:
            print("Train_Epoch:{}\t|Noise:{:.4f}\t|Sigma:{:.4f}\t|Loss:{:.4f}".format(epoch,gpr.noise.detach().item(),torch.mean(torch.diagonal(gpr.Covariance())).detach().item(),loss.item()))
            #print(loss)
        if schedule_lr:
            lr_scd.step()
            
    return loss.item() 