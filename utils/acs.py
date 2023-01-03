import numpy as np

import torch
from utils.toolkit import get_pairdistance

def acs_random(args):
    selected_clients = np.random.choice(args.num_clients, 
        args.active_selection, 
        replace=False)

    return selected_clients


def acs_badge(args, prev_params):
    params = torch.stack(prev_params)
    pdist_mat = torch.full((args.active_selection, args.num_clients), torch.inf)
    selected_clients = []

    # using k-means++
    seed_client = np.random.choice(args.num_clients, 
        1, replace=False).item()
    selected_clients.append(seed_client)
    pdist_mat[0:,] = get_pairdistance(params[seed_client,:], params)

    for t in range(1, args.active_selection):
        pmf, _ = pdist_mat.min(dim=0)
        pmf = torch.div(pmf, pmf.sum())

        select = pmf.multinomial(num_samples=1, replacement=False).item()
        selected_clients.append(select)

        pdist_mat[t,:] = get_pairdistance(params[select,:], params)
    
    return selected_clients


def acs_loss(args, prev_losses):
    loss_array = np.exp(prev_losses)
    total_loss = np.sum(loss_array)
    loss_list = loss_array.tolist()  

    pmf = list(map(lambda item: item/total_loss, loss_list))
    
    if args.verbose:
        print("pmf:", pmf)

    selected_clients = np.random.choice(args.num_clients, 
        args.active_selection, 
        replace=False, 
        p=pmf)

    return selected_clients


def acs_powd(args, size_array, prev_losses):
    """
    """
    # random sample d clients from data size of each client
    total_size = np.sum(size_array)
    norm = lambda x: x / total_size
    pmf = norm(size_array)
    
    if args.verbose:
        print("pmf:", pmf)

    select_d_client = np.random.choice(args.num_clients,
        args.powd,
        replace=False,
        p=pmf)

    # estimate local loss from sampled client 
    prev_d_loss = prev_losses[select_d_client]
    sorted_d_loss = np.argsort(-prev_d_loss)

    # select highest loss clients
    selected_clients = sorted_d_loss[:args.active_selection]

    return selected_clients


def acs_topk(args, prev_losses):
    """
    """
    sorted_loss = np.argsort(-prev_losses)
    selected_clients = sorted_loss[:args.active_selection]

    return selected_clients


def acs_fedcor(args, mu, sigma, alpha):
    """
    Active client selection algorithm for FedCor
    - mu (matrix): 0-vector
    - sigma (matrix): X.T X
    - alpha (float): scale factor
    """
    selected_clients = []
    candidate_clients = list(range(args.num_clients))

    """
    # accuracy and loss of every client
    gpr_acc, gpr_loss = train_federated_learning(args,epoch,
                        copy.deepcopy(global_model),gpr_idxs_users,train_dataset,user_groups) 
    # 0: index 1: values 2: noisy
    gpr_loss_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gpr_loss)-np.array(gt_global_losses[-1]),1),
                                    np.ones([args.num_users,1])],1)

    predict_loss,_,_=gpr.Predict_Loss(gpr_loss_data,gpr_idxs_users,np.delete(list(range(args.num_users)),gpr_idxs_users))

        mu_p, sigma_p = self.Posteriori(data[priori_idx,:])
            data = torch.tensor(data).to(self.noise)
            indexes = data[:,0].long()
            values = data[:,1]
            noisy = data[:,2]
            Cov = self.Covariance()
            
            Sigma_inv = torch.inverse(Cov[indexes,:][:,indexes]+torch.diag(noisy).to(self.noise)*(self.noise**2))
            mu = self.mu.to(self.noise)+((Cov[:,indexes].mm(Sigma_inv)).mm((values-self.mu[indexes].to(self.noise)).unsqueeze(1))).squeeze()
            Sigma = Cov
            return mu.detach(), Sigma.detach()
        noise_scale = 0.1
        while True:
            try:
                pdist = MultivariateNormal(loc = mu_p[posteriori_idx],
                                           covariance_matrix = sigma_p[posteriori_idx,:][:,posteriori_idx]+noise_scale*torch.eye(len(posteriori_idx)))
                break
            except ValueError:
                print(sigma_p.shape)
                noise_scale*=10
                if noise_scale > 100:
                    raise Exception("Cannot satisfy positive definiteness property")
        predict_loss = -pdist.log_prob(torch.tensor(data[posteriori_idx,1]).to(mu_p))
        predict_loss = predict_loss.detach().item()
        return predict_loss,mu_p,sigma_p
    
    print("GPR Predict Off-Policy Loss:{:.4f}".format(predict_loss))
    
    offpolicy_losses.append(predict_loss)

    gpr_dloss = np.sum((np.array(gpr_loss)-np.array(gt_global_losses[-1]))*weights)
    gpr_loss_decrease.append(gpr_dloss)
    gpr_acc_improve.append(gpr_acc-train_accuracy[-1])
    """

    for _ in range(args.active_selection):
        # compute the score for each candidate client
        score = np.zeros(args.num_clients)
        for i in candidate_clients:
            score[i] = np.dot(mu[i], np.dot(sigma[i], mu[i]))

        # select the client with the highest score
        selected = np.argmax(score)
        selected_clients.append(selected)

        # remove the selected client from the candidate list
        candidate_clients.remove(selected)

        # update mu and sigma
        mu = mu - alpha * np.dot(sigma[selected], mu)
        sigma = sigma - alpha * np.dot(sigma[selected], sigma)

    return selected_clients