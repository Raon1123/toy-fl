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
    # In FedCor paper implementation, they always use with "gpr_selection = True" option

    selected_clients = []
    candidate_clients = list(range(args.num_clients))


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


def acs_fedcor_warmup():
    pass