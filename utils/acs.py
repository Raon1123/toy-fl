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


def gpr_warmup(args, epoch, gpr,
    gt_global_losses, gpr_data):
    gpr.update_loss(np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gt_global_losses[-1]),1)],1))
    epoch_gpr_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                    np.ones([args.num_users,1])],1)
    gpr_data.append(epoch_gpr_data)
    print("Training GPR")
    TrainGPR(gpr,
            gpr_data[max([(epoch-args.gpr_begin-args.group_size+1),0]):epoch-args.gpr_begin+1],
            args.train_method,
            lr = 1e-2,
            llr = 0.0,
            gamma = args.GPR_gamma,
            max_epoches=args.GPR_Epoch+50,
            schedule_lr=False,
            verbose=args.verbose)


def gpr_optimal(args, epoch, gpr, m, 
    global_model, train_dataset, user_groups,
    gt_global_losses, gpr_data):
    gpr.update_loss(np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gt_global_losses[-1]),
                                    1)],1))
    gpr.Reset_Discount()
    print("Training with Random Selection For GPR Training:")
    random_idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    gpr_acc,gpr_loss = train_federated_learning(args,epoch,
                        copy.deepcopy(global_model),random_idxs_users,train_dataset,user_groups)
    epoch_gpr_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gpr_loss)-np.array(gt_global_losses[-1]),1),
                                    np.ones([args.num_users,1])],1)
    gpr_data.append(epoch_gpr_data)
    print("Training GPR")
    TrainGPR(gpr,gpr_data[-ceil(args.group_size/args.GPR_interval):],
            args.train_method,lr = 1e-2,llr = 0.0,gamma = args.GPR_gamma**args.GPR_interval,max_epoches=args.GPR_Epoch,schedule_lr=False,verbose=args.verbose)


def acs_fedcor_warmup():
    pass