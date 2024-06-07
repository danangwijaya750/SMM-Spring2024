import random
import numpy as np
import torch
from torch_sparse import SparseTensor, matmul
from utils import convert_adj_mat_edge_index_to_r_mat_edge_index, convert_r_mat_edge_index_to_adj_mat_edge_index, sample_mini_batch
from torch_geometric.utils import structured_negative_sampling

def bpr_loss(users_emb_final,
             users_emb_0,
             pos_items_emb_final,
             pos_items_emb_0,
             neg_items_emb_final,
             neg_items_emb_0,
             lambda_val):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (torch.Tensor): e_u_k
        users_emb_0 (torch.Tensor): e_u_0
        pos_items_emb_final (torch.Tensor): positive e_i_k
        pos_items_emb_0 (torch.Tensor): positive e_i_0
        neg_items_emb_final (torch.Tensor): negative e_i_k
        neg_items_emb_0 (torch.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        torch.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2)) # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples


    bpr_loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores))

    loss = bpr_loss + reg_loss

    return loss

def get_user_positive_items(edge_index):
    """
    Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: user -> list of positive items for each
    """

    # key: user_id, val: item_id list
    user_pos_items = {}

    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()

        if user not in user_pos_items:
            user_pos_items[user] = []

        user_pos_items[user].append(item)

    return user_pos_items

# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list[list[long]]): list of lists of item_ids. Cntaining highly rated items of each user.
                            In other words, this is the list of true_relevant_items for each user

        r (list[list[boolean]]): list of lists indicating whether each top k item recommended to each user
                            is a top k ground truth (true relevant) item or not

        k (int): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """

    # number of correctly predicted items per user
    # -1 here means I want to sum at the inner most dimension
    num_correct_pred = torch.sum(r, dim=-1)

    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i]) for i in range(len(groundTruth))])

    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

# wrapper function to get evaluation metrics
def get_metrics(model,
                input_edge_index, # adj_mat based edge index
                input_exclude_edge_indices, # adj_mat based exclude edge index
                k):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model

        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate

        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation

        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    # get the embedding tensor at layer 0 after training
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight


    # convert adj_mat based edge index to r_mat based edge index so we have have
    # the first list being user_ids and second list being item_ids for the edge index
    edge_index = convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index)

    # This is to exclude the edges we have seen before in our predicted interaction matrix (r_mat_rating)
    # E.g: for validation set, we want want to exclude all the edges in training set
    exclude_edge_indices = [convert_adj_mat_edge_index_to_r_mat_edge_index(exclude_edge_index) \
                                      for exclude_edge_index in input_exclude_edge_indices]



    # Generate predicted interaction matrix (r_mat_rating)
    # (num_users x 64) dot_product (num_item x 64).T
    r_mat_rating = torch.matmul(user_embedding, item_embedding.T)

    # shape: num_users x num_item
    rating = r_mat_rating

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        # it's a dict: user -> positive item list
        user_pos_items = get_user_positive_items(exclude_edge_index)

        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            # [user] * len(item) can give us [user1, user1, user1...] with len of len(item)
            # this makes easier to do the masking below
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set the excluded entry in the rat_mat_rating matrix to a very small number
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    # dict of user -> pos_item list
    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list of lists
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]


    # r here is "pred_relevant_items ∩ actually_relevant_items" list for each user
    r = []
    for user in users:
        user_true_relevant_item = test_user_pos_items[user.item()]
        # list of Booleans to store whether or not a given item in the top_K_items for a given user
        # is also present in user_true_relevant_item.
        # this is later on used to compute n_rel_and_rec_k
        label = list(map(lambda x: x in user_true_relevant_item, top_K_items[user]))
        r.append(label)

    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg

def get_embs_for_bpr(model, input_edge_index, BATCH_SIZE, device):
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(input_edge_index)
    edge_index_to_use = convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index)

    # Mini-batching for evaluation and loss calculation
    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(BATCH_SIZE, edge_index_to_use)

    # Ensure indices are on the appropriate device
    user_indices, pos_item_indices, neg_item_indices = user_indices.to(device), pos_item_indices.to(device), neg_item_indices.to(device)

    # Filter out invalid negative item indices
    valid_mask = neg_item_indices < items_emb_final.size(0)
    valid_neg_item_indices = neg_item_indices[valid_mask]

    # Filter user and positive item embeddings based on valid negative item indices
    valid_user_indices = user_indices[valid_mask]
    valid_pos_item_indices = pos_item_indices[valid_mask]

    # Slice tensors based on valid indices
    users_emb_final_batch = users_emb_final[valid_user_indices]
    users_emb_0_batch = users_emb_0[valid_user_indices]
    pos_items_emb_final_batch = items_emb_final[valid_pos_item_indices]
    pos_items_emb_0_batch = items_emb_0[valid_pos_item_indices]
    neg_items_emb_final_batch = items_emb_final[valid_neg_item_indices]
    neg_items_emb_0_batch = items_emb_0[valid_neg_item_indices]

    return users_emb_final_batch, users_emb_0_batch, pos_items_emb_final_batch, pos_items_emb_0_batch, neg_items_emb_final_batch, neg_items_emb_0_batch

def evaluation(model, edge_index, exclude_edge_indices, k, lambda_val):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k"""

    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(edge_index)

    r_mat_edge_index = convert_adj_mat_edge_index_to_r_mat_edge_index(edge_index)
    edges = structured_negative_sampling(r_mat_edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]

    # Filter out invalid negative item indices
    valid_mask = neg_item_indices < items_emb_final.size(0)
    valid_neg_item_indices = neg_item_indices[valid_mask]

    # Filter user and positive item embeddings based on valid negative item indices
    valid_user_indices = user_indices[valid_mask]
    valid_pos_item_indices = pos_item_indices[valid_mask]

    users_emb_final = users_emb_final[valid_user_indices]
    users_emb_0 = users_emb_0[valid_user_indices]
    pos_items_emb_final = items_emb_final[valid_pos_item_indices]
    pos_items_emb_0 = items_emb_0[valid_pos_item_indices]
    neg_items_emb_final = items_emb_final[valid_neg_item_indices]
    neg_items_emb_0 = items_emb_0[valid_neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg = get_metrics(model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg