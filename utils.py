import random
import numpy as np
import torch
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import structured_negative_sampling

def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index, num_users, num_places):
    R = torch.zeros((num_users, num_places))
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = 1

    R_transpose = torch.transpose(R, 0, 1)
    adj_mat = torch.zeros((num_users + num_places , num_users + num_places))
    adj_mat[: num_users, num_users :] = R.clone()
    adj_mat[num_users :, : num_users] = R_transpose.clone()
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo = adj_mat_coo.indices()
    return adj_mat_coo

def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index, num_users, num_places):
    sparse_input_edge_index = SparseTensor(row=input_edge_index[0],
                                           col=input_edge_index[1],
                                           sparse_sizes=((num_users + num_places), num_users + num_places))
    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[: num_users, num_users :]
    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    return r_mat_edge_index

# helper function for training and compute BPR loss
# since this is a self-supervised learning, we are relying on the graph structure itself and
# we don't have label other than the graph structure so we need to the folloing function
# which random samples a mini-batch of positive and negative samples
def sample_mini_batch(batch_size, edge_index):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    # structured_negative_sampling is a pyG library
    # Samples a negative edge :obj:`(i,k)` for every positive edge
    # :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    # tuple of the form :obj:`(i,j,k)`.
    #
    #         >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
    #         ...                               [0, 1, 2, 3]])
    #         >>> structured_negative_sampling(edge_index)
    #         (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))
    edges = structured_negative_sampling(edge_index)

    # 3 x edge_index_len
    edges = torch.stack(edges, dim=0)

    # here is whhen we actually perform the batch sampe
    # Return a k sized list of population elements chosen with replacement.
    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)

    batch = edges[:, indices]

    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


# NGCF UTILS
def split_mtx(X, n_folds=200):
  """
  Split a matrix/Tensor into n parts.
  Useful for processing large matrices in batches
  """
  X_folds = []
  fold_len = X.shape[0]//n_folds
  for i in range(n_folds):
    start = i * fold_len
    if i == n_folds -1:
      end = X.shape[0]
    else:
      end = (i + 1) * fold_len
    X_folds.append(X[start:end])
  return X_folds

def to_sparse_tensor(X):
  """
  Convert a sparse numpy object to a sparse pytorch tensor.
  Note that the tensor does not yet live on the GPU
  """
  coo = X.tocoo().astype(np.float32)
  i = torch.LongTensor(np.mat((coo.row, coo.col)))
  v = torch.FloatTensor(coo.data)
  return torch.sparse.FloatTensor(i, v, coo.shape)

def save_state(model, optimizer, iteration, name):
  torch.save(model.state_dict(), name+".pt")
  torch.save(optimizer.state_dict(), name+"-opt.pt")
  torch.save(torch.IntTensor(iteration), name+"-it.pt")