import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.nn import init, LeakyReLU, Linear, Module, ModuleList, Parameter

from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.nn import GCNConv

from torch_sparse import SparseTensor, matmul

import utils


class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """
    def __init__(self, num_users,
                 num_items,
                 embedding_dim=8,
                 K=3,
                 add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops

        # define user and item embedding for direct look up.
        # embedding dimension: num_user/num_item x embedding_dim

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0

        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        # "Fills the input Tensor with values drawn from the normal distribution"
        # according to LightGCN paper, this gives better performance
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: Tensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """

        """
            compute \tilde{A}: symmetrically normalized adjacency matrix
            \tilde_A = D^(-1/2) * A * D^(-1/2)    according to LightGCN paper

            this is essentially a metrix operation way to get 1/ (sqrt(n_neighbors_i) * sqrt(n_neighbors_j))

            if your original edge_index look like
            tensor([[   0,    0,    0,  ...,  609,  609,  609],
                    [   0,    2,    5,  ..., 9444, 9445, 9485]])

                    torch.Size([2, 99466])

            then this will output:
                (
                 tensor([[   0,    0,    0,  ...,  609,  609,  609],
                         [   0,    2,    5,  ..., 9444, 9445, 9485]]),
                 tensor([0.0047, 0.0096, 0.0068,  ..., 0.0592, 0.0459, 0.1325])
                 )

              where edge_index_norm[0] is just the original edge_index

              and edge_index_norm[1] is the symmetrically normalization term.

            under the hood it's basically doing
                def compute_gcn_norm(edge_index, emb):
                    emb = emb.weight
                    from_, to_ = edge_index
                    deg = degree(to_, emb.size(0), dtype=emb.dtype)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

                    return norm

        """
        edge_index_norm = gcn_norm(edge_index=edge_index,
                                   add_self_loops=self.add_self_loops)

        # concat the user_emb and item_emb as the layer0 embing matrix
        # size will be (n_users + n_items) x emb_vector_len.   e.g: 10334 x 64
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0

        embs = [emb_0] # save the layer0 emb to the embs list

        # emb_k is the emb that we are actually going to push it through the graph layers
        # as described in lightGCN paper formula 7
        emb_k = emb_0

        # push the embedding of all users and items through the Graph Model K times.
        # K here is the number of layers
        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)


        # this is doing the formula 8 in LightGCN paper

        # the stacked embs is a list of embedding matrix at each layer
        #    it's of shape n_nodes x (n_layers + 1) x emb_vector_len.
        #        e.g: torch.Size([10334, 4, 64])
        embs = torch.stack(embs, dim=1)

        # From LightGCn paper: "In our experiments, we find that setting α_k uniformly as 1/(K + 1)
        #    leads to good performance in general."
        emb_final = torch.mean(embs, dim=1) # E^K


        # splits into e_u^K and e_i^K
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        # here using .weight to get the tensor weights from n.Embedding
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j, norm):
        # x_j is of shape:  edge_index_len x emb_vector_len
        #    e.g: torch.Size([77728, 64]
        #
        # x_j is basically the embedding of all the neighbors based on the src_list in coo edge index
        #
        # elementwise multiply by the symmetrically norm. So it's essentiall what formula 7 in LightGCN
        # paper does but here we are using edge_index rather than Adj Matrix
        return norm.view(-1, 1) * x_j
    

class NGCF(Module):
  def __init__(self, n_users, n_items, embed_size, n_layers, adj_matrix, device):
    super().__init__()
    self.n_users = n_users
    self.n_items = n_items
    self.embed_size = embed_size
    self.n_layers = n_layers
    self.adj_matrix = adj_matrix
    self.device = device

    # The (user/item)_embeddings are the initial embedding matrix E
    self.user_embeddings = Parameter(torch.rand(n_users, embed_size))
    self.item_embeddings = Parameter(torch.rand(n_items, embed_size))
    # The (user/item)_embeddings_final are the final concatenated embeddings [E_1..E_L]
    # Stored for easy tracking of final embeddings throughout optimization and eval
    self.user_embeddings_final = Parameter(torch.zeros((n_users, embed_size * (n_layers + 1))))
    self.item_embeddings_final = Parameter(torch.zeros((n_items, embed_size * (n_layers + 1))))

    # The linear transformations for each layer
    self.W1 = ModuleList([Linear(self.embed_size, self.embed_size) for _ in range(0, self.n_layers)])
    self.W2 = ModuleList([Linear(self.embed_size, self.embed_size) for _ in range(0, self.n_layers)])

    self.act = LeakyReLU()
    
    # Initialize each of the trainable weights with the Xavier initializer
    self.init_weights()

  def init_weights(self):
    for name, parameter in self.named_parameters():
      if ('bias' not in name):
        init.xavier_uniform_(parameter)

  def compute_loss(self, batch_user_emb, batch_pos_emb, batch_neg_emb):
    pos_y = torch.mul(batch_user_emb, batch_pos_emb).sum(dim=1)
    neg_y = torch.mul(batch_user_emb, batch_neg_emb).sum(dim=1)
    # Unregularized loss
    bpr_loss = -(torch.log(torch.sigmoid(pos_y - neg_y))).mean()
    return bpr_loss

  def forward(self, u, i, j):
    adj_splits = utils.split_mtx(self.adj_matrix)
    embeddings = torch.cat((self.user_embeddings, self.item_embeddings))
    final_embeddings = [embeddings]

    for l in range(self.n_layers):
      embedding_parts = []
      for part in adj_splits:
        embedding_parts.append(torch.sparse.mm(utils.to_sparse_tensor(part).to(self.device), embeddings))

      # Message construction
      t1_embeddings = torch.cat(embedding_parts, 0)
      t1 = self.W1[l](t1_embeddings)
      t2_embeddings = embeddings.mul(t1_embeddings)
      t2 = self.W2[l](t2_embeddings)

      # Message aggregation
      embeddings = self.act(t1 + t2)
      normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
      final_embeddings.append(normalized_embeddings)

    # Make sure to update the (user/item)_embeddings(_final)
    final_embeddings = torch.cat(final_embeddings, 1)
    final_u_embeddings, final_i_embeddings = final_embeddings.split((self.n_users, self.n_items), 0)
    self.user_embeddings_final = Parameter(final_u_embeddings)
    self.item_embeddings_final = Parameter(final_i_embeddings)

    batch_user_emb = final_u_embeddings[u]
    batch_pos_emb = final_i_embeddings[i]
    batch_neg_emb = final_i_embeddings[j]

    return self.compute_loss(batch_user_emb, batch_pos_emb, batch_neg_emb)
     