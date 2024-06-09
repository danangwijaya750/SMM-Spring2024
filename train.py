import argparse
import preprocess
import utils
import metrics
from models import GCN, LightGCN
import json
import pandas as pd
from tqdm import tqdm
import utils

import torch
from torch import nn, optim, Tensor

def train_eval_LightGCN(num_users, num_places,edge_index, train_edge_index, val_edge_index, test_edge_index, num_interactions):
    layers = 4
    model = LightGCN(num_users=num_users,
                    num_items=num_places,
                    K=layers)

    # define contants
    ITERATIONS = 10000
    EPOCHS = 10
    # ITERATIONS = 500
    BATCH_SIZE = 32
    LR = 1e-3
    ITERS_PER_EVAL = 200
    ITERS_PER_LR_DECAY = 200
    K = 20
    LAMBDA = 1e-6
    # LAMBDA = 1/2

    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}.")


    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    train_edge_index = utils.convert_r_mat_edge_index_to_adj_mat_edge_index(train_edge_index,num_users=num_users,num_places=num_places)
    val_edge_index = utils.convert_r_mat_edge_index_to_adj_mat_edge_index(val_edge_index,num_users=num_users,num_places=num_places)
    test_edge_index = utils.convert_r_mat_edge_index_to_adj_mat_edge_index(test_edge_index,num_users=num_users,num_places=num_places)

    edge_index = edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)

    # training loop
    train_losses = []
    val_losses = []
    val_recall_at_ks = []

    for iter in tqdm(range(ITERATIONS)):
        # forward propagation
        users_emb_final, users_emb_0,  pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0 \
                    = metrics.get_embs_for_bpr(model, train_edge_index,BATCH_SIZE=BATCH_SIZE, device=device,num_users=num_users,num_places=num_places)

        # loss computation
        train_loss = metrics.bpr_loss(users_emb_final,
                            users_emb_0,
                            pos_items_emb_final,
                            pos_items_emb_0,
                            neg_items_emb_final,
                            neg_items_emb_0,
                            LAMBDA)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # validation set
        if iter % ITERS_PER_EVAL == 0:
            model.eval()

            with torch.no_grad():
                val_loss, recall, precision, ndcg = metrics.evaluation(model,
                                                            val_edge_index,
                                                            [train_edge_index],
                                                            K,
                                                            LAMBDA,num_users=num_users,num_places=num_places
                                                            )

                print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")

                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
                val_recall_at_ks.append(round(recall, 5))
            model.train()

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()

    # evaluate on test set
    model.eval()
    test_edge_index = test_edge_index.to(device)

    test_loss, test_recall, test_precision, test_ndcg = metrics.evaluation(model,
                                                                test_edge_index,
                                                                [train_edge_index, val_edge_index],
                                                                K,
                                                                LAMBDA, num_users=num_users,num_places=num_places
                                                                )

    print(f"[test_loss: {round(test_loss, 5)}, test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}, test_ndcg@{K}: {round(test_ndcg, 5)}")

def train_eval_GCN(num_users, num_places, train_edge_index, val_edge_index, test_edge_index, num_interactions):
    model = GCN


def main():
    parser = argparse.ArgumentParser(description="Select a machine learning model.")
    parser.add_argument('--model', type=str, choices=['GCN', 'LightGCN'], required=True, help='The name of the model to use.')
    parser.add_argument('--interaction', type=str, choices=['rating','review'],required=False, default='rating')
    parser.add_argument('--threshold', type=float, required=False, default=2.5)
    parser.add_argument('--data_path', type=str, required=True, help='The path to the data file.')
    args = parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # data, num_users, num_places = preprocess.load_encode_data(data=data)
    # edge_index = preprocess.load_edge(data)
    # train_edge_index, val_edge_index, test_edge_index, num_interactions = preprocess.data_split(edge_index=edge_index)

    if args.model == 'GCN':
        train_eval_GCN(num_users, num_places, train_edge_index, val_edge_index, test_edge_index, num_interactions)

    elif args.model == 'LightGCN':
        if args.interaction == 'rating':
            data,num_users,num_places = preprocess.load_encode_data(data=data)
            edge_index = preprocess.load_edge(data,src_index_col='user_id',dst_index_col='place_id',link_index_col='rating',rating_threshold=args.threshold)
            train_edge_index, val_edge_index, test_edge_index, num_interactions = preprocess.data_split(edge_index=edge_index)
            train_eval_LightGCN(num_users, num_places, edge_index,train_edge_index, val_edge_index, test_edge_index, num_interactions)
        elif args.interaction =='review' :
            data,num_users,num_places = preprocess.load_encode_data(data=data)
            data = preprocess.preprocess_sentiment_score(data)
            edge_index = preprocess.load_edge(data,src_index_col='user_id',dst_index_col='place_id',link_index_col='normalized_weighted_sentiment_score',rating_threshold=args.threshold)
            train_edge_index, val_edge_index, test_edge_index, num_interactions = preprocess.data_split(edge_index=edge_index)
            train_eval_LightGCN(num_users, num_places, train_edge_index, val_edge_index, test_edge_index, num_interactions)

if __name__ == '__main__':
    main()