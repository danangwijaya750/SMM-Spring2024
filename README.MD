# Social Media Mining Final Project

## Training
1. LightGCN model with user rating as interaction
    python3 train.py --model LightGCN --interaction rating --threshold 3.0 --data_path data/places_data_new.json
2. LightGCN model with user review sentiment score as interaction
    python3 train.py --model LightGCN --interaction review --threshold 3.0 --data_path data/places_data_new.json