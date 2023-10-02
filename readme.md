# QPL
### Code for 'Learning User Geographical Preferences for Personalized POI Search' paper

### Requirements

* Python 3.7
* Pytorch 1.11
* transformers 4.16.2

Install required packages
```
pip intall -r requirements.txt
```


### Datasets
'./Datasets/Beijing_data/right_text.csv' includes all POIs in Beijing dataset.

'./Datasets/Beijing_data/left_text.csv' includes the queries that are posed by users from Beijing dataset.

'./Datasets/Beijing_data/test.csv' provides some samples of clicking log. Each click record consists of a user ID, a query, and the POI that the user clicked. The whole anonymized dataset will be released upon acceptance

'./Datasets/Beijing_data/emb_matrix_file.pkl' the saved embedding matrix for word token.

### checkpoints

'./checkpoints_/trainer.pt' the saved model checkpoint file for loading to test the performance


### Testing

To test QPL

```

python runner.py
```

The expected output:

```
Validation: normalized_discounted_cumulative_gain@5(0.0): 0.3716 - normalized_discounted_cumulative_gain@10(0.0): 0.3992 - mean_reciprocal_rank@10(0.0): 0.3408
```

This can reproduce NDCG@5, NDCG@10, and MRR@10 of our model on Beijing dataset.