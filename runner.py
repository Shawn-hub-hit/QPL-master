from importlib_metadata import PathDistribution
from evaluation import *
from trainer import *
from model import *
from dataload import *
from Config import *
from dataset_process import *
os.chdir(sys.path[0])

constant = Constant(cur_time='10-13')

parser = argparse.ArgumentParser()
# data arguments
parser.add_argument('--train', action='store_true', help='start train')
parser.add_argument('--test', action='store_true', help='start test')

parser.add_argument('--test_file', type=str, default=datapath.joinpath('test.txt'))
parser.add_argument('--left_text_file', type=str, default=datapath.joinpath('left-text.txt'))
parser.add_argument('--right_text_file', type=str, default=datapath.joinpath('right-text.txt'))
parser.add_argument('--emb_matrix_file', type=str, default=datapath.joinpath('emb_matrix_file.pkl'))

parser.add_argument('--save_path', type=str, default=constant.savedir)
parser.add_argument('--num_neg', type=int, default=16)
parser.add_argument('--num_test_neg', type=int, default=None)
parser.add_argument('--grid_size', type=int, default=100)   #The performance varies very little with grid size from 100 to 1000.
parser.add_argument('--max_seq_len', type=int, default=60)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--save_file', type=str, default='model.pt')
parser.add_argument('--fix_left_length', type=int, default=15)
parser.add_argument('--fix_right_length', type=int, default=40)

config = parser.parse_args(['--test'])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(device)
# load dataset
interactions = Interactions(config, constant.logger)
config.num_test_neg = len(interactions.right_text_df) - 1

test_dataset = my_Datasets(relation=interactions.test_relation_df, interaction=interactions, stage='test', 
                        batch_size=config.test_batch_size, shuffle=False)

testloader = Dataloaders(
    dataset=test_dataset,
    stage='dev',
    fixed_length_left=config.fix_left_length,
    fixed_length_right=config.fix_right_length,
)

model_params = {}
model_params['dropout_rate'] = 0.1
model_params['hidden_size'] = 768
model_params['d_emb'] = 64
model_params['train'] = config.train
model_params['embedding_freeze'] = False
model_params['GridCount_x'] = interactions.GridCount_x
model_params['GridCount_y'] = interactions.GridCount_y
model_params['user_num'] = 20197
model_params['pooling'] = 'max'
model_params['padding_idx'] = 0
model_params['mask_value'] = 0
model_params['embedding'] = interactions.embedding_matrix
model_params['fix_left_length'] = config.fix_left_length
model_params['fix_right_length'] = config.fix_right_length


model = QPL(model_params=model_params, device = device)
model.to(device)
constant.logger.info(config)

str_print = 'Trainable params: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
constant.logger.info(str_print)


optimizer = torch.optim.Adadelta(model.parameters())

metrics = [
    NormalizedDiscountedCumulativeGain(k=5),
    NormalizedDiscountedCumulativeGain(k=10),
    MeanReciprocalRank(k=10),
    ]

trainer = Trainer(
    config=config,
    metrics=metrics,
    model=model,
    optimizer=optimizer,
    trainloader=None,
    validloader=testloader,
    validate_interval=None,
    epochs=15,
    checkpoint= checkpoints_u / 'trainer.pt',
    constant=constant,
    block_num=60,
    save_all=True
)

if config.test:
    trainer.run_test()


