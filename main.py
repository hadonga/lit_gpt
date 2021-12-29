import os
from argparse import ArgumentParser

from data.data_interface import CifarDataModule
from model.model_interface import GptModule

import pytorch_lightning as pl
import torchmetrics
pl.seed_everything(hash("setting random seeds") % 2**32 - 1)

import wandb
from pytorch_lightning.loggers import WandbLogger
wandb.login()
wandb_logger = WandbLogger(project='lit_gpt')

# Parameters for training
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def main(args):
    # load dataset
    dataset = CifarDataModule(args)
    dataset.prepare_data()
    dataset.setup()

    # load model
    # initialize a trainer instance and kick off training
    model = GptModule(args, dataset.vocab_size, dataset.block_size)

    # callback
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, 'checkpoints'),
        verbose=True,
    )
    early_stopping = pl.callbacks.EarlyStopping(min_delta=0.00001,patience =3)
    metrics=pl.callbacks.TQDMProgressBar()

    # initialize trainer
    trainer = pl.Trainer(
        gpus=-1,
        auto_scale_batch_size=True,
        logger=wandb_logger,
        callbacks=[checkpoint, metrics],
        max_epochs=args.train_epochs,  # todo run a bigger model and longer, this is tiny
    )

    trainer.fit(model, dataset)


if __name__ == '__main__':
    parser= ArgumentParser(add_help=False)
    parser.add_argument('--dataset_dir', default='./')
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--train_epochs', type=int, default= 20)
    parser.add_argument('--batch_size',type= int, default= 8)
    parser.add_argument('--num_workers', type=int,default=8)
    parser.add_argument('--learning_rate',type= float, default= 3e-4)

    args = parser.parse_args()

    main(args)

