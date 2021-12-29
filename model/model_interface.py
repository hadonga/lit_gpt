import pytorch_lightning as pl
from model.gpt_model import GPT, GPTConfig,GPT1Config
import torch

class GptModule(pl.LightningModule):
    def __init__(self, args, vocab_size = None, block_size =None):
        super().__init__()
        self.args = args

        # parameters for model
        mconf = GPTConfig(vocab_size, block_size, embd_pdrop=0.0,
                          resid_pdrop=0.0, attn_pdrop=0.0,
                          n_layer=12, n_head=8, n_embd=256)
        self.model = GPT(mconf)

    def forward(self, x, y):
        logits, loss = self.model(x, y)
        return logits, loss

    def training_step(self, batch, batch_idx):
        x, y = batch['x'],batch['y']
        logits,loss = self(x,y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pass
        # x, y = batch['x'], batch['y']
        # logits, loss = self(x, y)
        # return {'val_loss': loss}

    def validation_end(self, outputs):
        pass
        # avg_loss = torch.stack([i['val_loss'] for i in outputs]).mean()
        # self.loss_monitor= avg_loss
        # return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.95))
        return {"optimizer": optimizer}


