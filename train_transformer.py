import genchem
import torch
import pytorch_lightning as pl
import pandas as pd
import wandb
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from vae_cyc import Transformer, TransformerDataset
from sklearn.model_selection import train_test_split 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='vae_transformer', type=str)
    parser.add_argument('--name', default='vae_transformer', type=str)
    parser.add_argument('--dataset_path', default='moses', type=str) 
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--gpus', default=1, type=int)


    args = parser.parse_args()
    # get extension of dataset file 
    ext = args.dataset_path.split('.')[-1]

    if ext == 'csv':
        df = pd.read_csv(args.dataset_path)
    elif ext == 'parquet':
        df = pd.read_parquet(args.dataset_path)
    print(df.columns)
    
    max_len = df.SMILES.str.len().max()
    vocab = genchem.Vocab(df, args.smiles_col)

    train_df, val_df = train_test_split(df, test_size=0.01)

    
    train_ds = TransformerDataset(train_df[args.smiles_col], vocab, max_len=max_len)
    val_ds = TransformerDataset(val_df[args.smiles_col], vocab, max_len=max_len)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, collate_fn=train_ds.collate)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=128, collate_fn=val_ds.collate)

    wandb.finish()
    wandb_logger = WandbLogger(project=args.project_name, log_model=True)
    # wandb.init()
    
    m = Transformer(vocab_size=len(vocab.char2idx), 
                    embed_size=256, 
                    latent_dim=256, 
                    num_heads=4, 
                    hidden_dim=256, 
                    num_layers=6, 
                    max_seq_length=max_len, 
                    vocab=vocab)
    
    checkpoint = ModelCheckpoint(dirpath='/home/jovyan/data/transformer-vae-moses', 
                                          save_top_k=3, 
                                          save_last=True,
                                          monitor='val_r_loss', 
                                          filename='vae-transformer-{epoch}-{train_loss:.2f}',
                                          verbose=True)
    
    trainer = pl.Trainer(devices=[args.gpus], logger=[wandb_logger], callbacks=[checkpoint], max_epochs=100)
    
    
    trainer.fit(m, train_dl, val_dl)

if __name__ == '__main__':
    main()