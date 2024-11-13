import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import wandb
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint 

# def generate(m, vocab, max_len=100):
#     current_tokens = [m.vocab.char2idx[m.vocab.sos_token]]
#     tgt_tokens = torch.tensor([current_tokens]).long().to(m.device)
#     s = ""
#     last_hidden_state = torch.normal(0, 1.0, size=(1, 256)).to(m.device)
#     resized_latent = m.fc_latent_to_hidden(last_hidden_state) 
#     resized_latent = resized_latent.unsqueeze(1).repeat(1, 1, 1) 
#     emb = m.embedding(tgt_tokens) + m.positional_encoding[:, :tgt_tokens.size(1), :]
#     for i in range(max_len -2 ):
#         tgt_mask = m.generate_square_subsequent_mask(emb.size(1)).to(emb.device)
#         decoded = m.decoder(emb, resized_latent, tgt_mask=tgt_mask)
#         outputs_v = m.output_fc(decoded)
#         outputs_v = outputs_v[:,-1,:] 
#         top_char = torch.argmax(outputs_v)
#         if top_char == vocab.char2idx[vocab.eos_token]:
#             break
#         current_tokens.append(top_char.item())
#         tgt_tokens = torch.tensor([current_tokens]).long().to(m.device)
#         emb = m.embedding(tgt_tokens) + m.positional_encoding[:, :tgt_tokens.size(1), :]
        
#         s += vocab.idx2char[top_char.item()]
#     s = vocab.reverse_special(s)
    
#     if genchem.valid(s):
#         print(s)
#     return s

class Transformer(pl.LightningModule):
    def __init__(self, vocab_size, embed_size, latent_dim, num_heads, hidden_dim, num_layers, max_seq_length, vocab):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(embed_size, latent_dim)
        self.fc_logvar = nn.Linear(embed_size, latent_dim)
        self.fc_latent_to_hidden = nn.Linear(latent_dim, embed_size)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads, hidden_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_fc = nn.Linear(embed_size, vocab_size)
        self.vocab = vocab 
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.char2idx[vocab.pad_token], reduction='sum')
        self.kl_weight = 0.6
        self.save_hyperparameters()
        

    def compute_loss(self, outputs, targets, logvar, mu):
        r_loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss * self.kl_weight
        return r_loss, kl_loss
    
    def cyclical_annealing(self,T,M,step,R=0.4, max_kl_weight=1):
        """
        Implementing: <https://arxiv.org/abs/1903.10145>
        T = Total steps 
        M = Number of cycles 
        R = Proportion used to increase beta
        t = Global step 
        """
        period = (T/M) # N_iters/N_cycles 
        internal_period = (step) % (period)  # Itteration_number/(Global Period)
        tau = internal_period/period
        if tau > R:
            tau = max_kl_weight
        else:
            tau = min(max_kl_weight, tau/R) # Linear function 
        return tau
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, x, mask=None):
        emb = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        encoded = self.encoder(emb, src_key_padding_mask=mask)
        last_hidden_state = encoded[:, -1, :] # z this hidden state is the last hidden state.
        mu_vector = self.fc_mu(last_hidden_state)
        logvar_vector = self.fc_logvar(last_hidden_state)
        return last_hidden_state, emb, logvar_vector, mu_vector
    
    def decode(self, last_hidden_state, x, mask=None):
        resized_latent = self.fc_latent_to_hidden(last_hidden_state)
        resized_latent = resized_latent.unsqueeze(1).repeat(1, self.max_seq_length, 1)
        # add positional encoding to reshaped latents 
        
        # resized_latent = resized_latent + self.positional_encoding[:, :resized_latent.size(1), :] 
        
        # Add positional encoding
        hidden = resized_latent + self.positional_encoding[:, :self.max_seq_length, :]
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        # Pass through decoder
        # decoded = self.decoder(x, hidden, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
        # decoded = self.decoder(x, hidden, tgt_mask=tgt_mask,tgt_is_causal=True,tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
        decoded = self.decoder(x, hidden, tgt_mask=tgt_mask,tgt_key_padding_mask=mask)

        outputs_v = self.output_fc(decoded)
        
        return outputs_v

    def forward(self,batch):
        with_bos, with_eos, masks = batch
        last_hidden_states, memory, logvar, mu = self.encode(with_bos, mask=masks)
        z = self.reparameterize(mu, logvar)
        output_v = self.decode(z, memory, mask=masks)
        return output_v, with_eos, logvar, mu

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001) 

    def training_step(self, batch, batch_idx):
        outputs_v, with_eos, logvar, mu = self.forward(batch)
        r_loss, kl_loss = self.compute_loss(outputs_v, with_eos, logvar, mu)
        self.kl_weight = self.cyclical_annealing(100000, 100, step=self.global_step, max_kl_weight=0.6)
        r = {}
        r['r_loss'] = r_loss 
        r['kl_loss'] = kl_loss
        r['loss'] = r_loss + kl_loss
        r['kl_weight'] = self.kl_weight
        for key in r:
            self.log(key, r[key])
        return r
    
    def validation_step(self, batch, batch_idx):
        outputs_v, with_eos, logvar, mu = self.forward(batch)
        r_loss, kl_loss = self.compute_loss(outputs_v, with_eos, logvar, mu)
        r = {}
        r['r_loss'] = r_loss 
        r['kl_loss'] = kl_loss
        r['loss'] = r_loss + kl_loss
        for key in r:
            self.log(f'val_{key}', r[key])
        return r
    

        
        