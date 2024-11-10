import torch 
import torch.nn as nn
import torch.optim as optim
import lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VAE(pl.LightningModule):
    def __init__(self, 
        latent_size, 
        hidden_dim, 
        num_layers, 
        vocab, 
        bidirectional=True,
        word_dropout=0.5,lr=0.001, 
        t_kl_weight=0.0025, c_step=1000, 
        dropout=0.3, 
        annealing=True, 
        emb_dim=32, enc_num_layers=2, 
        kl_weight=0.2,
        max_kl_weight=1.0,
        n_cycles_anneal=20):
        super().__init__()

        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab = vocab
        self.bidirectional = bidirectional
        self.word_dropout = word_dropout
        self.lr = lr
        self.t_kl_weight = t_kl_weight
        self.c_step = c_step
        self.dropout = dropout
        self.annealing = annealing
        self.emb_dim = emb_dim
        self.enc_num_layers = enc_num_layers
        self.n_cycles_anneal = n_cycles_anneal
        self.kl_weight = kl_weight
        self.anneal_function = 'cyclical'
        self.max_kl_weight = max_kl_weight
        self.enc_hidden_factor = (2 if bidirectional else 1) * enc_num_layers
        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.encoder = nn.GRU(self.emb_dim, hidden_dim, batch_first=True, num_layers=enc_num_layers,bidirectional=bidirectional, dropout=dropout)

        self.decoder = nn.GRU(self.emb_dim + self.latent_size, hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=False, dropout=dropout)

        self.mu_fc = nn.Linear(hidden_dim * self.enc_hidden_factor, latent_size)
        self.logvar_fc = nn.Linear(hidden_dim * self.enc_hidden_factor, latent_size)

        self.embedding = nn.Embedding(vocab.vocab_size, emb_dim, padding_idx=vocab.pad_idx)

        self.latent2hidden = nn.Linear(latent_size, hidden_dim * self.num_layers)
        self.outputs2vocab = nn.Linear(hidden_dim, vocab.vocab_size)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, reduction='sum')

        self.save_hyperparameters()
    
    def resize_hidden_encoder(self,h, batch_size):
        h = h.transpose(0,1).contiguous()
        if self.bidirectional or self.enc_num_layers > 1:
            hidden_factor = self.enc_hidden_factor
            h = h.view(batch_size, self.hidden_dim * hidden_factor)
        else:
            h = h.squeeze(0)
        return h
    
    def resize_hidden_decoder(self, h, batch_size):
        if self.num_layers > 1:
            h = h.view(batch_size, self.num_layers, self.hidden_dim)
            h = h.transpose(0,1).contiguous()
        else:
            h = h.unsqueeze(0)
        return h
    
    def _cyclical_anneal(self, T, M, t, R=0.7):
        tm = T / M 
        mod = (t - 1) % tm
        a = mod / tm
        if a > R:
            a = R 
        else:
            a = min(1, a)
        return a
    
    def kl_anneal_function(self, anneal_function, step):
        if anneal_function == 'cyclical':
            return self._cyclical_anneal(self.c_step, self.n_cycles_anneal, t=step, R=self.max_kl_weight)
        else:
            print('Anneal function not implemented')
            return self.kl_weight
    
    def mask_inputs(self, x):
        x_mutate = x.clone()
        prob = torch.rand(x.size())
        prob[(x_mutate - self.vocab.sos_idx) * (x_mutate - self.vocab.eos_idx) == 0] = 1
        x_mutate[prob < self.word_dropout] = self.vocab.unk_idx
        x_mutate = x_mutate.to(self.device).long()
        return x_mutate

    def compute_loss(self, outputs, targets, lengths, logvar, mu):
        targets = targets[:, :torch.max(torch.tensor(lengths)).item()].contiguous().view(-1)
        outputs = outputs.view(-1, outputs.size(2))
        r_loss = self.criterion(outputs, targets)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss * self.kl_weight 
        return r_loss, kl_loss
    
    def enc_forward(self, x, lengths):
        batch_size = x.size(0)

        e = self.embedding(x)

        lengths = torch.tensor(lengths).cpu().numpy()

        x_packed = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False) 

        _, h = self.encoder(x_packed) # h shape is (num_layers * num_directions, batch, hidden_size)

        h = self.resize_hidden_encoder(h, batch_size)

        mu, logvar = self.mu_fc(h), self.logvar_fc(h)

        z = torch.normal(0,1, size=mu.size()).to(self.device)
        std = torch.exp(0.5 * logvar)
        z = mu + std * z
        return z, mu, logvar 
    
    def dec_forward(self,x,z, lengths):
        batch_size = x.shape[0]
        x = self.mask_inputs(x)
        e = self.embedding(x)
        z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
        x_input = torch.cat([e, z_0], dim=-1)
        lengths = torch.tensor(lengths).cpu().numpy()
        packed_input = pack_padded_sequence(x_input, lengths, batch_first=True, enforce_sorted=False)

        h = self.latent2hidden(z)
        h = self.resize_hidden_decoder(h, batch_size)
        outputs, _ = self.decoder(packed_input, h)
        padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0]
        outputs_v = self.outputs2vocab(padded_outputs)
        return outputs_v
    
    def forward(self, data):
        with_bos, with_eos, lengths = data 
        z, mu, logvar = self.enc_forward(with_bos, lengths)
        outputs = self.dec_forward(with_bos, z, lengths)
        return outputs, with_eos, lengths, logvar, mu
    
    def training_step(self, batch, batch_idx):
        if self.annealing:
            self.kl_weight = self.kl_anneal_function(self.anneal_function, self.global_step)
        else:
            self.kl_weight = self.t_kl_weight
        outputs = self.forward(batch)
        r_loss, kl_loss = self.compute_loss(*outputs)
        r = {}
        loss = r_loss + kl_loss
        r['loss'] = loss
        r['r_loss'] = r_loss
        r['kl_loss'] = kl_loss
        r['lr'] = self.lr
        for key in r:
            self.log(f'train/{key}', r[key])
        print(r)
        return r 
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        r_loss, kl_loss = self.compute_loss(*outputs)
        r = {}
        loss = r_loss + kl_loss
        r['loss'] = loss
        r['r_loss'] = r_loss
        r['kl_loss'] = kl_loss
        for key in r:
            self.log(f'val/{key}', r[key])
        return r
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val/loss'}
    
    def create_model(self, hparams):
        return VAE(**hparams)
        
