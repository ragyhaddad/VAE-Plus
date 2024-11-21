import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn


class Transformer(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embed_size,
        latent_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_seq_length,
        vocab,
        total_steps_annealing=100000,
        num_cycles=100,

    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=vocab.pad_idx)

        # Positional encoding - learnable
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, embed_size)
        )
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            embed_size, num_heads, hidden_dim, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(embed_size, latent_dim)
        self.fc_logvar = nn.Linear(embed_size, latent_dim)
        self.fc_latent_to_hidden = nn.Linear(latent_dim, embed_size)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            embed_size, num_heads, hidden_dim, batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_fc = nn.Linear(embed_size, vocab_size)
        self.vocab = vocab
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=vocab.char2idx[vocab.pad_token], reduction="sum"
        )
        self.kl_weight = 0.6
        self.total_steps_annealing = total_steps_annealing
        self.num_cycles = num_cycles
        self.save_hyperparameters()

    def compute_loss(self, outputs, targets, logvar, mu):
        r_loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss * self.kl_weight
        return r_loss, kl_loss

    def cyclical_annealing(self, T, M, step, R=0.4, max_kl_weight=1):
        """
        T = Total steps
        M = Number of cycles
        R = Proportion used to increase beta
        t = Global step
        """
        period = T / M  # N_iters/N_cycles
        internal_period = (step) % (period)  # Itteration_number/(Global Period)
        tau = internal_period / period
        if tau > R:
            tau = max_kl_weight
        else:
            tau = min(max_kl_weight, tau / R)  # Linear function
        return tau

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def encode(self, x, mask=None):
        emb = self.embedding(x) + self.positional_encoding[:, : x.size(1), :]
        encoded = self.encoder(emb, src_key_padding_mask=mask)

        if mask is not None:
            mask = ~mask.bool()
            lengths = (mask).sum(dim=1)
            last_hidden_state = torch.stack([encoded[i, lengths[i] - 1, :] for i in range(encoded.size(0))])

        else:
            last_hidden_state = encoded[:, -1, :]
              
        mu_vector = self.fc_mu(last_hidden_state)
        logvar_vector = self.fc_logvar(last_hidden_state)
        return last_hidden_state, emb, logvar_vector, mu_vector

    def resize_latent_to_memory(self, last_hidden_state, length):
        resized_latent = last_hidden_state.unsqueeze(1).repeat(1, length, 1)
        return resized_latent

    def decode(self, z, tgt, mask=None):
        z = self.fc_latent_to_hidden(z)
        z = self.resize_latent_to_memory(z, tgt.size(1))
        tgt = self.embedding(tgt) + self.positional_encoding[:, : tgt.size(1), :]

        # Autoregressive training - This allows the model to generate the next token in the sequence without seeing the future tokens.
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        decoded = self.decoder(
            tgt, z, tgt_mask=tgt_mask, tgt_key_padding_mask=mask
        )
        outputs_v = self.output_fc(decoded) # Output logits

        return outputs_v

    def forward(self, batch):
        with_bos, with_eos, masks = batch
        _, _, logvar, mu = self.encode(with_bos, mask=masks)
        z = self.reparameterize(mu, logvar)
        output_v = self.decode(z, with_bos, mask=masks)
        return output_v, with_eos, logvar, mu

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def training_step(self, batch, batch_idx):
        outputs_v, with_eos, logvar, mu = self.forward(batch)
        r_loss, kl_loss = self.compute_loss(outputs_v, with_eos, logvar, mu)
        self.kl_weight = self.cyclical_annealing(
            self.total_steps_annealing, self.num_cycles, step=self.global_step, max_kl_weight=0.6
        )
        r = {}
        r["r_loss"] = r_loss
        r["kl_loss"] = kl_loss
        r["loss"] = r_loss + kl_loss
        r["kl_weight"] = self.kl_weight
        for key in r:
            self.log(f"train_{key}", r[key])
        return r

    def validation_step(self, batch, batch_idx):
        outputs_v, with_eos, logvar, mu = self.forward(batch)
        r_loss, kl_loss = self.compute_loss(outputs_v, with_eos, logvar, mu)
        r = {}
        r["r_loss"] = r_loss
        r["kl_loss"] = kl_loss
        r["loss"] = r_loss + kl_loss
        for key in r:
            self.log(f"val_{key}", r[key])
        return r

    def smiles_to_latent(self, smiles, include_mask=False):
        from vae_cyc.dataset import TransformerSMILESDataset
        ds = TransformerSMILESDataset(smiles, self.vocab, max_len=self.max_seq_length)
        dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=ds.collate)
        with torch.no_grad():
            for batch in dl:
                with_bos, _, masks = batch
                if include_mask:
                    _, _, _, mu = self.encode(with_bos.to(self.device), mask=masks.to(self.device))
                else:
                    _, _, _, mu = self.encode(with_bos.to(self.device), mask=None)
                latents.append(mu)
        latents = torch.cat(latents, dim=0)
        return latents

    def smiles_to_hidden(self, smiles, include_mask=False):
        smiles = self.vocab.encode_special(smiles[0])
        input_tokens = torch.tensor([self.vocab.sos_idx] + [self.vocab.char2idx[i] for i in smiles]).unsqueeze(0)
        input_tokens = input_tokens.to(self.device)
        _, _, _, hidden = self.encode(input_tokens, mask=None)
        return hidden

    def generate(self, smiles=None, include_mask=False, argmax=False):
        current_tokens = [self.vocab.char2idx[self.vocab.sos_token]]
        tgt_tokens = torch.tensor([current_tokens]).long().to(self.device)
        s = ""
        if smiles is not None:
            z = self.smiles_to_hidden(smiles, include_mask=include_mask)
        else:      
            z = torch.normal(0, 1, size=(1, self.latent_dim)).to(self.device)
        tgt_tokens = self.embedding(tgt_tokens) + self.positional_encoding[:, : tgt_tokens.size(1), :]
        
        z = self.fc_latent_to_hidden(z)
        z = self.resize_latent_to_memory(z, tgt_tokens.size(1)) 
        
        with torch.no_grad():
            for i in range(self.max_seq_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1)).to(self.device)
                decoded = self.decoder(tgt_tokens,z, tgt_mask=tgt_mask)
                output_v = self.output_fc(decoded)
                output_v = output_v[:, -1, :]
                if argmax:
                    top_char = torch.argmax(output_v)
                else:
                    top_char = torch.multinomial(F.softmax(output_v, dim=1), 1)
                if top_char == self.vocab.eos_idx:
                    break 
                current_tokens.append(top_char.item())
                tgt_tokens = torch.tensor([current_tokens]).long().to(self.device)
                
                tgt_tokens = self.embedding(tgt_tokens) + self.positional_encoding[:, : tgt_tokens.size(1), :]
                s += self.vocab.idx2char[top_char.item()]
                
        return s

