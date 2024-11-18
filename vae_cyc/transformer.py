import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import math
from torch import nn

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Create the positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class TransformerWithLearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.embedding(positions)

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
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.positional_encoding = PositionalEncoding(embed_size, max_seq_len   =max_seq_length)

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
        Implementing: <https://arxiv.org/abs/1903.10145>
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
        # emb = self.embedding(x) + self.positional_encoding[:, : x.size(1), :]
        emb = self.embedding(x) + self.positional_encoding(x)
        encoded = self.encoder(emb, src_key_padding_mask=mask)
        # masked encodings
        if mask is not None:
            # Pooled average 
            # expanded_mask = mask.unsqueeze(-1).repeat(1, 1, encoded.size(-1))
            # masked_encodings = (expanded_mask * encoded)
            # sum_encodings = masked_encodings.sum(dim=1) # sum over the sequence length
            # non_padded_tokens = mask.sum(dim=1)
            # last_hidden_state = sum_encodings / non_padded_tokens.unsqueeze(-1)

            last_non_padding_indices = mask.sum(dim=1) - 1  # Shape: (batch_size,)
            last_non_padding_indices = last_non_padding_indices.long()
            last_hidden_state = torch.stack(
                [encoded[i, last_non_padding_indices[i], :] for i in range(encoded.size(0))])
        else:
            last_hidden_state = encoded[
                :, -1, :
            ]  # z latent is the last hidden state of the encoded sequence (but account for padding) this implementation is not yet doing that

        mu_vector = self.fc_mu(last_hidden_state)
        logvar_vector = self.fc_logvar(last_hidden_state)
        return last_hidden_state, emb, logvar_vector, mu_vector

    def resize_latent_to_memory(self, last_hidden_state, length):
        resized_latent = last_hidden_state.unsqueeze(1).repeat(1, length, 1)
        return resized_latent

    def decode(self, z, tgt, mask=None):
        z = self.fc_latent_to_hidden(z)
        z = self.resize_latent_to_memory(z, tgt.size(1))
        # memory = z + self.positional_encoding[:, : tgt.size(1), :]
        memory = z + self.positional_encoding(tgt)

        # Autoregressive training - This allows the model to generate the next token in the sequence without seeing the future tokens.
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        decoded = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=mask,
        )

        outputs_v = self.output_fc(decoded)  # Output logits

        return outputs_v

    def forward(self, batch):
        with_bos, with_eos, masks = batch
        last_hidden_state, memory, logvar, mu = self.encode(with_bos, mask=masks)
        z = self.reparameterize(mu, logvar)
        output_v = self.decode(z, memory, mask=masks)
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

    def smiles_to_latent(self, smiles):
        from vae_cyc.dataset import TransformerSMILESDataset
        ds = TransformerSMILESDataset(smiles, self.vocab, max_len=self.max_seq_length)
        dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=ds.collate)

        latents = []
        with torch.no_grad():
            for batch in dl:
                with_bos, _, masks = batch
                _, _, logvar, mu = self.encode(with_bos.to(self.device), mask=None)
                latents.append(mu)
        latents = torch.cat(latents, dim=0)
        return latents

    def generate(self, num_samples=1, smiles=None):
        current_tokens = [self.vocab.char2idx[self.vocab.sos_token]]
        tgt_tokens = torch.tensor([current_tokens]).long().to(self.device)
        s = ""
        if smiles is not None:
            last_hidden_state = self.smiles_to_latent(smiles)
        else:
            last_hidden_state = torch.normal(0, 1, size=(1, self.latent_dim)).to(self.device)
        last_hidden_state = self.fc_latent_to_hidden(last_hidden_state)
        memory = self.resize_latent_to_memory(last_hidden_state, 1)
        emb = self.embedding(tgt_tokens) + self.positional_encoding(tgt_tokens)
        # print(self.positional_encoding(tgt_tokens).shape)
        with torch.no_grad():
            for i in range(self.max_seq_length):
                tgt_mask = self.generate_square_subsequent_mask(emb.size(1)).to(emb.device)
                decoded = self.decoder(emb, memory, tgt_mask=tgt_mask)
                output_v = self.output_fc(decoded)
                output_v = output_v[:, -1, :]
                top_char = torch.argmax(output_v)
                if top_char == self.vocab.eos_idx:
                    break 
                current_tokens.append(top_char.item())
                tgt_tokens = torch.tensor([current_tokens]).long().to(self.device)
                emb = self.embedding(tgt_tokens) + self.positional_encoding(tgt_tokens)
                s += self.vocab.idx2char[top_char.item()]
            print(s)
        return s

