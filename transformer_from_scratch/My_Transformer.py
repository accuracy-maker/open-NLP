import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# define class self-attention module
class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        is_plot=False
        
    ):
        """ init
        Parameters
        ----------
            embed_size: [int] the dimention of embedding
            heads: [int] the number of attention heads
        
        """
        super(SelfAttention,self).__init__()
        self.is_plot = is_plot
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads
        
        assert self.embed_size % self.heads == 0, "heads should be divided by embed size"
        
        self.query = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim,bias=False)
        
        self.fc = nn.Linear(self.head_dim * self.heads,self.embed_size)
    
    def plot_attn(self, attention_matrix, xlabel="Query", ylabel="Key"):
        """
        Plot the attention matrix as a heatmap.

        Parameters:
        - attention_matrix: The attention matrix to be plotted.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(attention_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Self-Attention Heatmap")
        plt.show()
    
    
    
    def forward(self,query,key,value,mask):
        """foward
        
        Parameters
        ----------
            query: [torch.Tensor] shape of query `Batch, seq_len, embed_size`
            key: [torch.Tensor] shape of key `Batch, seq_len, embed_size`
            value: [torch.Tensor] shape of value `Batch, seq_len, embed_size`
            mask: mask makes the attention map causal
            is_plot: whether to plot the figure of attn
            
        Returns
        -------
            attn: `Batch,heads,q_len,k_len`
            out: `Batch,seq_len,embed_size`          
            
        """
        batch = query.shape[0]
        
        # in the self-attention, the query,key,value are the same
        query_len,key_len,value_len = query.shape[1],key.shape[1],value.shape[1]
        
        # split the embed size into heads*head_dim
        query = query.view(batch,query_len,self.heads,self.head_dim)
        key = key.view(batch,key_len,self.heads,self.head_dim)
        value = value.view(batch,value_len,self.heads,self.head_dim)
        
        # linear map
        queries = self.query(query)
        keys = self.key(key)
        values = self.value(value)
        
        # compute the energy Q * K^T
        # queries: `batch,q_len,heads,head_dim`
        # keys: `batch,k_len,heads,head_dim`
        # energy: `batch,heads,q_len,k_len`
        energy = torch.einsum("bqhd,bkhd->bhqk",[queries,keys])
        
        # note: if you wanna use attn map, you must consider the mask that will make the attn causal
        if mask is not None:
            energy = energy.masked_fill(mask==0,float("1e-20"))
            
        attention = torch.softmax(
            energy / (self.embed_size ** (0.5)),
            dim=3
        )
        
        if self.is_plot:
            self.plot_attn(attention[0, 0].detach().numpy(), xlabel="Query Position", ylabel="Key Position")
        
        
        # compute the output
        # energy: `batch,heads,q_len,k_len`
        # values: `batch,v_len,heads,head_dim`
        # out: `batch,q_len,heads,head_dim` --> `batch,q_len,embed_size`
        out = torch.einsum("bhql,blhd->bqhd",[attention,values]).reshape(batch,query_len,self.heads*self.head_dim)
        
        # linear map
        out = self.fc(out)
        
        return attention,out
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        forward_expansion,
        is_plot=False
    ):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads,is_plot=is_plot)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,query,key,value,mask):
        attn,out = self.attention(query,key,value,mask)
        x = self.dropout(self.norm1(query + out))
        
        out = self.dropout(self.norm2(x + self.feed_forward(x)))
        
        return out

# Transformer Encoder includes transformer block, postional encoding and word embedding
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        heads,
        device,
        num_layers,
        dropout,
        forward_expansion,
        max_length,
        is_plot=False
    ):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
        self.positional_embedding = nn.Embedding(max_length,embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion,
                    is_plot=is_plot
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,src,mask):
        """ processing of forward
        
        Parameters
        ----------
            src: torch.Tensor `N(batch),seq_len)`
            mask: src_mask 0 representing padding
            
        Returns
        -------
            out: torch.Tensor `N,seq_len,embed_size`
        """
        
        N,seq_len = src.shape
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        
        out = self.dropout(self.word_embedding(src) + self.positional_embedding(positions))
        
        for layer in self.layers:
            out = layer(out,out,out,mask)
        
        return out
    
# DecoderBlock includes Masked attention + TransformerBlock    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        forward_expansion,
        is_plot=False
    ):
        super(DecoderBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads,is_plot=is_plot)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size,heads,dropout,forward_expansion,is_plot=is_plot)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,value,key,src_mask,trg_mask):
        """processing of forward
        
        Parameters
        ----------
            x: input of decoder(right shift traget embedding) `N,seq_len,embed_size`
            value,key: output of the encoder `N,seq_len,embed_size`
            src_mask: padding mask
            trg_mask: causal mask
            
        Returns
        -------
            out: `N,seq_len,emb_size`
        
        """
        attn,out = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(x + out))
        output = self.transformer_block(query,key,value,src_mask)
        
        return output


# Transformer Decoder includes N layers of decoder block + Linear + softmax 
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        heads,
        device,
        num_layers,
        forward_expansion,
        dropout,
        max_len,
        is_plot=False
    ):
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_len,embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion,
                    is_plot=is_plot
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,enc_out,src_mask,trg_mask):
        """processing of foward
        
        Parameters
        ----------
            x: right shift of trg `N,seq_len`
            enc_out: `N,seq_len,embed_size`
            src_mask: padding mask
            trg_mask: causal mask

        Returns
        -------
            logits: output probabilities `N,seq_len,1`
        """
        
        N,seq_len = x.shape
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            x = layer(x,enc_out,enc_out,src_mask,trg_mask)
        
        out  = self.fc_out(x)
        
        #out `N,seq_len,trg_vocab`
        out = torch.softmax(out,dim=2)
        
        return out
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.,
        device='cpu',
        max_len=100,
        is_plot=False
    ):
        super(Transformer,self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            heads,
            device,
            num_layers,
            dropout,
            forward_expansion,
            max_len,
            is_plot=is_plot
        )
        
        self.decoder=Decoder(
            trg_vocab_size,
            embed_size,
            heads,
            device,
            num_layers,
            forward_expansion,
            dropout,
            max_len,
            is_plot=is_plot
        )
        
        self.src_padding_idx = src_pad_idx
        self.trg_padding_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self,src):
        #src: `N,seq_len`
        src_mask = (src != self.src_padding_idx).unsqueeze(1).unsqueeze(2)
        # src_mask `N,1,1,src_len`
        return src_mask.to(self.device)
    
    def make_trg_mask(self,trg):
        N,trg_len = trg.shape
        
        trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(N,1,trg_len,trg_len)
        
        return trg_mask.to(self.device)
    
    def forward(self,src,trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src,src_mask)
        out  = self.decoder(trg,enc_src,src_mask,trg_mask)
        return out
        
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.tensor([
        [1,5,6,4,3,9,5,2,0],
        [1,8,7,3,4,5,6,7,2]
    ]).to(device)
    
    trg = torch.tensor([
        [1,7,4,3,5,9,2,0],
        [1,5,6,2,4,7,6,2]
    ]).to(device)
    
    src_padding_idx = 0
    trg_padding_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    
    model = Transformer(src_vocab_size,trg_vocab_size,src_padding_idx,trg_padding_idx).to(device)
        
    out = model(x,trg[:,:-1])
    print(out.shape)