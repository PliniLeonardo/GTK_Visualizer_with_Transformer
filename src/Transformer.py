import torch
import torch.nn as nn
import math
from itertools import combinations_with_replacement
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from focal_loss.focal_loss import FocalLoss


def find_all_combinations(numbers):
    """
    Find all possible combinations of two numbers from the given list,
    including self-combinations (n, n) while maintaining the order.
    """
    result = list(combinations_with_replacement(numbers, 2))
    
    return result

#--------------- LIGHTNING TRANSFORMER CLASS--------------------
class LightningEncoder(pl.LightningModule):
    def __init__(self, config ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = config["model"]["input_dim"]
        self.d_model = config["model"]["d_model"]
        self.nhead = config["model"]["num_head"]
        self.num_layers = config["model"]["num_layers"]
        self.hidden_dim_classifier = config["model"]["hidden_dim_classifier"]
        self.dropout = config["model"]["dropout"]
        self.lr = config["model"]["lr"]
        self.lr_scheduler_step_size = config["model"]["lr_scheduler_step_size"]
        self.lr_scheduler_gamma = config["model"]["lr_scheduler_gamma"]

        if config["loss"] == "CrossEntropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif config["loss"] == "Focal":
            self.loss_fn = FocalLoss(gamma=2)
        else:
            raise ValueError(f"Loss '{config['loss']}' non supportata")
        self.model = Encoder( config)
            

    def forward(self, data, df, max_len):
        return self.model(data, df, max_len)

    def training_step(self, batch, batch_idx):
        data, gt, df, max_len, _ = batch
        output = self(data, df, max_len)
        # Adatta la loss in base al formato di gt
        if gt.shape[-1] == 2:
            loss = self.loss_fn(output.view(-1, 2), gt.argmax(dim=-1).view(-1))
        else:
            loss = self.loss_fn(output.view(-1, 2), gt.view(-1).long())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, gt, df, max_len, _ = batch
        output = self(data, df, max_len)
        if gt.shape[-1] == 2:
            loss = self.loss_fn(output.view(-1, 2), gt.argmax(dim=-1).view(-1))
        else:
            loss = self.loss_fn(output.view(-1, 2), gt.view(-1).long())
        self.log('val_loss', loss, prog_bar=True, batch_size=data.size(0))
        return loss
    
    def test_step(self, batch, batch_idx):
        data, gt, df, max_len, _ = batch
        output = self(data, df, max_len)
        if gt.shape[-1] == 2:
            loss = self.loss_fn(output.view(-1, 2), gt.argmax(dim=-1).view(-1))
        else:
            loss = self.loss_fn(output.view(-1, 2), gt.view(-1).long())
        acc = (output.argmax(dim=-1) == gt.argmax(dim=-1)).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {"loss": loss, "acc": acc}


    # def configure_optimizers(self):
    #     return optim.Adam(self.parameters(), lr=self.lr)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= self.lr_scheduler_step_size, gamma= self.lr_scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # oppure "step" se vuoi aggiornare ogni batch
                "frequency": 1,
            }
        }

#--------------- END OF LIGHTNING TRANSFORMER CLASS--------------------

# ------------------- ORIGINAL TRANSFORMER CLASS ----------------------
class Encoder(nn.Module):
    def __init__(self, config) :
        self.input_dim =config["model"]["input_dim"]
        self.d_model = config["model"]["d_model"]
        self.nhead = config["model"]["num_head"]
        self.num_layers = config["model"]["num_layers"]
        self.hidden_dim_classifier = config["model"]["hidden_dim_classifier"]
        self.dropout = config["model"]["dropout"]
        self.device = config["model"]["device"]
    
        super(Encoder, self).__init__()   
        self.embedding = nn.Linear(self.input_dim, self.d_model)    

        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.hidden_dim_classifier, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        #MLP
        self.fc1 = nn.Linear(self.d_model*2, self.d_model*2)
        self.fc2 = nn.Linear(self.d_model*2, 2)


    def forward(self, data, df,max_len): # max_edges):
        #embedding
        data = self.embedding(data.to(self.device)) #data.shape = (batch_size, num_nodes, input_dim)
        #transformer encoder
        transformer_output = self.transformer_encoder(data) 

        #MLP
        #create tensor for MLP
        dims = [dataframe.shape[0] for dataframe in df]
        node_indices = [find_all_combinations(list(range(dim))) for dim in dims]
        mlp_tensor_to_pad = [transformer_output[i,node_indices[i]] for i in range(len(node_indices))]
        pad_amount = [ (max_len**2)- x.shape[0] for x in mlp_tensor_to_pad]
        #[ (max_len**2)- x.shape[0] for x in mlp_tensor_to_pad]
        #[ 4*((max_len+5)**2) -x.shape[0] for x in mlp_tensor_to_pad]
        mlp_tensor = [F.pad(x, (0, 0, 0, 0, 0, padding)) for x,padding in zip(mlp_tensor_to_pad, pad_amount) ]
        mlp_tensor = [x.view(x.shape[0],-1) for x in mlp_tensor]
        mlp_input = torch.cat([x.unsqueeze(0) for x in mlp_tensor], dim =0).to(self.device)

        mlp_output = torch.relu(self.fc1(mlp_input))
        mlp_output = self.fc2(mlp_output)
        # mlp_output = torch.sigmoid(mlp_output)
        mlp_output = torch.softmax(mlp_output, dim = -1)

        return mlp_output
        
# ------------------- END OF ORIGINAL TRANSFORMER CLASS ----------------------


### ONNX Equivalent model

class EncoderONNX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['model']['input_dim']
        self.d_model = config['model']['d_model']
        self.num_head = config['model']['num_head']
        self.num_layers = config['model']['num_layers']
        self.hidden_dim_classifier = config['model']['hidden_dim_classifier']
        self.dropout = config['model']['dropout']

        # Layers
        self.embedding = nn.Linear(self.input_dim, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_head,
            dim_feedforward=self.hidden_dim_classifier,
            dropout=self.dropout,
            batch_first=False  # Match Lightning model
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.fc1 = nn.Linear(self.d_model * 2, self.d_model * 2)
        self.fc2 = nn.Linear(self.d_model * 2, 2)

    def forward(self, data, pair_indices):
        """
        data: (N, input_dim)  --> no batch dimension
        pair_indices: (K, 2) with pairs (i, j)
        output: (1, K, 2) where K = number of pairs - MANTIENE LA DIMENSIONE BATCH
        """
        # Add batch dimension and transpose for batch_first=False
        x = data.unsqueeze(1).transpose(0, 1)  # (1, N, input_dim)

        # Transformer
        x = self.embedding(x)
        x = self.transformer_encoder(x)  # (1, N, d_model)
        x = x.transpose(0, 1)  # (N, 1, d_model) - NON rimuovere la dimensione batch

        # Use provided pair indices - mantieni la dimensione batch
        x_i = x[pair_indices[:, 0]]  # (K, 1, d_model)
        x_j = x[pair_indices[:, 1]]  # (K, 1, d_model)

        # MLP on pairs
        pairs = torch.cat([x_i, x_j], dim=-1)  # (K, 1, 2*d_model)
        out = torch.relu(self.fc1(pairs))      # (K, 1, 2*d_model)
        out = self.fc2(out)                    # (K, 1, 2)
        out = torch.softmax(out, dim=-1)       # (K, 1, 2)
        
        # Transpose per ottenere (1, K, 2)
        out = out.transpose(0, 1)              # (1, K, 2)

        return out