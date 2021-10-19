import utils

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from transformers import AutoTokenizer, AutoModel

import operator as op
import itertools as it
import functools as ftools
import tqdm

class LMModel(nn.Module):
    def __init__(self,
            classification_targets=[],
            lm="bert-base-uncased",
            embed_strategy="cls",
            freeze_lm=False,
            max_length=256,
            epochs=3,
            batch_size=16,
            optimizer=torch.optim.Adam,
            optimizer_params=dict(lr=5e-5),
            device=utils.DEVICE):
        super(LMModel,self).__init__()
        
        tokenizer = AutoTokenizer.from_pretrained(lm)
        lm = AutoModel.from_pretrained(lm).to(device)
        
        classification_heads = [nn.Linear(768,ct) for ct in classification_targets]
        softmax = nn.Softmax(dim=1)
        
        print(list(map(op.methodcaller("parameters"),classification_heads)))
        
        loss = nn.CrossEntropyLoss()
        optimizer = optimizer(params=[*lm.parameters(),
                                      *it.chain(*map(op.methodcaller("parameters"),classification_heads))],**optimizer_params)
        
        ## Set attributes
        # Config
        self.embed_strategy = embed_strategy
        self.max_length = max_length
        self.device = device
        
        # Tokenizer
        self.tokenizer = tokenizer
        
        # Layers
        self.lm = lm
        self.classification_heads = classification_heads
        self.softmax = softmax
        
        # Training
        self.freeze_lm = freeze_lm
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.loss = loss
        self.optimizer = optimizer

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None):
        x = self.lm(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)[0]
        
        x = x[:,0,:] if self.embed_strategy == "cls" else torch.mean(x,1)
        y = [self.softmax(ch(x)) for ch in self.classification_heads]
        
        return y or x # The classification or the embeddings
    
    ## Utility functions
    def fit(self, X_train, y_train): # Only works with classification heads
        self._train(X_train,y_train)
            
    def predict(self,X):
        dl = self._convert2batched(X)
        dl = tqdm.tqdm(dl) # Add progress bar
        
        self.lm.eval()
        with torch.no_grad():
            if not self.classification_heads:
                x = torch.cat(tuple(map(self._predict,dl)))
                x = x.cpu().numpy()
            else:
                x = zip(*map(self._predict,dl))
                x = map(torch.cat,x)
                x = map(op.methodcaller("cpu"),x)
                x = map(op.methodcaller("numpy"),x)
                x = list(x)
                
        return x
        
    ## Private functions
    def _convert2batched(self,X,ys=None,shuffle=False):
        # Convert to tensors
        X = self.tokenizer(
            X, padding=True, max_length=self.max_length,
            truncation=True, return_tensors="pt").data
        ds = TensorDataset(*X.values())
        
        if ys:
            ys = map(torch.tensor,ys)
            ys = list(map(TensorDataset,ys))
            ys = ConcatDataset(ys)
        
            ds = ConcatDataset((ds,ys))
        return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
                pin_memory=True,
                collate_fn=lambda x:x)
    
    def _train(self, X_train, y_train):
        dl = self._convert2batched(X_train,y_train,shuffle=True)
        
        for e in range(self.epochs):
            self._train_epoch(dl)
            
    def _train_epoch(self,dl: DataLoader):
        for batch in tqdm.tqdm(dl):
            print(len(batch))
            X, ys = batch
            self._train_step(X,ys)
            
    def _train_step(self,X,ys):
        self.optimizer.zero_grad()
        
        ys_pred = self._predict(X)
        loss = self._compute_loss(ys_pred,ys)
        
        loss.backward()
        self.optimizer.step()
        
    def _predict(self,X):
        input_ids, token_type_ids, attention_mask = map(op.methodcaller("to",self.device),X)
        return self(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
    
    def _compute_loss(self,y_pred,y_true):
        y_true = map(op.methodcaller("to",self.device),y_true)
        
        losses = list(it.starmap(self.loss,zip(ys_pred,y_true)))
        return torch.mean(losses)
        
            
        