import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import numpy as np


class TrainModel(pl.LightningModule):
    def __init__(self, model, prices, learning_rate=1e-3, optimizer=None):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.model = model
        self.prices = prices
        self.criterion = nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.optimizer = optimizer
        self.threshold = 0.5
        self.save_hyperparameters() #ignore=['model']

    def forward(self, x):
        return self.model(x)

    def calculate_total_price(self, predictions, labels):
        """
        Berechnet den Gesamtpreis basierend auf den Vorhersagen und tatsächlichen Labels
        """
        # Vorhersagen und tatsächliche Labels basierend auf Threshold filtern
        predicted_labels = [i for i, pred in enumerate(predictions) if pred > self.threshold]
        true_labels = [i for i, label in enumerate(labels) if label > 0.5]

        # Berechnung der Gesamtpreise
        predicted_price = sum(self.prices[label][1] for label in predicted_labels if label in self.prices)
        true_price = sum(self.prices[label][1] for label in true_labels if label in self.prices)

        return predicted_price, true_price

    def step(self, batch, stage):
        x, y = batch
        y_pred = self(x)  # Vorhersagen

        # Hauptverlust berechnen (MSE auf allen Labels)
        loss = self.criterion(y_pred, y)

        # MAE und MSE der Hauptlabels
        mae = self.mae(y_pred, y)
        mse = self.mse(y_pred, y)

        # Berechnung der Gesamtpreise
        batch_predicted_prices = []
        batch_true_prices = []
        for i in range(len(y)):
            predicted_price, true_price = self.calculate_total_price(y_pred[i], y[i])
            batch_predicted_prices.append(predicted_price)
            batch_true_prices.append(true_price)

        # Manuelle Berechnung von MAE und MSE für Gesamtpreise
        total_mae = sum(abs(tp - pp) for tp, pp in zip(batch_true_prices, batch_predicted_prices)) / len(batch_true_prices)
        total_mse = sum((tp - pp) ** 2 for tp, pp in zip(batch_true_prices, batch_predicted_prices)) / len(batch_true_prices)

        # Logs für Progress-Bar und Training/Validierung/Test
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_mse", mse, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_total_mae", total_mae, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_total_mse", total_mse, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, stage="test")

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Scheduler erstellen
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.965)
        return [self.optimizer], [scheduler]


# Lightning Module
class RegressionModel(pl.LightningModule):
    """ Adjustable regression model :num_filters, :pooling"""
    def __init__(self, num_layers=6, num_filters=[32,32,32,32,32,32], pooling = [[1,2,3],[4]], kernel_size=3, learning_rate=1e-3):
        super().__init__()
        layers = []
        in_channels = 3  # Input-Kanäle, eg. RGB-Bilder
        self.learning_rate = learning_rate
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, num_filters[i], kernel_size, padding="same", stride=1))
            layers.append(nn.ReLU())
            if i in pooling[0]: 
                layers.append(nn.MaxPool2d(kernel_size=2))
            elif i in pooling[1]:
                layers.append(nn.MaxPool2d(kernel_size=3))

            in_channels = num_filters[i]
        
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_filters[len(num_filters)-1]* 64, 100))  #adjust size acording to input size and kernel sizes
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, 50)) 
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50, 1))  # Regression Output
        
        self.model = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze(-1) 
        loss = self.criterion(y_pred, y)

        mae = self.mae(y_pred, y)
        mse = self.mse(y_pred, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_mae", mae, prog_bar=True, on_epoch=True)
        self.log("train_mse", mse, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze(-1)  
        loss = self.criterion(y_pred, y)

        mae = self.mae(y_pred, y)
        mse = self.mse(y_pred, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_epoch=True)
        self.log("val_mse", mse, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze(-1)
        loss = self.criterion(y_pred, y)
        mae = self.mae(y_pred, y)
        mse = self.mse(y_pred, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_mae", mae, prog_bar=True)
        self.log("test_mse", mse, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        print(self.learning_rate)

        # initialize a scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.965)
        # Alternativ: oter Scheduler like CosineAnnealingLR, ReduceLROnPlateau, etc.
      
        return [optimizer], [scheduler]
    
    def freeze_backbone(self):
        # Loop through all layers of the model except the last 5
        for param in self.model.parameters():
            param.requires_grad = False

        # Now unfreeze the parameters of the last 5 layers
        for param in self.model[-5:].parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True  
    
    #def train_dataloader(self):

        #return super().train_dataloader()
    
    #def val_dataloader(self):
