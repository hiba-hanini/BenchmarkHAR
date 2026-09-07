import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ISWC22_HAR.models.TinyHAR import TinyHAR_Model
from src.models.base_model import BaseHARModel
from src.utils.reproducibility import DEFAULT_SEED
from types import SimpleNamespace



class TinyHARWrapper(BaseHARModel):

    def __init__(self, config):
        super().__init__(config)
        params = self.config.get("params", {})
        self.model_name = config["name"]
        self.epochs = params.get("epochs", 100)
        self.batch_size = params.get("batch_size", 32)
        self.seed = self.config.get("seed", DEFAULT_SEED)
        self.model = None
        self._input_shape = None
        self._n_classes = int(self.config.get("n_classes", 1))
        self.device = torch.device(
         "cuda" if torch.cuda.is_available() else "cpu"
        )


    def fit(
        self,
        X_train,
        y_train,
        n_classes,
        verbose=0,
        validation_data=None,
        callbacks=None,
        **kwargs
    ):
    
        print("[FIT] TinyHARWrapper")
    
        self._n_classes = int(n_classes)
        self._input_shape = X_train.shape[1:]
    
        self.model = TinyHAR_Model(
            input_shape=self._input_shape,
            number_class=self._n_classes,
            filter_num=32,
        ).to(self.device)
    
        # Convert X
        if isinstance(X_train, np.ndarray):
            X_train = np.ascontiguousarray(X_train)
            X_train = torch.from_numpy(X_train).float()
        else:
            X_train = torch.tensor(X_train, dtype=torch.float32)
    
        # Convert y
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).long()
        else:
            y_train = torch.tensor(y_train, dtype=torch.long)
    
        # TinyHAR expects (B, 1, T, F)
        if X_train.dim() == 5:
            X_train = X_train.squeeze(2)
    
        print(
            f"[DEBUG] X_train tensor shape={X_train.shape}, "
            f"dtype={X_train.dtype}"
        )
        print(
            f"[DEBUG] y_train tensor shape={y_train.shape}, "
            f"dtype={y_train.dtype}"
        )
    
        generator = torch.Generator()
        generator.manual_seed(self.seed)
    
        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
    
        criterion = torch.nn.CrossEntropyLoss()
    
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )
    
        # --------------------------------------------------
        # History
        # --------------------------------------------------
        history = {
            "loss": []
        }
    
        # Validation data
        val_loader = None
    
        if validation_data is not None:
    
            X_val, y_val = validation_data
    
            if isinstance(X_val, np.ndarray):
                X_val = np.ascontiguousarray(X_val)
                X_val = torch.from_numpy(X_val).float()
            else:
                X_val = torch.tensor(X_val, dtype=torch.float32)
    
            if isinstance(y_val, np.ndarray):
                y_val = torch.from_numpy(y_val).long()
            else:
                y_val = torch.tensor(y_val, dtype=torch.long)
    
            if X_val.dim() == 5:
                X_val = X_val.squeeze(2)
    
            val_loader = DataLoader(
                TensorDataset(X_val, y_val),
                batch_size=self.batch_size,
                shuffle=False,
            )
    
            history["val_loss"] = []
    
        # --------------------------------------------------
        # Training
        # --------------------------------------------------
        self.model.train()
    
        for epoch in range(self.epochs):
    
            total_loss = 0.0
    
            for batch_idx, (xb, yb) in enumerate(loader):
    
                xb = xb.to(self.device)
                yb = yb.to(self.device)
    
                optimizer.zero_grad()
    
                # Time first forward pass
                if epoch == 0 and batch_idx == 0:
                    t0 = time.time()
    
                    logits = self.model(xb)
    
                    t1 = time.time()
    
                    print(
                        f"[DEBUG] first forward time="
                        f"{t1-t0:.4f}s "
                        f"logits_shape={getattr(logits, 'shape', None)}"
                    )
    
                else:
                    logits = self.model(xb)
    
                loss = criterion(logits, yb)
    
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            # Average training loss
            epoch_loss = total_loss / len(loader)
    
            history["loss"].append(epoch_loss)
    
            # --------------------------------------------------
            # Validation loss
            # --------------------------------------------------
            if val_loader is not None:
    
                self.model.eval()
    
                val_total_loss = 0.0
    
                with torch.no_grad():
    
                    for xb, yb in val_loader:
    
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
    
                        logits = self.model(xb)
    
                        loss = criterion(logits, yb)
    
                        val_total_loss += loss.item()
    
                val_loss = val_total_loss / len(val_loader)
    
                history["val_loss"].append(val_loss)
    
                self.model.train()
    
            # --------------------------------------------------
            # Verbose output
            # --------------------------------------------------
            if verbose:
    
                msg = (
                    f"Epoch {epoch+1}/{self.epochs} "
                    f"loss={epoch_loss:.4f}"
                )
    
                if val_loader is not None:
                    msg += f" val_loss={val_loss:.4f}"
    
                print(msg)
    
        self.is_fitted = True
    
        # Keras-like History object
        return SimpleNamespace(history=history)


    def predict(self, X_test):

        if self.model is None:
            raise RuntimeError("TinyHARWrapper.predict() called before fit().")

        self.model.eval()


        if isinstance(X_test, np.ndarray):
            X_test = np.ascontiguousarray(X_test)
            X_test = torch.from_numpy(X_test).float()
        else:
            X_test = torch.tensor(X_test, dtype=torch.float32)

        # mirror the reshape done in `fit`
        if X_test.dim() == 5:
            X_test = X_test.squeeze(2)

        loader = DataLoader(
            TensorDataset(X_test),
            batch_size=self.batch_size
        )


        preds = []

        with torch.no_grad():

            for (xb,) in loader:

                xb = xb.to(self.device)

                logits = self.model(xb)

                pred = torch.argmax(
                    logits,
                    dim=1
                )

                preds.extend(
                    pred.cpu().numpy()
                )


        return np.array(preds)



    def save(self, path):
        self._check_is_fitted()
        torch.save(
            self.model.state_dict(),
            path
        )

    def load(self, path):
        if self.model is None:
            if self._input_shape is None:
                raise RuntimeError(
                    "TinyHARWrapper.load() requires a fitted/training-time input shape. "
                    "Set self._input_shape before calling load()."
                )
            self.model = TinyHAR_Model(
                input_shape=self._input_shape,
                number_class=self._n_classes,
                filter_num=32,
            ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.is_fitted = True

        return self
