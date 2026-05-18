import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import torchmetrics
from src.utils.nn.modules import LoggedLitModule, LoggedImageClassifierModule


class ConcreteLoggedLitModule(LoggedLitModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam
        self.optimizer_params = {"lr": 1e-3}

    def forward(self, xs):
        return self.fc(xs)


class TestLoggedLitModule:

    def test_all_steps_return_loss_and_predictions(self):
        """training/validation/test steps compute loss, log scalars, and return y_hats."""
        module = ConcreteLoggedLitModule()
        module.log_dict = MagicMock()

        xs = torch.randn(8, 4)
        ys = torch.randint(0, 2, (8,))
        batch = (xs, ys)

        for step_fn, step_name in [
            (module.training_step, "training"),
            (module.validation_step, "validation"),
            (module.test_step, "test"),
        ]:
            result = step_fn(batch, idx=0)
            assert "loss" in result and "y_hats" in result
            assert result["loss"].ndim == 0
            assert result["y_hats"].shape == (8, 2)

        logged_keys = [str(c) for c in module.log_dict.call_args_list]
        assert any("training/" in s for s in logged_keys)
        assert any("validation/" in s for s in logged_keys)
        assert any("test/" in s for s in logged_keys)

    def test_metrics_logged_and_optimizer_configured(self):
        """Attached metrics are called during steps; configure_optimizers returns an optimizer."""
        module = ConcreteLoggedLitModule()
        module.log_dict = MagicMock()

        # Use a real nn.Module-compatible metric (ModuleList rejects non-Module objects)
        real_metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        module.training_metrics.append(real_metric)

        xs = torch.randn(8, 4)
        ys = torch.randint(0, 2, (8,))
        result = module.training_step((xs, ys), idx=0)

        assert "training/multiclassaccuracy" in {
            k: v for call in module.log_dict.call_args_list for k, v in call[0][0].items()
        }

        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)


class TestLoggedImageClassifierModule:

    def test_accuracy_tracked_and_dtype_fix_applied(self):
        """
        Encapsulates the full LoggedImageClassifierModule contract:
        - train/val/test accuracy metrics are wired into their respective metric lists
        - a normal int-target batch flows through without errors
        - float ys are cast to int for Accuracy (binary MSELoss case)
        """

        class ConcreteClassifier(LoggedImageClassifierModule):
            def __init__(self):
                super().__init__()
                # Match num_classes=2 hardcoded in LoggedImageClassifierModule
                self.fc = nn.Linear(4, 2)
                self.loss = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.SGD
                self.optimizer_params = {"lr": 0.01}

            def forward(self, xs):
                return self.fc(xs)

        module = ConcreteClassifier()
        module.log_dict = MagicMock()

        # 1. Metric lists are populated
        assert len(module.training_metrics) == 1
        assert len(module.validation_metrics) == 1
        assert len(module.test_metrics) == 1

        # 2. Normal int-target forward pass
        xs = torch.randn(8, 4)
        ys_int = torch.randint(0, 2, (8,))
        result = module.training_step((xs, ys_int), idx=0)
        assert "loss" in result
        assert result["y_hats"].shape == (8, 2)

        # 3. Float ys path — dtype coercion to int before Accuracy
        logging_scalars = {}
        y_hats = torch.randn(8, 2)
        ys_float = torch.randint(0, 2, (8,)).float()
        module.log_metric(module.train_acc, logging_scalars, y_hats, ys_float)
        assert "multiclassaccuracy" in logging_scalars
        assert isinstance(logging_scalars["multiclassaccuracy"], torch.Tensor)
