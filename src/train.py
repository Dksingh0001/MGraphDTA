"""Training and validation pipeline for Davis MGraphDTA regression."""

import argparse
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from src.dataset import DavisDataset
from src.models.mgraphdta import MGraphDTA


DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 3000
DEFAULT_STEPS_PER_EPOCH = 50
DEFAULT_PATIENCE = 400
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_SEED = 42
DeviceArg = Optional[Union[str, torch.device]]


class _PyGDataView(Dataset):
	"""Expose each Davis dictionary item's PyG graph for graph-aware collation."""

	def __init__(self, dataset: Any):
		self.dataset = dataset

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, index: int) -> Any:
		item = self.dataset[index]
		if isinstance(item, dict):
			try:
				return item["pyg_data"]
			except KeyError as exc:
				raise ValueError(
					"Dictionary dataset items must contain a 'pyg_data' graph"
				) from exc
		return item


def resolve_device(device: DeviceArg = None) -> torch.device:
	"""Resolve auto mode to CUDA when available and CPU otherwise."""
	if device is None or device == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")

	resolved = torch.device(device)
	if resolved.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available")
	return resolved


def set_random_seed(seed: int) -> None:
	"""Seed Python, NumPy, PyTorch, and available CUDA generators."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def make_dataloader(
	dataset: Any,
	batch_size: int = DEFAULT_BATCH_SIZE,
	shuffle: bool = False,
	num_workers: int = 0,
) -> DataLoader:
	"""Build a PyG loader with the requested batch size."""
	if batch_size < 1:
		raise ValueError("batch_size must be positive")
	if num_workers < 0:
		raise ValueError("num_workers cannot be negative")
	return DataLoader(
		_PyGDataView(dataset),
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
	)


def split_train_validation(
	dataset: Dataset,
	validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
	seed: Optional[int] = DEFAULT_SEED,
) -> tuple[Subset, Subset]:
	"""Split only the training dataset into disjoint train and validation subsets."""
	if not 0 < validation_fraction < 1:
		raise ValueError("validation_fraction must be between 0 and 1")
	if len(dataset) < 2:
		raise ValueError("dataset must contain at least two samples")

	validation_size = min(
		len(dataset) - 1,
		max(1, round(len(dataset) * validation_fraction)),
	)
	generator = torch.Generator()
	if seed is not None:
		generator.manual_seed(seed)
	indices = torch.randperm(len(dataset), generator=generator).tolist()
	validation_indices = indices[:validation_size]
	training_indices = indices[validation_size:]
	return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def _batch_loss(
	model: nn.Module,
	batch: Any,
	criterion: nn.Module,
	device: torch.device,
) -> tuple[Tensor, int]:
	batch = batch.to(device)
	prediction = model(batch).reshape(-1)
	target = batch.y.reshape(-1)
	if prediction.shape != target.shape:
		raise ValueError(
			f"Prediction shape {prediction.shape} does not match target shape {target.shape}"
		)
	return criterion(prediction, target), target.numel()


def train_one_epoch(
	model: nn.Module,
	dataloader: Iterable[Any],
	optimizer: Optimizer,
	criterion: Optional[nn.Module] = None,
	device: DeviceArg = None,
	max_steps: Optional[int] = None,
) -> float:
	"""Run one training epoch, optionally bounded to a fixed number of batches."""
	if max_steps is not None and max_steps < 1:
		raise ValueError("max_steps must be positive")

	resolved_device = resolve_device(device)
	model.to(resolved_device)
	model.train()
	loss_function = criterion or nn.MSELoss()
	iterator = iter(dataloader)
	total_loss = 0.0
	total_items = 0
	steps = 0

	while max_steps is None or steps < max_steps:
		try:
			batch = next(iterator)
		except StopIteration:
			if steps == 0:
				raise ValueError("dataloader is empty")
			if max_steps is None:
				break
			iterator = iter(dataloader)
			try:
				batch = next(iterator)
			except StopIteration as exc:
				raise ValueError("dataloader is empty") from exc

		optimizer.zero_grad(set_to_none=True)
		loss, item_count = _batch_loss(model, batch, loss_function, resolved_device)
		loss.backward()
		optimizer.step()
		total_loss += loss.detach().item() * item_count
		total_items += item_count
		steps += 1

	return total_loss / total_items


def validate(
	model: nn.Module,
	dataloader: Iterable[Any],
	criterion: Optional[nn.Module] = None,
	device: DeviceArg = None,
) -> float:
	"""Calculate sample-weighted validation loss without changing model mode."""
	resolved_device = resolve_device(device)
	model.to(resolved_device)
	was_training = model.training
	model.eval()
	loss_function = criterion or nn.MSELoss()
	total_loss = 0.0
	total_items = 0

	try:
		with torch.no_grad():
			for batch in dataloader:
				loss, item_count = _batch_loss(
					model, batch, loss_function, resolved_device
				)
				total_loss += loss.item() * item_count
				total_items += item_count
	finally:
		model.train(was_training)

	if total_items == 0:
		raise ValueError("validation dataloader is empty")
	return total_loss / total_items


def evaluate(
	model: nn.Module,
	dataloader: Iterable[Any],
	criterion: Optional[nn.Module] = None,
	device: DeviceArg = None,
) -> float:
	"""Evaluate a finalized model on the independent test loader."""
	return validate(model, dataloader, criterion=criterion, device=device)


def save_checkpoint(
	path: Union[str, Path],
	model: nn.Module,
	optimizer: Optional[Optimizer] = None,
	*,
	epoch: Optional[int] = None,
	best_validation_loss: Optional[float] = None,
	seed: Optional[int] = None,
) -> None:
	"""Save model and optional optimizer/training metadata to a checkpoint."""
	checkpoint_path = Path(path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	payload: Dict[str, Any] = {
		"model_state_dict": model.state_dict(),
		"epoch": epoch,
		"best_validation_loss": best_validation_loss,
		"seed": seed,
	}
	if optimizer is not None:
		payload["optimizer_state_dict"] = optimizer.state_dict()
	temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
	torch.save(payload, temporary_path)
	temporary_path.replace(checkpoint_path)


def load_checkpoint(
	path: Union[str, Path],
	model: nn.Module,
	optimizer: Optional[Optimizer] = None,
	device: DeviceArg = None,
) -> Dict[str, Any]:
	"""Restore model and optional optimizer state, mapping tensors to device."""
	resolved_device = resolve_device(device)
	checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(resolved_device)
	if optimizer is not None and "optimizer_state_dict" in checkpoint:
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	return checkpoint


def fit(
	model: nn.Module,
	train_loader: Iterable[Any],
	validation_loader: Iterable[Any],
	*,
	epochs: int = DEFAULT_EPOCHS,
	steps_per_epoch: Optional[int] = DEFAULT_STEPS_PER_EPOCH,
	learning_rate: float = DEFAULT_LEARNING_RATE,
	patience: Optional[int] = DEFAULT_PATIENCE,
	device: DeviceArg = None,
	seed: Optional[int] = None,
	optimizer: Optional[Optimizer] = None,
	criterion: Optional[nn.Module] = None,
	checkpoint_path: Optional[Union[str, Path]] = None,
	on_epoch_end: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
	"""Train and validate, saving the lowest-validation-loss checkpoint."""
	if epochs < 1:
		raise ValueError("epochs must be positive")
	if steps_per_epoch is not None and steps_per_epoch < 1:
		raise ValueError("steps_per_epoch must be positive")
	if learning_rate <= 0:
		raise ValueError("learning_rate must be positive")
	if patience is not None and patience < 1:
		raise ValueError("patience must be positive or None")
	if seed is not None:
		set_random_seed(seed)

	resolved_device = resolve_device(device)
	model.to(resolved_device)
	optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate)
	loss_function = criterion or nn.MSELoss()
	best_validation_loss = float("inf")
	best_epoch = 0
	stale_epochs = 0
	history = []

	for epoch in range(1, epochs + 1):
		training_loss = train_one_epoch(
			model,
			train_loader,
			optimizer,
			loss_function,
			resolved_device,
			max_steps=steps_per_epoch,
		)
		validation_loss = validate(
			model, validation_loader, loss_function, resolved_device
		)
		record = {
			"epoch": epoch,
			"training_loss": training_loss,
			"validation_loss": validation_loss,
		}
		history.append(record)
		if on_epoch_end is not None:
			on_epoch_end(record)

		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			best_epoch = epoch
			stale_epochs = 0
			if checkpoint_path is not None:
				save_checkpoint(
					checkpoint_path,
					model,
					optimizer,
					epoch=epoch,
					best_validation_loss=best_validation_loss,
					seed=seed,
				)
		else:
			stale_epochs += 1
			if patience is not None and stale_epochs >= patience:
				break

	return {
		"history": history,
		"best_epoch": best_epoch,
		"best_validation_loss": best_validation_loss,
		"device": str(resolved_device),
	}


def train_davis(
	*,
	processed_dir: str = "data/processed/davis",
	raw_dir: str = "data/raw/davis",
	checkpoint_path: Union[str, Path] = "checkpoints/mgraphdta_best.pt",
	batch_size: int = DEFAULT_BATCH_SIZE,
	epochs: int = DEFAULT_EPOCHS,
	steps_per_epoch: Optional[int] = DEFAULT_STEPS_PER_EPOCH,
	learning_rate: float = DEFAULT_LEARNING_RATE,
	patience: Optional[int] = DEFAULT_PATIENCE,
	validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
	seed: Optional[int] = DEFAULT_SEED,
	device: DeviceArg = None,
	num_workers: int = 0,
	on_epoch_end: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
	"""Train on a seeded training-fold split, then evaluate the independent test fold."""
	if seed is not None:
		set_random_seed(seed)
	train_dataset = DavisDataset(
		split="train", raw_dir=raw_dir, processed_dir=processed_dir
	)
	training_subset, validation_subset = split_train_validation(
		train_dataset, validation_fraction=validation_fraction, seed=seed
	)
	train_loader = make_dataloader(
		training_subset, batch_size, shuffle=True, num_workers=num_workers
	)
	validation_loader = make_dataloader(
		validation_subset, batch_size, shuffle=False, num_workers=num_workers
	)
	model = MGraphDTA()
	training_result = fit(
		model,
		train_loader,
		validation_loader,
		epochs=epochs,
		steps_per_epoch=steps_per_epoch,
		learning_rate=learning_rate,
		patience=patience,
		device=device,
		seed=seed,
		checkpoint_path=checkpoint_path,
		on_epoch_end=on_epoch_end,
	)
	load_checkpoint(checkpoint_path, model, device=device)

	# Do not construct or iterate over the independent test dataset before this point.
	test_dataset = DavisDataset(
		split="test", raw_dir=raw_dir, processed_dir=processed_dir
	)
	test_loader = make_dataloader(
		test_dataset, batch_size, shuffle=False, num_workers=num_workers
	)
	test_mse = evaluate(model, test_loader, device=device)
	return {**training_result, "test_mse": test_mse}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--processed-dir", default="data/processed/davis")
	parser.add_argument("--raw-dir", default="data/raw/davis")
	parser.add_argument("--checkpoint", default="checkpoints/mgraphdta_best.pt")
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
	parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
	parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
	parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
	parser.add_argument(
		"--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
	)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
	parser.add_argument("--num-workers", type=int, default=0)
	args = parser.parse_args()

	def report_epoch(record: Dict[str, float]) -> None:
		print(
			"epoch={epoch} training_mse={training_loss:.6f} "
			"validation_mse={validation_loss:.6f}".format(**record)
		)

	result = train_davis(
		processed_dir=args.processed_dir,
		raw_dir=args.raw_dir,
		checkpoint_path=args.checkpoint,
		batch_size=args.batch_size,
		epochs=args.epochs,
		steps_per_epoch=args.steps_per_epoch,
		learning_rate=args.learning_rate,
		patience=args.patience,
		validation_fraction=args.validation_fraction,
		seed=args.seed,
		device=args.device,
		num_workers=args.num_workers,
		on_epoch_end=report_epoch,
	)
	print(
		f"Best validation MSE {result['best_validation_loss']:.6f} "
		f"at epoch {result['best_epoch']} on {result['device']}"
	)
	print(f"Final independent test MSE {result['test_mse']:.6f}")


if __name__ == "__main__":
	main()
