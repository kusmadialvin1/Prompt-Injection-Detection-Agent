
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from PIGuard import PIGuard
from datasets import load_dataset
from util import set_seed, get_logger, compute_accuracy
from params import parse_args
from eval import evaluate


def atomic_save(obj, path):
    temp_path = path + ".tmp"
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    cpu_rng_state = state["torch"].detach().cpu().to(torch.uint8)
    torch.set_rng_state(cpu_rng_state)

    if torch.cuda.is_available() and "cuda" in state:
        cuda_rng_states = [
            rng_state.detach().cpu().to(torch.uint8)
            for rng_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)


def train():
    global logger
    args = parse_args()

    set_seed(args)
    logger = get_logger(
        os.path.join(args.logs, "log_{}.txt".format(args.name))
    )

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = args.epochs
    save_path = args.checkpoint_path
    os.makedirs(save_path, exist_ok=True)

    latest_path = os.path.join(save_path, "latest_checkpoint.pth")
    best_model_path = os.path.join(save_path, "best_model.pth")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def preprocess_function(examples):
        encoding = tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


    train_dataset = load_dataset(
    "json",
    data_files=args.train_set,
    split="train",
).select_columns(["prompt", "label", "source"])

    valid_dataset = load_dataset(
    "json",
    data_files=args.valid_set,
    split="train",
).select_columns(["prompt", "label", "source"])

    encoded_train = train_dataset.map(
    preprocess_function,
    batched=True,
)

    encoded_valid = valid_dataset.map(
    preprocess_function,
    batched=True,
)

    encoded_train = encoded_train.map(
    lambda examples: {"labels": examples["label"]},
    batched=True,
)

    encoded_valid = encoded_valid.map(
    lambda examples: {"labels": examples["label"]},
    batched=True,
)

    encoded_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

    encoded_valid.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

    encoded_dataset = {
    "train": encoded_train,
    "valid": encoded_valid,
}

    validation_loader = DataLoader(
        encoded_dataset["valid"],
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    model = PIGuard(
        "microsoft/deberta-v3-base",
        num_labels=2,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    steps_per_epoch = (
        len(encoded_dataset["train"]) + args.batch_size - 1
    ) // args.batch_size

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=epochs * steps_per_epoch,
    )

    start_epoch = 0
    start_step = 0
    best_accuracy = 0.0
    global_step = 0
    rng_state_to_restore = None

    if args.resume:
        print(f"Loading resume file: {args.resume}")
        saved = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        if isinstance(saved, dict) and "model_state_dict" in saved:
            model.load_state_dict(saved["model_state_dict"])
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            scheduler.load_state_dict(saved["scheduler_state_dict"])

            start_epoch = saved["epoch"]
            start_step = saved["next_step"]
            best_accuracy = saved["best_accuracy"]
            global_step = saved["global_step"]
            rng_state_to_restore = saved.get("rng_state")

            print(
                f"Resuming at epoch {start_epoch + 1}, "
                f"next batch {start_step}, "
                f"global step {global_step}"
            )
        else:
            # Backward compatibility with older weights-only checkpoints.
            model.load_state_dict(saved, strict=False)
            print("Loaded model weights only; starting a new training schedule.")

    def save_full_checkpoint(epoch, next_step):
        checkpoint = {
            "format_version": 1,
            "epoch": epoch,
            "next_step": next_step,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_accuracy": best_accuracy,
            "rng_state": capture_rng_state(),
            "train_set": os.path.abspath(args.train_set),
            "valid_set": os.path.abspath(args.valid_set),
            "epochs": epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": getattr(args, "seed", None),
        }
        atomic_save(checkpoint, latest_path)
        print(
            f"Full checkpoint saved: {latest_path} "
            f"(epoch {epoch + 1}, next batch {next_step})"
        )

    for epoch in range(start_epoch, epochs):
        # A deterministic shuffle per epoch allows the same batch order
        # to be reconstructed when resuming within an epoch.
        generator = torch.Generator()
        generator.manual_seed(
            int(getattr(args, "seed", 42)) + epoch
        )

        train_loader = DataLoader(
            encoded_dataset["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )

        batch_start = start_step if epoch == start_epoch else 0

        if rng_state_to_restore is not None:
            restore_rng_state(rng_state_to_restore)
            rng_state_to_restore = None

        model.train()

        for step, batch in enumerate(train_loader):
            if step < batch_start:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, loss = model(
                input_ids,
                attention_mask,
                labels,
                mode="train",
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if step % args.display == 0:
                logger.info(
                    f"Epoch: {epoch + 1}/{epochs}, "
                    f"Step: {step}/{len(train_loader)}, "
                    f"Global step: {global_step}, "
                    f"Loss: {loss.item():.4f}"
                )

            should_validate = (
                (step > 0 and step % args.save_step == 0)
                or step == len(train_loader) - 1
            )

            if not should_validate:
                continue

            model.eval()
            loss_list = []
            logits_list = []
            labels_list = []

            with torch.no_grad():
                for eval_step, eval_batch in enumerate(validation_loader):
                    eval_ids = eval_batch["input_ids"].to(device)
                    eval_mask = eval_batch["attention_mask"].to(device)
                    eval_labels = eval_batch["labels"].to(device)

                    eval_logits, eval_loss = model(
                        eval_ids,
                        eval_mask,
                        eval_labels,
                    )

                    loss_list.append(eval_loss.item())
                    logits_list.append(eval_logits.cpu())
                    labels_list.append(eval_labels.cpu())

            combined_logits = torch.cat(logits_list, dim=0)
            combined_labels = torch.cat(labels_list, dim=0)
            predictions = combined_logits.argmax(1)

            benign_acc, injection_acc, total_acc = compute_accuracy(
                predictions,
                combined_labels,
            )

            mean_loss = float(np.mean(loss_list))
            print(
                f"Validation: epoch={epoch + 1}, step={step}, "
                f"total_accuracy={total_acc:.4f}, "
                f"benign_accuracy={benign_acc:.4f}, "
                f"injection_accuracy={injection_acc:.4f}, "
                f"loss={mean_loss:.4f}"
            )

            if total_acc > args.save_thres:
                if total_acc > best_accuracy:
                    best_accuracy = total_acc
                    atomic_save(model.state_dict(), best_model_path)
                    print(f"Saved best model: {best_model_path}")

            model.train()

            # Save even when validation accuracy is below the threshold.
            # This is the full, resumable checkpoint.
            save_full_checkpoint(epoch, step + 1)

            # Save even when validation accuracy is below the threshold.
            # This is the full, resumable checkpoint.
            save_full_checkpoint(epoch, step + 1)

        start_step = 0

    logger.info("Training finished. Evaluating the best available model.")

    if os.path.exists(best_model_path):
        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device,
                weights_only=True,
            )
        )
        logger.info(f"Loaded best model from {best_model_path}")
    else:
        logger.warning(
            "No best_model.pth was created; evaluating final model weights."
        )

    evaluate(model, args.dataset_root)


if __name__ == "__main__":
    train()
