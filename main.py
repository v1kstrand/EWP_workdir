# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import os
import runpy
import signal
import subprocess
import sys
import time
from torchvision import transforms
from tqdm.auto import tqdm

import torch
from torch import nn

#import augmentations as aug

import src.dataset as ds
import src.models as m
from src.env_utils import load_env_file
from src.tracker import NullTracker, build_tracker
from src.rotations import relative_quat_from_euler_pairs
from copy import deepcopy

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

# Model
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--equi", type=int, default=256)
parser.add_argument("--experience", type=str, choices=["SIENoVar","SIE","EWPInterpol","SIEOnlyEqui","VICReg","SimCLR","VICRegPartInv",
                                                        "SimCLROnlyEqui","SIERotColor","SimCLRAugSelf","SimCLRAugSelfRotColor",
                                                        "SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod","VICRegEquiMod"],
                                                        default="SIE")
parser.add_argument("--hypernetwork", type=str, choices=["linear","deep"],default="linear")
# Only for when using an expander
parser.add_argument("--mlp", default="2048-2048-2048")
#Predictor architecture, in format "intermediate1-intermediate2-..."
parser.add_argument("--predictor", default="")
parser.add_argument("--pred-size-in",type=int, default=10)
parser.add_argument("--predictor-relu",  action="store_true")

# Predictor
parser.add_argument("--predictor-type",type=str,choices=["hypernetwork","mlp"],default="hypernetwork")
parser.add_argument("--bias-pred", action="store_true")
parser.add_argument("--bias-hypernet", action="store_true")
parser.add_argument("--pose-mlp-hidden-dim", type=int, default=0)
parser.add_argument("--pose-mlp-layers", type=int, default=2)
parser.add_argument("--pose-ema", action="store_true")
parser.add_argument("--pose-ema-momentum", type=float, default=0.95)
parser.add_argument("--simclr-temp",type=float,default=0.1)
parser.add_argument("--ec-weight",type=float,default=1)
parser.add_argument("--tf-num-layers",type=int,default=1)



# Optim
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--base-lr", type=float, default=1e-3)
parser.add_argument("--scale-lr-by-batch", action="store_true")
parser.add_argument("--lr-reference-batch-size", type=int, default=1024)
parser.add_argument("--wd", type=float, default=1e-6)

parser.add_argument("--warmup-start",type=int, default=0)
parser.add_argument("--warmup-length",type=int, default=0)


# Data
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)
parser.add_argument("--images-file", type=Path, default="./data/train_images.npy", required=True)
parser.add_argument("--labels-file", type=Path, default="./data/val_images.npy", required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--latent-cache-file", type=Path, default=None)

# Checkpoints
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path,default="EXP_DIR/logs/")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--eval-freq", type=int, default=10)
parser.add_argument("--log-freq-time", type=int, default=30)
parser.add_argument("--comet-project-name", type=str, default="sie_3die")
parser.add_argument("--comet-workspace", type=str, default=None)
parser.add_argument("--comet-experiment-key", type=str, default=None)
parser.add_argument("--comet-disabled", action="store_true")
parser.add_argument("--env-file", type=Path, default=Path("/notebooks/env.txt"))
parser.add_argument("--env-override", action="store_true")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--compile-backend", type=str, default="inductor")
parser.add_argument("--compile-mode", type=str, default="max-autotune")
parser.add_argument("--compile-fullgraph", action="store_true")
parser.add_argument("--compile-dynamic", action="store_true")

# Loss
parser.add_argument("--sim-coeff", type=float, default=10.0)
parser.add_argument("--equi-factor", type=float, default=0.45)
parser.add_argument("--std-coeff", type=float, default=10.0)
parser.add_argument("--cov-coeff", type=float, default=1.0)
# Kept as no-op args for compatibility with existing EWPInterpol launchers.
parser.add_argument("--interp-weight", type=float, default=0.0)
parser.add_argument("--position-weight", type=float, default=0.0)

# Running
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--prefetch-factor", type=int, default=2)
parser.add_argument("--no-pin-memory", action="store_true")
parser.add_argument("--no-amp", action="store_true")
parser.add_argument("--port", type=int, default=52472)
parser.add_argument("--grad-clip-norm", type=float, default=0.0)


def apply_repo_runtime_init() -> None:
    repo_root = Path(__file__).resolve().parent
    init_path = repo_root / "init.py"
    if not init_path.exists():
        return
    runpy.run_path(str(init_path), run_name="__main__")
    print(f"Runtime init loaded: {init_path}")


def compile_selected_forwards(model, args):
    compile_targets = ["backbone"]
    for name in ["projector_inv", "projector_equi", "predictor", "pose_head"]:
        if hasattr(model, name):
            compile_targets.append(name)

    for name in compile_targets:
        module = getattr(model, name, None)
        if module is None or not hasattr(module, "forward"):
            continue
        module.forward = torch.compile(
            module.forward,
            backend=args.compile_backend,
            mode=args.compile_mode,
            dynamic=args.compile_dynamic,
            fullgraph=args.compile_fullgraph,
        )
    return compile_targets




def _adapt_state_dict_for_model(model: nn.Module, state_dict: dict) -> dict:
    model_keys = set(model.state_dict().keys())
    adapted = dict(state_dict)
    ckpt_keys = set(adapted.keys())
    if ckpt_keys == model_keys:
        return adapted

    if all(key.startswith("module.") for key in ckpt_keys):
        stripped = {key[len("module."):]: value for key, value in adapted.items()}
        if set(stripped.keys()) == model_keys:
            return stripped

    if all(("module." + key) in ckpt_keys for key in model_keys):
        stripped = {key[len("module."):]: value for key, value in adapted.items() if key.startswith("module.")}
        if set(stripped.keys()) == model_keys:
            return stripped

    # Older EWP checkpoints predate the EMA pose head. Seed it from the saved student pose head.
    for model_key in list(model_keys):
        if not model_key.startswith("pose_head_ema.") or model_key in adapted:
            continue
        student_key = "pose_head." + model_key[len("pose_head_ema."):]
        if student_key in adapted:
            adapted[model_key] = adapted[student_key]

    filtered = {key: value for key, value in adapted.items() if key in model_keys}
    if set(filtered.keys()) == model_keys:
        return filtered

    return adapted

def get_effective_lr(args) -> float:
    if not args.scale_lr_by_batch:
        return args.base_lr
    return args.base_lr * (args.batch_size / float(args.lr_reference_batch_size))


def main():
    args = parser.parse_args()
    args.effective_lr = get_effective_lr(args)
    loaded, skipped = load_env_file(args.env_file, override=args.env_override)
    if args.env_file.exists():
        print(f"Env loaded from {args.env_file}: loaded={loaded}, skipped={skipped}")
    else:
        print(f"Env file not found, continuing without it: {args.env_file}")
    apply_repo_runtime_init()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv("SLURM_NODEID")) * args.ngpus_per_node
        args.world_size = int(os.getenv("SLURM_NNODES")) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:{args.port}"
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f"tcp://localhost:{args.port}"
        args.world_size = args.ngpus_per_node
    args.use_ddp = args.world_size > 1
    if args.ngpus_per_node == 1:
        main_worker(0, args)
    else:
        torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)



def main_worker(gpu, args):
    if args.use_ddp:
        args.rank += gpu
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.root_log_dir.mkdir(parents=True, exist_ok=True)

    # Config dump
    if args.rank == 0:
        dict_args = deepcopy(vars(args))
        for key,value in dict_args.items():
            if isinstance(value,Path):
                dict_args[key] = str(value)
        with open(args.exp_dir / "params.json", 'w') as f:
            json.dump(dict_args, f)

    # Comet setup
    if args.rank == 0:
        tracker = build_tracker(
            backend="comet_required",
            out_dir=args.exp_dir,
            project_name=args.comet_project_name,
            api_key_env="COMET_API_KEY",
            workspace=args.comet_workspace,
            experiment_key=args.comet_experiment_key,
            disabled=args.comet_disabled,
        )
        tracker.log_parameters(dict_args)
        print(" ".join(sys.argv))
        print(
            f"LR setup: base_lr={args.base_lr} "
            f"effective_lr={args.effective_lr} "
            f"batch_size={args.batch_size} "
            f"scale_lr_by_batch={args.scale_lr_by_batch} "
            f"lr_reference_batch_size={args.lr_reference_batch_size}"
        )
    else:
        tracker = NullTracker()

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )
    if args.experience in ["SIERotColor","SimCLRAugSelfRotColor","SimCLROnlyEquiRotColor","SimCLREquiModRotColor"]:
        dataset = ds.Dataset3DIEBenchRotColor(args.dataset_root, args.images_file, args.labels_file, transform=transforms.Compose([transforms.Resize((args.resolution, args.resolution)), transforms.ToTensor(), normalize]), latent_cache_file=args.latent_cache_file)
    else:
        dataset = ds.Dataset3DIEBench(args.dataset_root, args.images_file, args.labels_file, transform=transforms.Compose([transforms.Resize((args.resolution, args.resolution)), transforms.ToTensor(), normalize]), latent_cache_file=args.latent_cache_file)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if args.use_ddp else None
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    print("per_device_batch_size",per_device_batch_size)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        drop_last=args.compile,
        persistent_workers=args.num_workers > 0,
    )
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = True
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = torch.utils.data.DataLoader(**loader_kwargs)

    model = m.__dict__[args.experience](args).cuda(gpu)
    if args.experience in ["SimCLR","SimCLROnlyEqui","SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod"]:
        model.gpu = gpu
    if args.use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=False)

    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.effective_lr,
        weight_decay=args.wd
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model_state = _adapt_state_dict_for_model(model, ckpt["model"])
        model.load_state_dict(model_state)
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    train_model = model
    base_model = model.module if args.use_ddp else model
    if args.compile:
        if args.rank == 0:
            print("TORCHINDUCTOR_CACHE_DIR resolved to:", os.environ.get("TORCHINDUCTOR_CACHE_DIR"))
            print("TRITON_CACHE_DIR resolved to:", os.environ.get("TRITON_CACHE_DIR"))
            print(
                "Compiling selected forwards with torch.compile("
                f"backend='{args.compile_backend}', "
                f"mode='{args.compile_mode}', "
                f"dynamic={args.compile_dynamic}, "
                f"fullgraph={args.compile_fullgraph})"
            )
        compiled_names = compile_selected_forwards(base_model, args)
        if args.rank == 0:
            print("Compiled modules:", ",".join(compiled_names))

    start_time = last_logging = time.time()
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loader = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=args.rank != 0,
            dynamic_ncols=True,
            leave=True,
        )
        for local_step, batch in enumerate(epoch_loader):
            if len(batch) == 5:
                x, y, angles_1, angles_2, labels = batch
                other_params = None
            elif len(batch) == 6:
                x, y, angles_1, angles_2, other_params, labels = batch
            else:
                raise ValueError(f"Unexpected batch structure with {len(batch)} elements")

            step = epoch * len(loader) + local_step
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            angles_1 = angles_1.cuda(gpu, non_blocking=True)
            angles_2 = angles_2.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)
            z = relative_quat_from_euler_pairs(angles_1, angles_2)
            if other_params is not None:
                other_params = other_params.cuda(gpu, non_blocking=True)
                z = torch.cat([z, other_params], dim=-1)

            optimizer.zero_grad()

            # MAIN TRAINING PART
            with torch.amp.autocast("cuda", enabled=not args.no_amp):
                loss, classif_loss, stats, stats_eval = train_model.forward(x, y, z,labels)
                total_loss = loss + classif_loss

            scaler.scale(total_loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                grad_norm = None
            scaler.step(optimizer)
            scaler.update()
            if hasattr(base_model, "update_pose_head_ema"):
                base_model.update_pose_head_ema()
            if args.rank == 0:
                epoch_loader.set_postfix(
                    loss=f"{stats['loss'].item():.4f}",
                    total=f"{total_loss.item():.4f}",
                )

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                metrics = {
                    'General/epoch': float(epoch),
                    'General/time_elapsed': float(int(current_time - start_time)),
                    'General/lr': float(args.effective_lr),
                    'General/Current GPU memory': float(torch.cuda.memory_allocated(torch.cuda.device('cuda:0'))/1e9),
                    'General/Max GPU memory': float(torch.cuda.max_memory_allocated(torch.cuda.device('cuda:0'))/1e9),
                    'Loss/Total loss': float(stats["loss"].item()),
                }
                if args.experience in ["SimCLRAugSelf","SimCLRAugSelfFull","SimCLRAugSelfRotColor"]:
                    metrics['Loss/Invariance loss'] = float(stats["repr_loss_inv"].item())
                if not args.experience in ["SimCLR","SimCLRAugSelf","SimCLRAugSelfFull","SimCLRAugSelfRotColor","SimCLROnlyEqui","SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod"]:
                    metrics['Loss/Invariance loss'] = float(stats["repr_loss_inv"].item())
                    metrics['Loss/Std loss'] = float(stats["std_loss"].item())
                    metrics['Loss/Covariance loss'] = float(stats["cov_loss"].item())
                if not args.experience in ["VICReg","VICRegNoCov","VICRegCos","VICRegL1","VICRegL1repr","FullEqui","VICRegPartInv","SimCLR","VICRegPartInv2Exps","SimCLROnlyEqui","SIERotColor","SimCLROnlyEquiRotColor"] :
                    metrics['Loss/Equivariance loss'] = float(stats["repr_loss_equi"].item())
                if args.experience in ["SIEOnlyEqui","SIE","SIEAll","SIERotColor"]:
                    metrics['Loss/Pred Std loss'] = float(stats["pred_std_loss"].item())
                if "endpoint_loss" in stats:
                    metrics['Loss/Endpoint loss'] = float(stats["endpoint_loss"].item())
                if grad_norm is not None:
                    metrics['Stats/Grad norm'] = float(grad_norm.item())
                metrics['Stats/Corr. representations view1'] = float(stats["coremb_view1"].item())
                metrics['Stats/Corr. representations view2'] = float(stats["coremb_view2"].item())
                metrics['Stats/Std representations view1'] = float(stats["stdemb_view1"].item())
                metrics['Stats/Std representations view2'] = float(stats["stdemb_view2"].item())
                metrics['Stats/Corr. embeddings view1'] = float(stats["corhead_view1"].item())
                metrics['Stats/Corr. embeddings view2'] = float(stats["corhead_view2"].item())
                metrics['Stats/Std embeddings view1'] = float(stats["stdhead_view1"].item())
                metrics['Stats/Std embeddings view2'] = float(stats["stdhead_view2"].item())
                if "stdemb_pred" in stats.keys():
                    metrics['Stats/Corr. predictor output'] = float(stats["coremb_pred"].item())
                    metrics['Stats/Std predictor output'] = float(stats["stdemb_pred"].item())

                for key,value in stats_eval.items():
                    if "representations" in key:
                        metrics[f'Online eval reprs/{key}'] = float(value)
                    elif "embeddings" in key:
                        metrics[f'Online eval embs/{key}'] = float(value)
                for key,value in stats.items():
                    if "Latent/" in key:
                        metrics[key] = float(value)
                tracker.log_metrics(metrics, step=step)
                last_logging = current_time
        if args.rank == 0:
            epoch_loader.close()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        tracker.finish()
        backbone = model.module.backbone if args.use_ddp else model.backbone
        torch.save(backbone.state_dict(), args.exp_dir / "final_weights.pth")
    if args.use_ddp:
        torch.distributed.destroy_process_group()


def exclude_bias_and_norm(p):
    return p.ndim == 1

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    main()
