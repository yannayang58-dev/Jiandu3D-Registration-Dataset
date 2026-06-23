import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class JianduRegistrationDataset(Dataset):
    def __init__(self, data_root, split="train", n_points=1024):
        self.data_root = Path(data_root)
        self.split = split
        self.n_points = n_points

        manifest_path = self.data_root / f"{split}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到 manifest: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        print(f"[Dataset] {split}: {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def _resolve_npz_path(self, rec):
        if "npz" in rec and os.path.exists(rec["npz"]):
            return rec["npz"]

        if "relative_npz" in rec:
            p = self.data_root.parent / rec["relative_npz"]
            if p.exists():
                return str(p)

        raise FileNotFoundError(f"找不到 npz: {rec}")

    def __getitem__(self, idx):
        rec = self.records[idx]
        npz_path = self._resolve_npz_path(rec)

        data = np.load(npz_path, allow_pickle=True)

        source = data["source"].astype(np.float32)
        target = data["target"].astype(np.float32)
        R_gt = data["R_gt"].astype(np.float32)
        t_gt = data["t_gt"].astype(np.float32)
        T_gt = data["transform_gt"].astype(np.float32)

        if not np.isfinite(source).all():
            raise ValueError(f"source nan/inf: {npz_path}")
        if not np.isfinite(target).all():
            raise ValueError(f"target nan/inf: {npz_path}")
        if not np.isfinite(R_gt).all():
            raise ValueError(f"R_gt nan/inf: {npz_path}")
        if not np.isfinite(t_gt).all():
            raise ValueError(f"t_gt nan/inf: {npz_path}")

        source = self._resample(source, self.n_points)
        target = self._resample(target, self.n_points)

        return {
            "source": torch.from_numpy(source),
            "target": torch.from_numpy(target),
            "R_gt": torch.from_numpy(R_gt),
            "t_gt": torch.from_numpy(t_gt),
            "T_gt": torch.from_numpy(T_gt),
            "category": rec.get("category", "unknown"),
            "sample_id": rec.get("sample_id", ""),
            "pair_type": rec.get("pair_type", "B_to_A"),
        }

    @staticmethod
    def _resample(points, n_points):
        n = len(points)
        if n == n_points:
            return points

        if n > n_points:
            idx = np.random.choice(n, n_points, replace=False)
        else:
            idx = np.random.choice(n, n_points, replace=True)

        return points[idx].astype(np.float32)


# ============================================================
# Math utils
# ============================================================

def square_distance(src, dst):
    return torch.cdist(src, dst, p=2) ** 2


def transform_points(points, R, t):
    return torch.matmul(points, R.transpose(1, 2)) + t[:, None, :]


@torch.no_grad()
def weighted_procrustes_eval(source, target_corr, weights=None, eps=1e-6):
    """
    只用于 evaluation，不反传。
    """
    B, N, _ = source.shape
    device = source.device
    dtype = source.dtype

    if weights is None:
        weights = torch.ones(B, N, device=device, dtype=dtype)

    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights.clamp(min=eps)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=eps)

    source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    target_corr = torch.nan_to_num(target_corr, nan=0.0, posinf=0.0, neginf=0.0)

    src_centroid = torch.sum(source * weights[:, :, None], dim=1)
    tgt_centroid = torch.sum(target_corr * weights[:, :, None], dim=1)

    src_centered = source - src_centroid[:, None, :]
    tgt_centered = target_corr - tgt_centroid[:, None, :]

    H = torch.matmul(
        (src_centered * weights[:, :, None]).transpose(1, 2),
        tgt_centered
    )

    H = H + eps * torch.eye(3, device=device, dtype=dtype)[None, :, :]

    try:
        U, S, Vh = torch.linalg.svd(H)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)

        det = torch.det(torch.matmul(V, Ut))
        diag = torch.ones(B, 3, device=device, dtype=dtype)
        diag[:, 2] = torch.where(det < 0, -1.0, 1.0)

        R = torch.matmul(torch.matmul(V, torch.diag_embed(diag)), Ut)
        t = tgt_centroid - torch.matmul(src_centroid[:, None, :], R.transpose(1, 2)).squeeze(1)

    except Exception:
        R = torch.eye(3, device=device, dtype=dtype)[None].repeat(B, 1, 1)
        t = torch.zeros(B, 3, device=device, dtype=dtype)

    R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    return R, t


def rotation_error_deg(R_pred, R_gt):
    R_diff = torch.matmul(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    err = torch.acos(cos) * 180.0 / math.pi
    return err


def translation_error(t_pred, t_gt):
    return torch.norm(t_pred - t_gt, dim=1)


def rmse_nn(source_trans, target):
    dist2 = square_distance(source_trans, target)
    dist2 = torch.nan_to_num(dist2, nan=1e4, posinf=1e4, neginf=1e4)
    dist2 = torch.clamp(dist2, 0.0, 1e4)
    min_dist2 = dist2.min(dim=2)[0]
    return torch.sqrt(min_dist2.mean(dim=1) + 1e-8)


# ============================================================
# Encoder / Model
# ============================================================

class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=3, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, feat_dim, 1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        feat = self.net(x)
        return feat.transpose(1, 2)


def make_rie_input(points):
    center = points.mean(dim=1, keepdim=True)
    r = torch.norm(points - center, dim=2, keepdim=True)
    return torch.cat([points, r], dim=2)


class MatchingRegModel(nn.Module):
    """
    稳定版统一 baseline：
    DCP-style / RPMNet-style / PRNet-style / RIENet-style
    训练时只做 correspondence loss，不对 SVD 反传。
    """
    def __init__(self, model_type="dcp", feat_dim=256):
        super().__init__()
        self.model_type = model_type.lower()
        self.feat_dim = feat_dim

        in_dim = 4 if self.model_type == "rienet" else 3
        self.encoder = PointNetEncoder(in_dim=in_dim, feat_dim=feat_dim)

        if self.model_type == "prnet":
            self.overlap_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)
            )
        else:
            self.overlap_head = None

        if self.model_type == "rpmnet":
            self.temperature = 0.08
        else:
            self.temperature = math.sqrt(feat_dim)

    def forward_scores(self, source, target):
        if self.model_type == "rienet":
            source_in = make_rie_input(source)
            target_in = make_rie_input(target)
        else:
            source_in = source
            target_in = target

        fs = self.encoder(source_in)
        ft = self.encoder(target_in)

        fs = F.normalize(fs, dim=-1)
        ft = F.normalize(ft, dim=-1)

        scores = torch.matmul(fs, ft.transpose(1, 2)) / self.temperature
        scores = torch.clamp(scores, -20.0, 20.0)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=20.0, neginf=-20.0)

        overlap_logits = None
        if self.overlap_head is not None:
            overlap_logits = self.overlap_head(fs).squeeze(-1)
            overlap_logits = torch.clamp(overlap_logits, -20.0, 20.0)
            overlap_logits = torch.nan_to_num(overlap_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        return scores, overlap_logits


# ============================================================
# Loss
# ============================================================

@torch.no_grad()
def make_gt_matches(source, target, R_gt, t_gt, match_radius=0.08, min_matches=64):
    """
    用 GT 把 source 变到 target 坐标系，然后找最近邻作为监督对应。
    """
    B, N, _ = source.shape
    source_gt = transform_points(source, R_gt, t_gt)

    dist2 = square_distance(source_gt, target)
    dist2 = torch.nan_to_num(dist2, nan=1e4, posinf=1e4, neginf=1e4)

    nn_dist2, nn_idx = dist2.min(dim=2)

    valid = nn_dist2 < (match_radius ** 2)

    # 如果某个 batch 有效点太少，用最近的 top-k 兜底
    for b in range(B):
        if valid[b].sum() < min_matches:
            k = min(min_matches, N)
            topk = torch.topk(nn_dist2[b], k=k, largest=False).indices
            valid[b] = False
            valid[b, topk] = True

    return nn_idx, valid


def matching_loss(scores, overlap_logits, source, target, R_gt, t_gt, args):
    """
    只训练对应关系，不训练 SVD。
    """
    B, N, M = scores.shape

    nn_idx, valid = make_gt_matches(
        source,
        target,
        R_gt,
        t_gt,
        match_radius=args.match_radius,
        min_matches=args.min_matches,
    )

    flat_scores = scores.reshape(B * N, M)
    flat_labels = nn_idx.reshape(B * N)
    flat_valid = valid.reshape(B * N)

    if flat_valid.sum() == 0:
        return None, {}

    loss_ce = F.cross_entropy(flat_scores[flat_valid], flat_labels[flat_valid])

    loss = loss_ce
    logs = {"loss_ce": float(loss_ce.detach().cpu())}

    if overlap_logits is not None:
        overlap_target = valid.float()
        loss_ov = F.binary_cross_entropy_with_logits(overlap_logits, overlap_target)
        loss = loss + args.w_overlap * loss_ov
        logs["loss_overlap"] = float(loss_ov.detach().cpu())

    return loss, logs


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def predict_pose(model, source, target):
    scores, overlap_logits = model.forward_scores(source, target)

    P = torch.softmax(scores, dim=-1)
    P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

    target_corr = torch.matmul(P, target)
    weights = P.max(dim=-1)[0]

    if overlap_logits is not None:
        weights = weights * torch.sigmoid(overlap_logits)

    R_pred, t_pred = weighted_procrustes_eval(source, target_corr, weights)
    return R_pred, t_pred


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()

    all_rre = []
    all_rte = []
    all_rmse = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        R_gt = batch["R_gt"].to(device)
        t_gt = batch["t_gt"].to(device)

        R_pred, t_pred = predict_pose(model, source, target)

        source_pred = transform_points(source, R_pred, t_pred)

        rre = rotation_error_deg(R_pred, R_gt)
        rte = translation_error(t_pred, t_gt)
        rmse = rmse_nn(source_pred, target)

        all_rre.append(rre.cpu())
        all_rte.append(rte.cpu())
        all_rmse.append(rmse.cpu())

    all_rre = torch.cat(all_rre)
    all_rte = torch.cat(all_rte)
    all_rmse = torch.cat(all_rmse)

    recall = ((all_rre < args.recall_r) & (all_rte < args.recall_t)).float().mean().item()

    return {
        "R_error": all_rre.mean().item(),
        "t_error": all_rte.mean().item(),
        "Rmse": all_rmse.mean().item(),
        "Recall": recall,
    }


# ============================================================
# Train
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    train_set = JianduRegistrationDataset(args.data_root, "train", args.n_points)
    val_set = JianduRegistrationDataset(args.data_root, "val", args.n_points)
    test_set = JianduRegistrationDataset(args.data_root, "test", args.n_points)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = MatchingRegModel(args.model, feat_dim=args.feat_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    best_recall = -1.0
    best_rmse = 1e9

    log_path = out_dir / "train_log.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_batch = 0
        skipped = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            R_gt = batch["R_gt"].to(device, non_blocking=True)
            t_gt = batch["t_gt"].to(device, non_blocking=True)

            if (not torch.isfinite(source).all()) or (not torch.isfinite(target).all()) \
               or (not torch.isfinite(R_gt).all()) or (not torch.isfinite(t_gt).all()):
                skipped += 1
                continue

            scores, overlap_logits = model.forward_scores(source, target)

            loss, logs = matching_loss(
                scores,
                overlap_logits,
                source,
                target,
                R_gt,
                t_gt,
                args,
            )

            if loss is None or not torch.isfinite(loss):
                skipped += 1
                continue

            optimizer.zero_grad()
            loss.backward()

            bad_grad = False
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break

            if bad_grad:
                optimizer.zero_grad()
                skipped += 1
                continue

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.item()
            total_batch += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "skip": skipped,
            })

        train_loss = total_loss / max(total_batch, 1)

        val_metrics = evaluate(model, val_loader, device, args)

        print(
            f"\n[Epoch {epoch}] "
            f"loss={train_loss:.6f} | "
            f"Val R={val_metrics['R_error']:.4f} "
            f"t={val_metrics['t_error']:.6f} "
            f"RMSE={val_metrics['Rmse']:.6f} "
            f"Recall={val_metrics['Recall']:.4f} "
            f"skip={skipped}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "skipped_batches": skipped,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        save_obj = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }

        torch.save(save_obj, out_dir / "last.pt")

        better = False
        if val_metrics["Recall"] > best_recall:
            better = True
        elif val_metrics["Recall"] == best_recall and val_metrics["Rmse"] < best_rmse:
            better = True

        if better:
            best_recall = val_metrics["Recall"]
            best_rmse = val_metrics["Rmse"]
            torch.save(save_obj, out_dir / "best.pt")
            print(f"[Save] best.pt  Recall={best_recall:.4f}, RMSE={best_rmse:.6f}")

    print("\n[Load best and test]")
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, test_loader, device, args)

    print("\n[Test Metrics]")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", type=str,
                   default="/home/sda/yyn/3D/Jiandu_ModelNetStyle/fracture")
    p.add_argument("--model", type=str, default="dcp",
                   choices=["dcp", "rpmnet", "prnet", "rienet"])
    p.add_argument("--out_dir", type=str, default="./checkpoints_jiandu_stable")

    p.add_argument("--n_points", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--feat_dim", type=int, default=256)

    p.add_argument("--match_radius", type=float, default=0.08)
    p.add_argument("--min_matches", type=int, default=64)
    p.add_argument("--w_overlap", type=float, default=0.1)

    p.add_argument("--recall_r", type=float, default=5.0)
    p.add_argument("--recall_t", type=float, default=0.05)

    p.add_argument("--grad_clip", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)