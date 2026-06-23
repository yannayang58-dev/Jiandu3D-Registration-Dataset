import os
import json
import math
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# 你只需要改这里
# ============================================================

# 你的整理后 canonical 数据集
INPUT_ROOT = "/home/sda/yyn/3D/synthetic/canonical_reference"

# 输出扰动版数据集
OUTPUT_ROOT = "/home/sda/yyn/3D/synthetic/registration_perturbed"

# 要处理的类别
CATEGORIES = [
    "transverse",
    "oblique",
    #"mixed",
    #"longitudinal",
]

# 是否覆盖旧的扰动输出
OVERWRITE = True

# 扰动范围
MAX_ROTATION_DEG = 25.0

# 平移单位是 mm
MAX_TX = 8.0
MAX_TY = 20.0
MAX_TZ = 4.0

# 随机种子
GLOBAL_SEED = 20260601

# 竖切是否扰动 middle 小碎片
PERTURB_LONGITUDINAL_MIDDLE = True


# ============================================================
# 下面不用改
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def force_remove_dir(path: Path):
    if not path.exists():
        return

    print(f"[Clean] 删除旧目录: {path}")

    try:
        shutil.rmtree(path)
        return
    except Exception as e:
        print(f"[Warn] shutil.rmtree 失败: {e}")
        os.system(f'rm -rf "{path}"')

    if path.exists():
        raise RuntimeError(f"目录删除失败: {path}")


def random_rotation_matrix(max_angle_deg: float, rng: np.random.Generator):
    """
    随机轴角旋转，角度范围 [-max_angle, max_angle]。
    """
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    angle = rng.uniform(-max_angle_deg, max_angle_deg) * math.pi / 180.0

    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    R = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)

    return R, angle


def make_transform_around_centroid(
    points: np.ndarray,
    max_angle_deg: float,
    max_tx: float,
    max_ty: float,
    max_tz: float,
    rng: np.random.Generator,
):
    """
    围绕 source 自身中心旋转，再平移。
    这样不会因为绕世界原点旋转导致碎片飞很远。
    """
    center = points.mean(axis=0)

    R, angle = random_rotation_matrix(max_angle_deg, rng)

    t_random = np.array([
        rng.uniform(-max_tx, max_tx),
        rng.uniform(-max_ty, max_ty),
        rng.uniform(-max_tz, max_tz),
    ], dtype=np.float64)

    # x' = R @ (x - c) + c + t
    #    = R @ x + (c + t - R @ c)
    t_global = center + t_random - R @ center

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t_global

    meta = {
        "rotation_angle_deg": float(angle * 180.0 / math.pi),
        "rotation_center": center.tolist(),
        "extra_translation_mm": t_random.tolist(),
        "global_translation_mm": t_global.tolist(),
    }

    return T, meta


def read_pcd_points(ply_path: Path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    return pcd, pts


def transform_ply_inplace(ply_path: Path, T: np.ndarray):
    if not ply_path.exists():
        return False

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        return False

    pcd.transform(T)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    return True


def write_gt(
    gt_dir: Path,
    source_name: str,
    target_name: str,
    source_files: list[str],
    T_noise: np.ndarray,
    T_gt: np.ndarray,
    noise_meta: dict,
):
    ensure_dir(gt_dir)

    gt_data = {
        "source": source_name,
        "target": target_name,
        "definition": "T_gt maps perturbed source point cloud to target canonical coordinate system.",
        "source_files_transformed": source_files,

        "T_noise_source_to_perturbed": T_noise.tolist(),
        "T_gt_perturbed_source_to_target": T_gt.tolist(),

        "R_gt": T_gt[:3, :3].tolist(),
        "t_gt": T_gt[:3, 3].tolist(),

        "noise_meta": noise_meta,
        "max_rotation_deg": MAX_ROTATION_DEG,
        "max_translation_xyz_mm": [MAX_TX, MAX_TY, MAX_TZ],
    }

    out_name = f"gt_{source_name}_to_{target_name}.json"
    with open(gt_dir / out_name, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    return out_name


def perturb_source_in_sample(
    sample_dir: Path,
    source_tag: str,
    rng: np.random.Generator,
):
    """
    source_tag:
        B -> fragment_B + fracture_B
        M -> fragment_M + fracture_M
    target 默认是 A。
    """
    frag_dir = sample_dir / "fragments"
    frac_dir = sample_dir / "fracture_faces"
    gt_dir = sample_dir / "GT"

    frag_path = frag_dir / f"fragment_{source_tag}.ply"
    frac_path = frac_dir / f"fracture_{source_tag}.ply"

    if not frag_path.exists():
        return None

    pcd, pts = read_pcd_points(frag_path)
    if len(pts) == 0:
        print(f"[Skip] 空点云: {frag_path}")
        return None

    T_noise, noise_meta = make_transform_around_centroid(
        points=pts,
        max_angle_deg=MAX_ROTATION_DEG,
        max_tx=MAX_TX,
        max_ty=MAX_TY,
        max_tz=MAX_TZ,
        rng=rng,
    )

    T_gt = np.linalg.inv(T_noise)

    transformed_files = []

    ok = transform_ply_inplace(frag_path, T_noise)
    if ok:
        transformed_files.append(f"fragments/fragment_{source_tag}.ply")

    if frac_path.exists():
        ok = transform_ply_inplace(frac_path, T_noise)
        if ok:
            transformed_files.append(f"fracture_faces/fracture_{source_tag}.ply")

    gt_file = write_gt(
        gt_dir=gt_dir,
        source_name=f"fragment_{source_tag}",
        target_name="fragment_A",
        source_files=transformed_files,
        T_noise=T_noise,
        T_gt=T_gt,
        noise_meta=noise_meta,
    )

    return {
        "source": f"fragment_{source_tag}",
        "target": "fragment_A",
        "gt_file": f"GT/{gt_file}",
        "transformed_files": transformed_files,
    }


def clear_old_gt(sample_dir: Path):
    """
    扰动数据集的 GT 重新生成，避免混入 identity GT。
    """
    gt_dir = sample_dir / "GT"

    if gt_dir.exists():
        shutil.rmtree(gt_dir)

    ensure_dir(gt_dir)


def copy_category(src_category_dir: Path, dst_category_dir: Path):
    if dst_category_dir.exists() and OVERWRITE:
        force_remove_dir(dst_category_dir)

    print(f"[Copy] {src_category_dir} -> {dst_category_dir}")
    shutil.copytree(src_category_dir, dst_category_dir)


def perturb_category(category: str, rng: np.random.Generator):
    src_category_dir = Path(INPUT_ROOT) / category
    dst_category_dir = Path(OUTPUT_ROOT) / category

    if not src_category_dir.exists():
        print(f"[Skip] 类别不存在: {src_category_dir}")
        return []

    copy_category(src_category_dir, dst_category_dir)

    sample_dirs = sorted([p for p in dst_category_dir.glob("sample_*") if p.is_dir()])

    print("\n" + "=" * 70)
    print(f"[Category] {category}")
    print(f"[Samples]  {len(sample_dirs)}")
    print("=" * 70)

    records = []

    for i, sample_dir in enumerate(sample_dirs):
        clear_old_gt(sample_dir)

        sample_record = {
            "category": category,
            "sample": sample_dir.name,
            "gt": [],
        }

        # B 是所有类别都扰动
        rec_B = perturb_source_in_sample(
            sample_dir=sample_dir,
            source_tag="B",
            rng=rng,
        )
        if rec_B is not None:
            sample_record["gt"].append(rec_B)

        # 竖切三块：M 也扰动
        if category == "longitudinal" and PERTURB_LONGITUDINAL_MIDDLE:
            rec_M = perturb_source_in_sample(
                sample_dir=sample_dir,
                source_tag="M",
                rng=rng,
            )
            if rec_M is not None:
                sample_record["gt"].append(rec_M)

        # 更新 sample_meta.json
        meta_path = sample_dir / "sample_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        else:
            meta = {}

        meta["perturbed"] = True
        meta["perturbation_gt"] = sample_record["gt"]

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        records.append(sample_record)

        if (i + 1) % 100 == 0:
            print(f"  已扰动 {i + 1}/{len(sample_dirs)}")

    # 类别级 index
    with open(dst_category_dir / "perturbation_index.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[Done] {category}: {len(records)} samples")
    return records


def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)

    if not input_root.exists():
        raise FileNotFoundError(f"INPUT_ROOT 不存在: {input_root}")

    ensure_dir(output_root)

    rng = np.random.default_rng(GLOBAL_SEED)

    summary = {}

    for category in CATEGORIES:
        records = perturb_category(category, rng)
        summary[category] = {
            "num_samples": len(records),
            "middle_perturbed": bool(category == "longitudinal" and PERTURB_LONGITUDINAL_MIDDLE),
        }

    with open(output_root / "perturbation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n全部扰动完成")
    print(f"输入: {INPUT_ROOT}")
    print(f"输出: {OUTPUT_ROOT}")
    print(f"summary: {output_root / 'perturbation_summary.json'}")


if __name__ == "__main__":
    main()