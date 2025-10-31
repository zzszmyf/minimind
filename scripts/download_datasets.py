#!/usr/bin/env python
"""
下载 MiniMind 数据集从 HuggingFace

从 HuggingFace 数据集仓库下载所有必要的数据文件到 ../../dataset 目录
数据集地址: https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# 项目根目录（scripts 文件夹的上一级）
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT.parent / "dataset"

# 确保数据集目录存在
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace 数据集信息
REPO_ID = "jingyaogong/minimind_dataset"
REPO_TYPE = "dataset"

# 所有需要下载的数据文件列表
# 根据项目训练脚本的需求列出所有数据文件
DATASET_FILES = [
    # 预训练数据
    ("pretrain_hq.jsonl", "pretrain_hq.jsonl"),

    # SFT 训练数据
    ("sft_mini_512.jsonl", "sft_mini_512.jsonl"),

    # DPO 训练数据
    ("dpo.jsonl", "dpo.jsonl"),

    # RLAIF/RL 训练数据
    ("rlaif-mini.jsonl", "rlaif-mini.jsonl"),

    # LoRA 训练数据
    ("lora_identity.jsonl", "lora_identity.jsonl"),

    # 推理蒸馏数据
    ("r1_mix_1024.jsonl", "r1_mix_1024.jsonl"),
]


def download_datasets():
    """下载所有数据文件"""
    print(f"开始下载数据集到: {DATASET_DIR}")
    print("=" * 60)

    failed_files = []

    for file_name, local_name in DATASET_FILES:
        local_path = DATASET_DIR / local_name

        # 如果文件已存在，跳过
        if local_path.exists():
            file_size = local_path.stat().st_size / (1024 * 1024)  # 转换为 MB
            print(f"✓ {file_name:30s} - 已存在 ({file_size:.2f} MB)")
            continue

        try:
            print(f"↓ {file_name:30s} - 下载中...", end="")
            sys.stdout.flush()

            # 从 HuggingFace 下载文件
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=file_name,
                repo_type=REPO_TYPE,
                local_dir=str(DATASET_DIR),
                local_dir_use_symlinks=False,
            )

            file_size = Path(path).stat().st_size / (1024 * 1024)  # 转换为 MB
            print(f" 完成 ({file_size:.2f} MB)")

        except Exception as e:
            print(f" 失败")
            failed_files.append((file_name, str(e)))
            print(f"  错误: {e}")

    print("=" * 60)

    # 总结下载结果
    total_files = len(DATASET_FILES)
    successful_files = total_files - len(failed_files)

    print(f"\n下载总结:")
    print(f"  总共: {total_files} 个文件")
    print(f"  成功: {successful_files} 个文件")
    print(f"  失败: {len(failed_files)} 个文件")

    if failed_files:
        print(f"\n失败的文件:")
        for file_name, error in failed_files:
            print(f"  - {file_name}: {error}")
        return False

    print(f"\n数据集已成功下载到: {DATASET_DIR}")

    # 验证所有文件
    print("\n验证文件:")
    for file_name, local_name in DATASET_FILES:
        local_path = DATASET_DIR / local_name
        if local_path.exists():
            file_size = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {local_name:30s} ({file_size:8.2f} MB)")
        else:
            print(f"  ✗ {local_name:30s} (缺失)")

    return True


if __name__ == "__main__":
    try:
        success = download_datasets()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n下载被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
