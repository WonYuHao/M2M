#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在同一异常类别(_info)内为每条样本构建 visual prompt 候选：
1) 优先同prompt精确匹配(排除自身)
2) 再用同类别语义相似prompt补充(排除自身)
3) 若仍没有任何非自身候选，则同类别随机选择一个非自身作为兜底
4) 允许把自身加入候选，但候选中必须至少有1条非自身
5) 每条样本候选数限制为 1~5

输出：
- /home/wyh/data/mvtec_ad/train_prompts_wiGT_full_with_visual_match.json
- /home/wyh/data/mvtec_ad/semantic_prompt_match_report.txt
"""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# ====== 硬编码路径（按需修改）======
INPUT_JSON = "/home/wyh/data/mvtec_ad/train_prompts_wiGT_full.json"
OUTPUT_JSON = "/home/wyh/data/mvtec_ad/train_prompts_wiGT_full_with_visual_match.json"
OUTPUT_REPORT = "/home/wyh/data/mvtec_ad/semantic_prompt_match_report.txt"

# 每张图候选数：1~5
MAX_CANDIDATES = 5
# 是否允许把自身加入候选（不会作为唯一候选）
ALLOW_SELF_CANDIDATE = True
# 语义模型
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 随机种子（用于兜底随机）
RANDOM_SEED = 42
# ================================


def _load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "未找到 sentence-transformers。请先安装: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name)


def _cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    emb = emb / norms
    return emb @ emb.T


def _build_neighbors(prompts: List[str], embedder) -> Dict[str, List[Tuple[str, float]]]:
    """返回每个prompt的相似prompt列表(降序，不含自己)。"""
    if len(prompts) <= 1:
        return {p: [] for p in prompts}

    emb = embedder.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
    sim = _cosine_sim_matrix(emb)

    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    for i, p in enumerate(prompts):
        idx_sorted = np.argsort(-sim[i])
        cur = []
        for j in idx_sorted:
            if j == i:
                continue
            cur.append((prompts[j], float(sim[i, j])))
        neighbors[p] = cur
    return neighbors


def _append_candidate(
    candidates: List[dict],
    used_indices: set,
    chosen_idx: int,
    chosen_item: dict,
    score: float,
    match_type: str,
):
    if chosen_idx in used_indices:
        return
    used_indices.add(chosen_idx)
    candidates.append(
        {
            "index": chosen_idx,
            "image": chosen_item.get("image"),
            "prompt": chosen_item.get("prompt"),
            "score": round(float(score), 6),
            "type": match_type,
            "is_self": False,
        }
    )


def main():
    random.seed(RANDOM_SEED)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("期望 JSON 顶层是 list[dict]，请检查数据格式。")
    if len(data) == 0:
        raise ValueError("输入数据为空。")

    embedder = _load_embedder(MODEL_NAME)

    # 分组：类别 -> 样本下标列表
    category_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        category = str(item.get("_info", "UNKNOWN")).strip()
        category_to_indices[category].append(idx)

    total = 0
    exact_primary_cnt = 0
    semantic_primary_cnt = 0
    random_fallback_primary_cnt = 0
    contain_self_cnt = 0

    report_lines = []
    report_lines.append("# Semantic prompt matching report (within same anomaly category)\n")

    for category in sorted(category_to_indices.keys()):
        indices = category_to_indices[category]

        # 类别内：prompt -> indices
        prompt_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            prompt = str(data[idx].get("prompt", "")).strip()
            if prompt:
                prompt_to_indices[prompt].append(idx)

        unique_prompts = sorted(prompt_to_indices.keys())
        neighbors_in_cat = _build_neighbors(unique_prompts, embedder)

        report_lines.append(f"## Category: {category}")
        report_lines.append(f"samples={len(indices)}, unique_prompts={len(unique_prompts)}")

        for idx in indices:
            total += 1
            item = data[idx]
            prompt = str(item.get("prompt", "")).strip()

            candidates: List[dict] = []
            used_indices = set()

            # 1) 同prompt精确匹配（排除自身）
            same_prompt_pool = [j for j in prompt_to_indices.get(prompt, []) if j != idx]
            random.shuffle(same_prompt_pool)
            for j in same_prompt_pool:
                _append_candidate(candidates, used_indices, j, data[j], 1.0, "exact")
                if len(candidates) >= MAX_CANDIDATES:
                    break

            # 2) 同类别语义匹配（排除自身）
            if len(candidates) < MAX_CANDIDATES and prompt:
                for cand_prompt, score in neighbors_in_cat.get(prompt, []):
                    cand_pool = [j for j in prompt_to_indices.get(cand_prompt, []) if j != idx]
                    random.shuffle(cand_pool)
                    for j in cand_pool:
                        _append_candidate(candidates, used_indices, j, data[j], score, "semantic")
                        if len(candidates) >= MAX_CANDIDATES:
                            break
                    if len(candidates) >= MAX_CANDIDATES:
                        break

            # 3) 如果除自身外没有候选 -> 同类别随机选择1个非自身兜底
            non_self_pool_in_cat = [j for j in indices if j != idx]
            if len(candidates) == 0:
                j = random.choice(non_self_pool_in_cat)
                _append_candidate(candidates, used_indices, j, data[j], 0.0, "random_in_category_fallback")

            # 4) 允许自身加入候选（但不会是唯一候选）
            if ALLOW_SELF_CANDIDATE and len(candidates) < MAX_CANDIDATES:
                _append_candidate(candidates, used_indices, idx, data[idx], 1.0, "self")
                # 标记self
                if candidates and candidates[-1]["index"] == idx:
                    candidates[-1]["is_self"] = True

            # 若 self 不是最后追加（极小概率结构改动时），统一修正标记
            for c in candidates:
                c["is_self"] = c["index"] == idx

            # 限制到 1~5
            candidates = candidates[:MAX_CANDIDATES]

            # 二次保证：至少有1条非自身
            has_non_self = any(c["index"] != idx for c in candidates)
            if not has_non_self:
                if len(non_self_pool_in_cat) == 0:
                    raise RuntimeError(
                        f"类别 {category} 只有1个样本(index={idx})，无法满足'至少1条非自身候选'约束。"
                    )
                j = random.choice(non_self_pool_in_cat)
                _append_candidate(candidates, set(c["index"] for c in candidates), j, data[j], 0.0, "random_in_category_fallback")
                candidates = candidates[:MAX_CANDIDATES]

            # 主候选 = 第一条（一定是非自身）
            primary = candidates[0]
            primary_type = primary["type"]
            if primary_type == "exact":
                exact_primary_cnt += 1
            elif primary_type == "semantic":
                semantic_primary_cnt += 1
            elif primary_type == "random_in_category_fallback":
                random_fallback_primary_cnt += 1

            if any(c["index"] == idx for c in candidates):
                contain_self_cnt += 1

            # 回写字段
            item["visual_match_candidates"] = candidates
            item["visual_match_has_non_self"] = True
            item["visual_match_index"] = primary["index"]
            item["visual_match_image"] = primary["image"]
            item["visual_match_prompt"] = primary["prompt"]
            item["visual_match_score"] = primary["score"]
            item["visual_match_type"] = primary_type

        # 类别内 prompt 频次简表
        for p, cnt_list in sorted(prompt_to_indices.items(), key=lambda x: (-len(x[1]), x[0])):
            report_lines.append(f'"{p}"\t{len(cnt_list)}')
        report_lines.append("")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    report_lines.append("# Summary")
    report_lines.append(f"total_samples={total}")
    report_lines.append(f"primary_exact={exact_primary_cnt}")
    report_lines.append(f"primary_semantic_in_category={semantic_primary_cnt}")
    report_lines.append(f"primary_random_in_category_fallback={random_fallback_primary_cnt}")
    report_lines.append(f"samples_containing_self_candidate={contain_self_cnt}")
    report_lines.append(f"max_candidates_per_sample={MAX_CANDIDATES}")
    report_lines.append(f"allow_self_candidate={ALLOW_SELF_CANDIDATE}")
    report_lines.append(f"model={MODEL_NAME}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines).rstrip() + "\n")

    print(f"Done.\nSaved JSON: {OUTPUT_JSON}\nSaved report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
