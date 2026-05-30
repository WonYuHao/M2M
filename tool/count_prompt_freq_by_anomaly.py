
import json
from collections import Counter, defaultdict

# ====== 硬编码路径（按需修改）======
INPUT_JSON = "/home/wyh/data/mvtec_ad/train_prompts_wiGT_full.json"
OUTPUT_TXT = "/home/wyh/data/mvtec_ad/prompt_freq_by_anomaly.txt"
# ================================


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("期望 JSON 顶层是 list[dict]，请检查文件结构。")

    # 每个异常类别（_info）单独统计 prompt 频率
    # category -> Counter(prompt -> freq)
    category_prompt_counter = defaultdict(Counter)

    for item in data:
        if not isinstance(item, dict):
            continue

        category = str(item.get("_info", "UNKNOWN")).strip()
        prompt = str(item.get("prompt", "")).strip()

        if prompt:
            category_prompt_counter[category][prompt] += 1

    lines = []
    lines.append("# Prompt frequency within each anomaly category\n")

    for category in sorted(category_prompt_counter.keys()):
        lines.append(f"## Category: {category}")
        prompt_counter = category_prompt_counter[category]

        # 按出现频率降序，再按文本字典序
        for prompt, freq in sorted(prompt_counter.items(), key=lambda x: (-x[1], x[0])):
            # 你需要的输出形式："text具体内容" + "出现频率"
            lines.append(f'"{prompt}"\t{freq}')

        lines.append("")

    output_text = "\n".join(lines).rstrip() + "\n"

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Done. Saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
