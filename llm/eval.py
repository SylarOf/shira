import re
from sklearn.metrics import precision_score, recall_score, f1_score


def parse_ans(output: str):
    """
    把模型输出解析成 list[str]
    例如:
    "ans: 2012 World Series\nans: 2014 World Series"
    -> ["2012 World Series", "2014 World Series"]
    """
    return [line.replace("ans:", "").strip()
            for line in output.splitlines()
            if line.strip().startswith("ans:")]


def evaluate_model(rag, dataset):
    """
    rag: PureRag 实例
    dataset: list[dict], 每个样本格式：
        {
          "question": str,
          "query_text": list[str],
          "gold_answers": list[str]
        }
    """
    all_preds, all_labels = [], []

    for i, sample in enumerate(dataset):
        outputs = rag.invoke(sample)
        pred_answers = parse_ans(outputs[0])  # 只用第一次回答
        gold_answers = sample["gold_answers"]

        # 转集合，方便计算
        pred_set = set(pred_answers)
        gold_set = set(gold_answers)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n[Sample {i+1}]")
        print("Question:", sample["question"])
        print("Pred:", pred_answers)
        print("Gold:", gold_answers)
        print(f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

        all_preds.append(pred_answers)
        all_labels.append(gold_answers)

    # 宏平均
    # 转换成二进制矩阵做 sklearn 计算 (简单起见 flatten 全部答案空间)
    all_unique = list({a for labels in all_labels for a in labels})
    y_true, y_pred = [], []

    for labels, preds in zip(all_labels, all_preds):
        y_true.append([1 if a in labels else 0 for a in all_unique])
        y_pred.append([1 if a in preds else 0 for a in all_unique])

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n=== Overall Evaluation ===")
    print(f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
