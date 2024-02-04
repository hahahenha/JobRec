import numpy as np

def hit_rate(recommended_items, test_items):
    """计算推荐的命中率"""
    hits = 0
    for item in recommended_items:
        if item in test_items:
            hits += 1
    return hits > 0

def average_precision(recommended_items, test_items):
    """计算平均精确度"""
    hits = 0
    sum_precisions = 0
    for n, item in enumerate(recommended_items, start=1):
        if item in test_items:
            hits += 1
            sum_precisions += hits / n
    return sum_precisions / min(len(test_items), len(recommended_items))

def precision(recommended_items, test_items):
    """计算精确度"""
    hits = sum(item in test_items for item in recommended_items)
    return hits / len(recommended_items)

def recall(recommended_items, test_items):
    """计算召回率"""
    hits = sum(item in test_items for item in recommended_items)
    return hits / len(test_items)

def f1_score(precision, recall):
    """计算 F1 分数"""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def mean_reciprocal_rank(recommended_items, test_items):
    """计算 MRR"""
    for idx, item in enumerate(recommended_items, start=1):
        if item in test_items:
            return 1 / idx
    return 0

def dcg(recommended_items, test_items):
    """计算 DCG"""
    score = 0
    for idx, item in enumerate(recommended_items, start=1):
        if item in test_items:
            score += 1 / np.log2(idx + 1)
    return score

def ndcg(recommended_items, test_items):
    """计算 NDCG"""
    actual_dcg = dcg(recommended_items, test_items)
    ideal_dcg = dcg(sorted(test_items, key=lambda x: recommended_items.index(x) if x in recommended_items else float('inf')), test_items)
    if ideal_dcg == 0:
        return 0
    return actual_dcg / ideal_dcg