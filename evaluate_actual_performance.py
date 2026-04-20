import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib

"""
评估模型在完整测试集上的实际性能
"""

# 加载模型
clf = joblib.load('word2vec_sentiment_model.pkl')

# 加载测试数据特征
X_test = np.load('X_test_word2vec.npy')

# 注意：我们没有测试集的真实标签，所以无法计算实际AUC
# 这里我们只能检查模型的预测概率分布

# 获取预测概率
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# 分析预测概率的分布
print("预测概率分布分析:")
print(f"预测概率均值: {y_pred_proba.mean():.4f}")
print(f"预测概率标准差: {y_pred_proba.std():.4f}")
print(f"预测概率最小值: {y_pred_proba.min():.4f}")
print(f"预测概率最大值: {y_pred_proba.max():.4f}")

# 分析预测结果的分布
y_pred = clf.predict(X_test)
print(f"\n预测结果分布:")
print(f"正面情感(1)数量: {sum(y_pred)}")
print(f"负面情感(0)数量: {len(y_pred) - sum(y_pred)}")

# 检查模型在训练集上的性能
print("\n重新评估模型在训练集上的性能:")
X_train = np.load('X_train_word2vec.npy')
y_train = np.load('y_train.npy')

# 在训练集上的预测
y_train_pred_proba = clf.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, y_train_pred_proba)
print(f"训练集AUC: {train_auc:.4f}")

# 检查Word2Vec模型的词汇表大小
from gensim.models import Word2Vec
word2vec_model = Word2Vec.load('word2vec.model')
print(f"\nWord2Vec模型词汇表大小: {len(word2vec_model.wv)}")

print("\n分析完成。实际AUC值可能与验证集不同，因为：")
print("1. 验证集是训练数据的一部分，而测试集是独立的")
print("2. 测试集可能包含与训练集不同的词汇和表达")
print("3. Word2Vec模型的词汇表可能不包含测试集中的所有词汇")
