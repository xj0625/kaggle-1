import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

"""
构建基于组合特征的情感分析模型
"""

# 加载组合特征和标签
print("加载组合特征...")
X = np.load('X_train_combined.npy')
y = np.load('y_train.npy')

print(f"组合特征形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练分类器
print("\n训练分类模型...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"模型准确率: {accuracy:.4f}")
print(f"模型AUC: {auc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 保存模型
model_name = 'combined_sentiment_model.pkl'
joblib.dump(clf, model_name)
print(f"\n模型已保存到 {model_name}")

print("\n基于组合特征的模型训练完成！")
