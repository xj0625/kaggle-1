import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# 加载特征和标签
X = np.load('X_train_word2vec.npy')
y = np.load('y_train.npy')

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练分类器
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
joblib.dump(clf, 'word2vec_sentiment_model.pkl')
print("\n模型已保存到 word2vec_sentiment_model.pkl")
