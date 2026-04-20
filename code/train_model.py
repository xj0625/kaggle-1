import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# 读取处理后的数据
df = pd.read_csv('processed_train_data.csv')

# 准备特征和标签
X = df['processed_review']
y = df['sentiment']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建词袋模型
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 训练分类器
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# 预测
y_pred = clf.predict(X_test_vec)
y_pred_proba = clf.predict_proba(X_test_vec)[:, 1]

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"模型准确率: {accuracy:.4f}")
print(f"模型AUC: {auc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 保存模型和向量器
import joblib
joblib.dump(clf, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\n模型已保存到 sentiment_model.pkl")
print("向量器已保存到 vectorizer.pkl")
