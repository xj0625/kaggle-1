import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 读取处理后的数据
df = pd.read_csv('processed_train_data.csv')

# 准备特征和标签
X = df['processed_review']
y = df['sentiment']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 尝试不同的向量器
vectorizers = {
    'CountVectorizer': CountVectorizer(max_features=5000),
    'TfidfVectorizer': TfidfVectorizer(max_features=5000)
}

# 尝试不同的分类器
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'MultinomialNB': MultinomialNB(),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 评估不同组合
for vec_name, vectorizer in vectorizers.items():
    print(f"\n使用 {vec_name}:")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    for clf_name, clf in classifiers.items():
        clf.fit(X_train_vec, y_train)
        y_pred_proba = clf.predict_proba(X_test_vec)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"  {clf_name} AUC: {auc:.4f}")

# 对最佳组合进行参数调优
print("\n参数调优:")
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [1000]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_vec, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# 测试最佳模型
best_clf = grid_search.best_estimator_
X_test_vec = vectorizer.transform(X_test)
y_pred_proba = best_clf.predict_proba(X_test_vec)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"最佳模型测试AUC: {auc:.4f}")

# 保存最佳模型
import joblib
joblib.dump(best_clf, 'best_sentiment_model.pkl')
joblib.dump(vectorizer, 'best_vectorizer.pkl')
print("\n最佳模型已保存到 best_sentiment_model.pkl")
print("最佳向量器已保存到 best_vectorizer.pkl")
