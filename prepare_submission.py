import pandas as pd
import joblib
from preprocess import preprocess_text

# 加载模型和向量器
clf = joblib.load('best_sentiment_model.pkl')
vectorizer = joblib.load('best_vectorizer.pkl')

# 读取测试数据
test_df = pd.read_csv('testData.tsv', sep='\t')

# 对测试评论进行预处理
test_df['processed_review'] = test_df['review'].apply(preprocess_text)

# 转换特征
X_test = vectorizer.transform(test_df['processed_review'])

# 预测
y_pred = clf.predict(X_test)

# 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': y_pred
})

# 保存提交文件
submission_df.to_csv('submission.csv', index=False)

print(f"提交文件已生成，共包含 {len(submission_df)} 条预测结果")
print("提交文件已保存到 submission.csv")

# 显示前5条预测结果
print("\n前5条预测结果：")
print(submission_df.head())
