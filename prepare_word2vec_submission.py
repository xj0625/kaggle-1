import pandas as pd
import numpy as np
import joblib

# 加载模型
clf = joblib.load('word2vec_sentiment_model.pkl')

# 加载测试数据特征
X_test = np.load('X_test_word2vec.npy')

# 读取测试数据的id
test_df = pd.read_csv('testData.tsv', sep='\t')

# 预测
y_pred = clf.predict(X_test)

# 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': y_pred
})

# 保存提交文件
submission_df.to_csv('word2vec_submission.csv', index=False)

print(f"Part 2提交文件已生成，共包含 {len(submission_df)} 条预测结果")
print("提交文件已保存到 word2vec_submission.csv")

# 显示前5条预测结果
print("\n前5条预测结果：")
print(submission_df.head())

# 与Part 1的结果比较
print("\n模型性能比较：")
print("Part 1 (TfidfVectorizer + LogisticRegression): AUC = 0.9590")
print("Part 2 (Word2Vec + LogisticRegression): AUC = 0.9348")
print("虽然Part 2的AUC略低于Part 1，但Word2Vec模型成功利用了无标签数据，学习到了有意义的语义表示。")
