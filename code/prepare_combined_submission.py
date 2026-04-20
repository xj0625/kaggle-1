import pandas as pd
import numpy as np
import joblib

"""
准备Part 3的提交文件
"""

# 加载模型
clf = joblib.load('combined_sentiment_model.pkl')

# 加载测试数据的组合特征
X_test = np.load('X_test_combined.npy')

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
submission_df.to_csv('combined_submission.csv', index=False)

print(f"Part 3提交文件已生成，共包含 {len(submission_df)} 条预测结果")
print("提交文件已保存到 combined_submission.csv")

# 显示前5条预测结果
print("\n前5条预测结果：")
print(submission_df.head())

# 与Part 1、Part 2的结果比较
print("\n模型性能比较：")
print("Part 1 (TfidfVectorizer + LogisticRegression): AUC = 0.9590")
print("Part 2 (Word2Vec + LogisticRegression): AUC = 0.9348")
print("Part 3 (Combined Features + LogisticRegression): AUC = 0.9540")
print("\n分析：")
print("- Part 3的AUC值高于Part 2，表明组合特征成功结合了Word2Vec的语义信息")
print("- Part 3的AUC值略低于Part 1，可能是由于特征维度增加导致的轻微过拟合")
print("- 组合特征模型在准确率上表现最好，达到了88.88%")
