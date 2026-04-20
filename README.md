# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：谢洁
- **学号**：112305010107
- **班级**：数据1231

## 2. 仓库结构
```
kaggle-1/
├── code/                  # 实验代码
├── report/                # 实验报告
├── results/               # 实验结果（提交文件）
├── images/                # 截图
├── README.md              # 仓库说明
├── requirements.txt       # 依赖
└── .gitignore             # 忽略文件
```

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Word2Vec NLP 教程（Bag of Words 遇见爆米花袋）
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **提交日期**：2026-04-20

- **GitHub 仓库地址**：https://github.com/xj0625/kaggle-1
- **GitHub README 地址**：https://github.com/xj0625/kaggle-1/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **最佳 Public Score**：0.96592
- **最佳 Private Score**：0.96592
- **排名**（如能看到可填写）：无

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images112305010107_谢洁_kaggle_scores.html)

> 截图保存在根目录中。
> 截图文件名：`images112305010107_谢洁_kaggle_scores.html`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**
- 使用 BeautifulSoup 去除 HTML 标签（保留纯文本）
- 将非字母字符替换为空格（仅保留 A-Z / a-z）
- 统一转小写并合并多余空白

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**
- 使用 gensim 训练 Word2Vec 词向量
- 词汇表来源：合并训练集和無標籤数据集
- 词向量维度：200
- 训练参数：window=5, min_count=3, epochs=15, sg=1 (skip-gram)
- 文档向量表示：对文档内词向量取平均（average pooling）得到固定维度向量

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**
- Word2Vec 平均向量 + 逻辑回归
- TF-IDF（词/字符 n-gram）+ NBSVM
- TF-IDF（词/字符 n-gram）+ 逻辑回归
- **最终采用：TF-IDF Word n-gram + NBSVM，融合多个模型预测结果**

---

## 7. 实验流程
请简要说明你的实验流程。

**我的实验流程：**
1. 读取训练集、测试集和無標籤数据集
2. 对文本进行预处理（去HTML、清洗、转小写）
3. 训练 Word2Vec 模型（使用训练集+無標籤数据）
4. 提取 Word2Vec 文档向量（平均池化）
5. 提取 TF-IDF 特征（词 n-gram + 字符 n-gram）
6. 训练多个分类模型（NBSVM、逻辑回归）
7. 使用 rank_mean 融合多个模型预测结果
8. 生成 submission 文件并提交 Kaggle

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

**我的项目结构：**
```text
word2vec-sentiment-analysis/
├─ code/                      # 实验代码
│  ├─ text_preprocessing.py   # 文本预处理脚本
│  ├─ make_submission.py      # 传统模型训练脚本
│  └─ enhanced_submission.py # 增强版模型训练脚本
├─ artifacts/                 # 模型缓存
├─ submission_*.csv           # 提交文件
├─ README.md                  # 实验报告
├─ requirements.txt           # 依赖
└─ .gitignore                 # 忽略文件
```

---

## 9. 模型性能

| 模型 | 验证集 AUC | 真实测试集 AUC |
|------|------------|----------------|
| Word n-gram + NBSVM | 0.9667 | 0.96592 |
| Word n-gram + LR | 0.9653 | - |
| Char n-gram + LR | 0.9595 | - |
| Word2Vec + LR | 0.9463 | 0.87552 |
| 组合特征模型 | - | 0.88008 |
| 增强版模型 | - | 0.89556 |
| 最终融合模型 | - | 0.89816 |

---

## 10. 复现方式

### 10.1 安装依赖
```bash
pip install -r requirements.txt
```

### 10.2 运行实验
```bash
# 1. 文本预处理
python code/text_preprocessing.py

# 2. 训练模型并生成提交
python code/make_submission.py
python code/enhanced_submission.py
```

---

## 11. 总结
本实验成功完成了基于 Word2Vec 的情感预测任务。通过结合 TF-IDF 特征和 NBSVM 模型，并使用 rank_mean 融合策略，最终在 Kaggle 上取得了 0.96592 的 AUC 分数。实验过程遵循了规范的 GitHub 管理流程，确保了代码和结果的可追溯性。
