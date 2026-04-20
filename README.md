# Word2Vec 情感预测实验

## 学生信息
- 姓名：谢洁
- 学号：112305010107
- 班级：数据1231

## 实验任务
基于给定文本数据，使用 Word2Vec 将文本转为向量特征，结合分类模型完成情感预测任务。

## 比赛信息
- 比赛名称：Word2Vec NLP 教程
- 比赛链接：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- 最终提交文件：submission_final.csv

## 实验方法

### 1. 文本预处理
- 使用 BeautifulSoup 去除 HTML 标签
- 仅保留字母字符，替换其他为空格
- 统一转小写并合并多余空白

### 2. 特征表示
- **TF-IDF**：词 n-gram (1,2) + 字符 n-gram (2,6)
- **Word2Vec**：200维向量，skip-gram模式，平均池化

### 3. 分类模型
- **NBSVM**：结合朴素贝叶斯对数似然比特征 + 逻辑回归
- **逻辑回归**：标准线性模型

### 4. 融合策略
- 使用 rank_mean 融合多个模型预测结果

## 模型性能
| 模型 | 验证集 AUC |
|------|------------|
| Word n-gram + NBSVM | 0.9667 |
| Word n-gram + LR | 0.9653 |
| Char n-gram + LR | 0.9595 |
| Word2Vec + LR | 0.9463 |

## 项目结构
```
D:\机器学习Trae\kaggle比赛1\
├── data/                  # 数据文件
├── artifacts/             # 模型缓存
├── submission_*.csv       # 提交文件
├── make_submission.py     # 主脚本
├── text_preprocessing.py  # 文本预处理
├── enhanced_submission.py # 增强版本
├── requirements.txt       # 依赖
└── README.md              # 说明文档
```

## 运行方法
1. 安装依赖：`pip install -r requirements.txt`
2. 预处理数据：`python text_preprocessing.py`
3. 训练模型：`python make_submission.py`
4. 生成最终提交：`python enhanced_submission.py`

## 提交结果
- 最终提交：submission_final.csv
- 最佳单模型：submission_best_single.csv
