# 基于NLP的商品评论情感分析
#### 环境安装
执行`pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple`
#### 数据准备
- 从京东商城上爬取商品评论数据（只爬取好评和差评）
- 爬取数据的逻辑见reptiles_jd/reptile.py
#### 基于情感词典的情感分析
- 情感词典的来源
    - 情感词莱阳BonsonNLP与知网整合
    - 停用词莱阳snowNLP
    - 程度副词来源知网
    - 否定词在其他博客中下载
- 整理情感词典(emotion_dict/dict_deal.py)