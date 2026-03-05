# エンコーダ型・デコーダ型言語モデルを併用した差別的発言スパン検出性能向上
言語処理学会第32回年次大会(NLP2026)に投稿した論文のモデル実装と評価に関するコードを公開する。

## コード  
- ベースモデル  
  XXX.ipynbでベースモデルの訓練と評価を行う。
- 多数決のモデル  
  Ensemble/XXX.pyでモデルを組み合わせて評価を行う。

## データセット  
本研究では[HateNorm](https://www.kaggle.com/competitions/hatenorm)のデータセットを用いている[^1][^2]。  
元のデータセットには2,421件の訓練データが含まれているが、その内の300件を検証データとする。  
datasets/HateNorm/valid_ids.jsonに検証データとして使用したデータのIDを列挙している。

## 環境
```
Python 3.9.21
transformers==4.56.2
torch==2.7.1
```

## 謝辞
本研究は，JST 経済安全保障重要技術育成プログラムJPMJKP24C3 の支援を受けたものである。  
また本研究は，東京科学大学のスーパーコンピュータTSUBAME4.0 を利用して実施した。

[^1]:Sarah Masud, Manjot Bedi, Mohammad Aflah Khan, Md Shad Akhtar, and Tanmoy Chakraborty. Proactively reducing the hate intensity of online posts via hate speech normalization. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, p. 3524–3534, New York, NY, USA, 2022. Association for Computing Machinery.
[^2]:Shrey Satapara, Sarah Masud, Hiren Madhu, Md. Aflah Khan, Md. Shad Akhtar, Tanmoy Chakraborty, Sandip Modha, and Thomas Mandl. Overview of the hasoc subtracks at fire 2023: Detection of hate spans and conversational hate-speech. In Proceedings of the 15th Annual Meeting of the Forum for Information Retrieval Evaluation, FIRE ’23, p. 10–12, New York, NY, USA, 2024. Association for Computing Machinery.
