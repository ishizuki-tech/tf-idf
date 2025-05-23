# 📘 TF-IDF 類似度評価スクリプト

このプロジェクトは、スワヒリ語の自由回答（例：音声認識出力）に対して、事前に定義された教育関連のフレーズと **TF-IDF（Term Frequency-Inverse Document Frequency）** に基づく **類似度スコア** を算出し、自動で評価を行う Python スクリプトです。

---

## 🧰 使用技術

* Python 3.x
* scikit-learn
* pandas
* onnx, onnxruntime（将来的なモデル変換に対応）
* JSON（TF-IDFモデル保存用）

---

## 📦 セットアップ手順

```bash
# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# パッケージのインストール
pip install --upgrade pip
pip install scikit-learn==1.3.2 skl2onnx==1.15.0 onnx==1.14.1 onnxruntime pandas
```

---

## 📝 スクリプトの使い方

### 1. 期待される回答の準備

```python
expected_answers = [
  "ndiyo, watoto huenda shule",               # はい、子どもたちは学校へ行きます
  "kila siku wanafunzi husoma",               # 毎日、生徒たちは勉強します
  ...
]
```

教師なし分類でも対応できるよう、**教育に関する多様な自然言語表現**を収録しています。

---

### 2. 類似度スコアの計算

```python
user_answer = "wanahitaji shule kwa"

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(expected_answers + [user_answer])

similarities = cosine_similarity(X[-1], X[:-1]).flatten()
```

ユーザーの自由回答に対して、各 `expected_answer` との類似度を計算します。

---

### 3. 上位スコアの表示

```python
ranked = sorted(zip(similarities, expected_answers), reverse=True)

for i, (sim, ref) in enumerate(ranked[:15], 1):
    print(f"{i}. Score: {sim:.3f} → \"{ref}\"")
```

スコアが高い順に回答例を表示します（最大15件）。しきい値（例：`0.6`）を用いて妥当性の評価も可能です。

---

### 4. TF-IDF モデルの保存

```python
tfidf_data = {
    "vocabulary": vectorizer.vocabulary_,
    "idf": vectorizer.idf_.tolist(),
    "expected_answers": expected_answers
}

with open("tfidf_model.json", "w", encoding="utf-8") as f:
    json.dump(tfidf_data, f, ensure_ascii=False, indent=2)
```

この JSON ファイルは、Android など他の環境で再利用可能な軽量モデル形式です。

---

## 📂 出力ファイル

* `tfidf_model.json`
  学習済み語彙（vocabulary）、IDF値、期待される回答リストを含むモデルファイル。

---

## 🔮 応用例

* Whisper などの音声認識出力の妥当性評価
* Android アプリでのオフライン質問応答評価
* LLM を使わない軽量な意図分類・フィードバック処理
* 音声・教育・評価ツールのフィールドアプリケーション

---

## ⚠️ 注意事項

* 入力文（`user_answer`）が短いと類似度が低くなる可能性があります。
* `ngram_range=(1, 2)` により、単語・2語連結表現の両方に対応し、精度向上を狙っています。

---
