# 📘 TF-IDF Similarity Evaluation Script

This project evaluates free-form Swahili responses (e.g., output from speech recognition) against a set of predefined education-related phrases using **TF-IDF (Term Frequency-Inverse Document Frequency)**. It computes **similarity scores** and automatically assesses the relevance of the input using a lightweight Python script.

---

## 🧰 Technologies Used

* Python 3.x  
* scikit-learn  
* pandas  
* onnx, onnxruntime (for future model conversion support)  
* JSON (for saving the TF-IDF model)

---

## 📦 Setup Instructions

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install scikit-learn==1.3.2 skl2onnx==1.15.0 onnx==1.14.1 onnxruntime pandas
```

---

## 📝 How to Use the Script

### 1. Prepare Expected Answers

```python
expected_answers = [
  "ndiyo, watoto huenda shule",               # Yes, children go to school
  "kila siku wanafunzi husoma",               # Students study every day
  ...
]
```

The list includes a variety of **natural language expressions related to education**, designed for unsupervised classification scenarios.

---

### 2. Calculate Similarity Score

```python
user_answer = "wanahitaji shule kwa"

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(expected_answers + [user_answer])

similarities = cosine_similarity(X[-1], X[:-1]).flatten()
```

This calculates the similarity between the user's free-form answer and each of the expected reference answers.

---

### 3. Display Top Matches

```python
ranked = sorted(zip(similarities, expected_answers), reverse=True)

for i, (sim, ref) in enumerate(ranked[:15], 1):
    print(f"{i}. Score: {sim:.3f} → \"{ref}\"")
```

Displays the top 15 reference phrases in descending order of similarity score. You can apply a threshold (e.g., `0.6`) to determine if the answer is valid.

---

### 4. Save the TF-IDF Model

```python
tfidf_data = {
    "vocabulary": vectorizer.vocabulary_,
    "idf": vectorizer.idf_.tolist(),
    "expected_answers": expected_answers
}

with open("tfidf_model.json", "w", encoding="utf-8") as f:
    json.dump(tfidf_data, f, ensure_ascii=False, indent=2)
```

This JSON model is lightweight and portable for use in other environments like Android.

---

## 📂 Output File

* `tfidf_model.json`  
  Contains the trained vocabulary, IDF values, and list of expected answers.

---

## 🔮 Example Use Cases

* Validating outputs from speech recognition tools like Whisper  
* Offline Q&A scoring in Android apps  
* Lightweight intent classification and feedback without LLMs  
* Field-ready tools for voice-based education and assessment

---

## ⚠️ Notes

* Short user inputs may result in lower similarity scores.  
* The `ngram_range=(1, 2)` setting helps improve accuracy by capturing both single and bi-word expressions.

---