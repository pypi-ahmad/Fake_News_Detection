# Fake News Detection

[![Status](https://img.shields.io/badge/status-active-success)](https://github.com/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview
This project builds a TF-IDF + Passive Aggressive classifier to detect fake news. The pipeline loads labeled articles, vectorizes their text with TF-IDF, trains a linear classifier, and evaluates accuracy on a hold-out test set.

## Key Features
- **Modern Python 3.11+** with clean, typed code and structured functions.
- **Scikit-learn pipeline** using TF-IDF and Passive Aggressive classification.
- **Robust validation** with explicit data checks and safe error handling.
- **Reproducible results** through fixed random seeds and clear configuration.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/pypi-ahmad/Fake_News_Detection.git
   cd Fake_News_Detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -U pip
   pip install numpy pandas scikit-learn jupyter
   ```

## Usage
Run the notebook in Jupyter:
```bash
jupyter notebook Fake_news_Detection.ipynb
```

Or run the workflow directly in a notebook cell:
```python
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from Fake_news_Detection import (  # if moved into a module later
    ModelConfig,
    build_vectorizer,
    evaluate_model,
    load_dataset,
    split_dataset,
    train_model,
    vectorize_text,
)

config = ModelConfig()
df = load_dataset(Path("news") / "news.csv")
x_train, x_test, y_train, y_test = split_dataset(df, config)
vectorizer = build_vectorizer(config)
x_train_vec, x_test_vec = vectorize_text(vectorizer, x_train, x_test)
model = train_model(x_train_vec, y_train, config)
predictions = model.predict(x_test_vec)
metrics = evaluate_model(y_test, predictions)
print(metrics)
```

## Project Structure
```
Fake_News_Detection/
├── Fake_news_Detection.ipynb   # Notebook with the full pipeline
├── news/
│   └── news.csv                # Labeled dataset
├── LICENSE
└── README.md
```

## License
MIT. See [LICENSE](LICENSE).```markdown
📰 Fake News Detection

This repository contains a machine learning project aimed at detecting fake news articles. The model uses a dataset of news articles to classify them as either real or fake.


📁 Repository Structure

```
Fake_News_Detection/
├── news/
│   └── news.csv                     # Dataset containing news articles
├── Fake_news_Detection.ipynb        # Jupyter Notebook for model implementation
├── LICENSE                          # License file
```

📊 Dataset

- File: `news/news.csv`
- Description: This dataset contains news articles with labels indicating whether they are fake or real.
- Columns:
  - `title`: The title of the news article.
  - `text`: The body/content of the news article.
  - `label`: The label indicating the authenticity of the article (`1` for fake and `0` for real).


🚀 Getting Started

Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Libraries:
  - Pandas
  - NumPy
  - Scikit-learn
   ```

Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fake_News_Detection.git
   cd Fake_News_Detection
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Fake_news_Detection.ipynb
   ```

3. Load the dataset:
   - Ensure `news.csv` is located in the `news/` folder.

4. Run all cells in the notebook to preprocess the data, train the model, and evaluate its performance.

---

🛠️ Tools and Technologies

- **Programming Language**: Python
- **Libraries Used**:
  - Pandas
  - NumPy
  - Scikit-learn

---

📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributions

Contributions are welcome! Feel free to fork this repository, make improvements, and submit a pull request.
