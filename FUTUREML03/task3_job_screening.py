import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

data = pd.read_csv("monster_com-job_sample.csv", encoding="latin1")

data = data[["job_description", "sector"]].dropna()

print("Dataset loaded:", data.shape)

X = data["job_description"]
y = data["sector"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.85,
        min_df=10,
        ngram_range=(1, 1)
    )),
    ("clf", LinearSVC(dual="auto", max_iter=2000))
])

model.fit(X_train, y_train)

print("Model training completed")

y_pred = model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample_job = [
    "Looking for a data analyst with strong Python, SQL, and machine learning skills"
]

prediction = model.predict(sample_job)

print("\nSample Job Prediction")
print("Predicted Sector:", prediction[0])
