import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv("customer_support_tickets.csv")

df = df.dropna(subset=["Ticket Description", "Ticket Type"])

X = df["Ticket Description"]
y = df["Ticket Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

report = classification_report(y_test, y_pred)

print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

print("Report saved as classification_report.txt")
