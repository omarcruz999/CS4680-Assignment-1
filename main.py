# Problem: Classifying an email as spam or not spam (ham)
# Target: Spam or not spam
# Features: Email text
# Dataset Link: https://www.kaggle.com/datasets/mfaisalqureshi/spam-email

""" 
Model Evaluation

The classification model was more suitable for this problem because it is naturally a classification 
problem since we are sorting emails into two categories. Linear regression does not fit well with 
this problem because it is used for predicting a continuous number. Forcing the model's output into a category
can lead to less reliable results as shown in the program's output.

"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

# Model - Classification
print('\n-------- Classification --------')

# Read the dataset from the CSV File
df = pd.read_csv('spam.csv')

# Feature data
x = df['Message']

# Target variables 
y = df['Category']

# Convert text into numerical format
vectorizer = CountVectorizer()
x_vectorized = vectorizer.fit_transform(x)

# Train model using vectorized data
model = SVC()
model.fit(x_vectorized, y)

# Define our test emails for classification
test_emails = [
    "Get 150 free msgs per week! Txt JOIN to 69696. Std txt rate. Reply STOP to end",
    "FINAL notice: Your mobile service update due. Call 08000930705 to confirm",
    "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 now",
    "Can we meet at bugis later? I finish work at 6",
    "Dinner at 7 la, I book table already",
    "I left my wallet at home, can borrow 10"
]

for text in test_emails:
    # Transform the new text
    text_vectorized = vectorizer.transform([text])
    
    # Predict using the trained model
    prediction = model.predict(text_vectorized)
    
    # Get the label ('spam' or 'ham') from the prediction array
    label = prediction[0]
    
    # Adjust the label
    final_label = "spam" if label == "spam" else "not spam"
    
    # Print the results
    print(f"'{text}' is {final_label}")

# Model - Linear Regression

from sklearn.linear_model import LinearRegression

print('\n-------- Linear Regression --------')

# Read the dataset from the CSV File
df = pd.read_csv('spam.csv')

# Feature data
x = df['Message']

# Target variables - mapped to numbers
y = df['Category'].map({'ham': 0, 'spam': 1}).values

# Convert text into numerical format
vectorizer = CountVectorizer()
x_vectorized = vectorizer.fit_transform(x)

# Train model using vectorized data
model = LinearRegression()
model.fit(x_vectorized, y)

for text in test_emails:
    # Transform the new text
    text_vectorized = vectorizer.transform([text])
    
    # Predict a score
    score = float(model.predict(text_vectorized)[0])

    # Round the score from 0 to 1
    score_rounded = max(0.0, min(1.0, score))
    
    # Set a label 
    final_label = "spam" if score >= 0.5 else "not spam"
    
    # Print the results
    print(f"score={score_rounded:.3f} | {text}: is {final_label}")

print('\n')