import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the datasets
fakeNews = pd.read_csv('Fake.csv')
realNews = pd.read_csv('True.csv')

import matplotlib.pyplot as plt
# Check out the distribution of fake news compare to real news
plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(fakeNews), color='blue')
plt.bar('Real News', len(realNews), color='red')
plt.title('Distribution of Fake News and Real News', size=15)
plt.xlabel('News Type', size=15)
plt.ylabel('# of News Articles', size=15)


total_len = len(fakeNews) + len(realNews)
plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(fakeNews) / total_len, color='blue')
plt.bar('Real News', len(realNews) / total_len, color='red')
plt.title('Distribution of Fake News and Real News', size=15)
plt.xlabel('News Type', size=15)
plt.ylabel('Proportion of News Articles', size=15)

# Add a label column to indicate real or fake news
fakeNews['isReal'] = 0
realNews['isReal'] = 1

# Concatenate the datasets
df = pd.concat([fakeNews, realNews], axis=0)
print(df.head(10))
print(df.tail(10))
# Split the dataset into training and testing sets
x = df["text"]
y = df["isReal"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Tokenize the text using Keras Tokenizer
max_features = 2000  # Set the maximum number of features
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)

# Convert text to sequences
x_train_sequences = tokenizer.texts_to_sequences(x_train)
x_test_sequences = tokenizer.texts_to_sequences(x_test)

# Pad sequences to ensure equal length
max_sequence_length = max([len(sequence) for sequence in x_train_sequences])
x_train_padded = pad_sequences(x_train_sequences, maxlen=max_sequence_length)
x_test_padded = pad_sequences(x_test_sequences, maxlen=max_sequence_length)

# Convert the padded sequences back to text
x_train_padded_text = tokenizer.sequences_to_texts(x_train_padded)
x_test_padded_text = tokenizer.sequences_to_texts(x_test_padded)

# Train the TfidfVectorizer on the training set and transform both training and testing sets
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train_padded_text)
xv_test = vectorization.transform(x_test_padded_text)

# Train the Logistic Regression model on padded sequences
lr = LogisticRegression()
lr.fit(xv_train, y_train)

# Make predictions on the padded testing set
pred_lr = lr.predict(xv_test)

# Evaluate the LR model
print(classification_report(y_test, pred_lr))
accuracy = accuracy_score(y_test, pred_lr)
print("Accuracy:", accuracy)


# Plot the confusion matrix
cm = confusion_matrix(y_test, pred_lr)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# Add numerical values to the confusion matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black', size=15)

plt.title('Confusion Matrix', size=15)
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Fake', 'Real'], rotation=45, size=12)
plt.yticks(tick_marks, ['Fake', 'Real'], size=12)
plt.xlabel('Predicted Label', size=12)
plt.ylabel('True Label', size=12)

plt.show()
