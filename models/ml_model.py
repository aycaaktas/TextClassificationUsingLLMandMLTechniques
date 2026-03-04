import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import  precision_recall_fscore_support
import seaborn as sns
from tqdm import tqdm  


stopwords = ['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']

# Step 1: Load the datasets (train and test separately)
train_data = pd.read_csv('data/cleaned_train_data_ml.tsv', sep='\t')
test_data = pd.read_csv('data/cleaned_test_data.tsv', sep='\t')

# Step 2: Preprocess the data (assumes data is already cleaned)
def remove_custom_stopwords(sentence):
    return ' '.join([word for word in sentence.split() if word not in stopwords]) 


train_data['clean_sentence'] = [remove_custom_stopwords(sentence) for sentence in tqdm(train_data['clean_sentence'], desc="Preprocessing Train Data")]
test_data['clean_sentence'] = [remove_custom_stopwords(sentence) for sentence in tqdm(test_data['clean_sentence'], desc="Preprocessing Test Data")]

# Step 3: TF-IDF Vectorization (Complete - fit on training data only)
tfidf_vectorizer = TfidfVectorizer(max_features=None)                   
X_train = tfidf_vectorizer.fit_transform(train_data['clean_sentence'])
X_test = tfidf_vectorizer.transform(test_data['clean_sentence'])


# Step 4: Encode labels (convert categories to numerical labels)

categories = pd.get_dummies(train_data['domain']).columns
y_train = pd.get_dummies(train_data['domain']).reindex(columns=categories, fill_value=0).values
y_test = pd.get_dummies(test_data['domain']).reindex(columns=categories, fill_value=0).values


## Step 5: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial', verbose=1) 
model.fit(X_train, y_train.argmax(axis=1)) 

# Step 6: Predict on test data

y_pred = []
for i in tqdm(range(X_test.shape[0]), desc="Predicting Test Data"):
    y_pred.append(model.predict(X_test[i]))

y_pred = np.array(y_pred).flatten()

# Step 7: Performance metrics - suitable for imbalanced datasets
report = classification_report(y_test.argmax(axis=1), y_pred)
print("Classification Report:")
print(report)

with open('../results/ml/classification_report.txt', 'w') as f:
    f.write(report)

conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('../results/ml/confusion_matrix_ml.png')
plt.close()

np.savetxt('../results/ml/confusion_matrix_ml.txt', conf_matrix, fmt='%d')



# Step 8: Calculate weighted Precision, Recall, F1-score, Accuracy, and Balanced Accuracy
precision_weighted, recall_weighted, fscore_weighted, _ = precision_recall_fscore_support(y_test.argmax(axis=1), y_pred, average='weighted')
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
balanced_accuracy = balanced_accuracy_score(y_test.argmax(axis=1), y_pred)

precision_weighted *= 100
recall_weighted *= 100
fscore_weighted *= 100
accuracy *= 100
balanced_accuracy *= 100


metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Balanced Accuracy']
values = [precision_weighted, recall_weighted, fscore_weighted, accuracy, balanced_accuracy]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.ylabel('Score (%)')
plt.ylim(0, 100)
plt.title('Overall Performance Metrics (Weighted Average)')

plt.tight_layout()
plt.savefig('../results/ml/overall_performance_metrics_with_accuracy_and_balanced_accuracy.png')
plt.close()

with open('../results/ml/performance_metrics_ml.txt', 'w') as f:
    f.write(f'Weighted Precision: {precision_weighted:.2f}\n')
    f.write(f'Weighted Recall: {recall_weighted:.2f}\n')
    f.write(f'Weighted F1-Score: {fscore_weighted:.2f}\n')
    f.write(f'Accuracy: {accuracy:.2f}\n')
    f.write(f'Balanced Accuracy: {balanced_accuracy:.2f}\n')