import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler



file_path = 'data/TWNERTC_TC_Fine Grained NER_DomainIndependent_NoiseReduction.DUMP'  

data_points = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        columns = line.strip().split('\t')
        
        if len(columns) == 3:
            domain, ner_tags, sentence = columns
            data_points.append((domain, ner_tags, sentence))

import pandas as pd
df = pd.DataFrame(data_points, columns=['domain', 'ner_tags', 'sentence'])

domain_counts = df['domain'].value_counts()

domain_counts_sorted = domain_counts.sort_values(ascending=False)

domain_list = list(domain_counts_sorted.items())


print("-"*50)

print("Domains and data point counts:")
for domain, count in domain_list:
    print(f"{domain}: {count} data points")

print("-"*50)

num_domains = len(domain_counts_sorted)
print(f"\nTotal number of domains: {num_domains}")

print("-"*50)

total_data_points = len(df)
print(f"Total number of data points: {total_data_points}")

print("-"*50)



# 1. Check for missing values
missing_values_count = df.isnull().sum().sum()
print(f"Total missing values in dataset: {missing_values_count}")


if missing_values_count > 0:
    df_before_cleaning = len(df)
    
    df.dropna(subset=['domain', 'sentence', 'ner_tags'], inplace=True)
    
    df_after_cleaning = len(df)
    missing_values_removed = df_before_cleaning - df_after_cleaning
    print(f"Data points removed due to missing values: {missing_values_removed}")
else:
    print("No missing values found. No rows were removed.")

print("-"*50)

# 2. Data Cleaning 

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


df['clean_sentence'] = df['sentence'].apply(clean_text)

print("-"*50)


# 3. Normalization 
df['clean_sentence'] = df['clean_sentence'].str.lower().str.strip()



print("-"*50)

print("\nBefore :")

print(df.head())

print("-"*50)

# 4. Deduplication
duplicates_before = df[df.duplicated(subset=['domain', 'clean_sentence', 'ner_tags'], keep=False)]

if len(duplicates_before) > 0:
    print("Duplicates found before removal:")
    print(duplicates_before)
    
    df.drop_duplicates(subset=['domain', 'clean_sentence', 'ner_tags'], inplace=True)
    
    duplicates_after = df.duplicated(subset=['domain', 'clean_sentence', 'ner_tags']).sum()
    
    duplicates_removed = len(duplicates_before) - duplicates_after
    print(f"Duplicates found and removed: {duplicates_removed}")
else:
    print("No duplicates found. No rows were removed.")

print("-"*50)

print("\nAfter :")

print(df.head())

total_data_points = len(df)
print(f"Total number of data points: {total_data_points}")

print("-"*50)

sentence_lengths = df['clean_sentence'].apply(lambda x: len(x.split()))
average_sentence_length = sentence_lengths.mean()
max_sentence_length = sentence_lengths.max()
min_sentence_length = sentence_lengths.min()

print(f'Average sentence length: {average_sentence_length}')
print(f'Maximum sentence length: {max_sentence_length}')
print(f'Minimum sentence length: {min_sentence_length}')


print("-"*50)

# 6. Split the dataset (90% train, 10% test)

X = df[['clean_sentence', 'sentence', 'ner_tags']] 
y = df['domain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_df = pd.DataFrame(X_train)
train_df['domain'] = y_train
train_file_path_tsv = os.path.join('..', 'data', 'cleaned_train_data_ml.tsv')
train_df.to_csv(train_file_path_tsv, sep='\t', index=False)

test_df = pd.DataFrame(X_test)
test_df['domain'] = y_test
test_file_path_tsv = os.path.join('..', 'data', 'cleaned_test_data_ml.tsv')
test_df.to_csv(test_file_path_tsv, sep='\t', index=False)


print(f"Training data saved to: {train_file_path_tsv}")
print(f"Test data saved to: {test_file_path_tsv}")