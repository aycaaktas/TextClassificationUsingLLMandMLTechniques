import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_ollama.llms import OllamaLLM
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import time

'''''
class PromptStrategy:
    def __init__(self, categories):
        self.categories = categories

    def get_extract_message(self, sentence, ner_tags):
        ner_dict = format_ner_tags(sentence, ner_tags)
        extract_prompt = f"""
        The sentence: "{sentence}" contains the following NER tags: {ner_dict}.
        Based on this information, extract the key entities or elements from the sentence.
        """
        return extract_prompt

    def get_hypothesize_message(self, extracted_response, ner_dict):
        hypothesize_prompt = f"""
        Based on the extracted information: "{extracted_response}" and the NER tags {ner_dict},
        hypothesize which category this sentence belongs to. The categories are: {', '.join(self.categories)}.
        """
        return hypothesize_prompt

    def get_classify_message(self, sentence, hypothesized_response):
        classify_prompt = f"""
        Given the hypothesis that the sentence: "{sentence}" fits the category "{hypothesized_response}",
        confirm if the sentence should be classified under this category. If not, provide the correct category.
        """
        return classify_prompt
'''''

def format_ner_tags(sentence, ner_tags):
    tokens = sentence.split()  
    ner_dict = {token: tag for token, tag in zip(tokens, ner_tags) if tag != 'O'}
    return ner_dict




class PromptStrategy:
    def __init__(self, categories):
        self.categories = categories

    def get_direct_classify_message(self, sentence, ner_tags):
        ner_dict = format_ner_tags(sentence, ner_tags)
        classify_prompt = f"""
        You are an expert in text classification. You must strictly choose one category from the list of categories provided below.
        If none of the categories match perfectly, choose the category that is the closest in meaning or concept.
        Do not invent new categories that are not listed. Only respond with a single category from the provided list.
        Sentence: "{sentence}"
        NER Tags: {ner_dict}
        Categories: {', '.join(self.categories)}
        """
        return classify_prompt


'''''
#Classify using the model with EHC prompt strategy
def classify_text_with_ehc_prompt(model, sentence, ner_tags, prompt_strategy, max_retries=3):
    ner_dict = format_ner_tags(sentence, ner_tags)
    
    for attempt in range(max_retries):
        try:
            # Step 1: Extract relevant information
            extract_prompt = prompt_strategy.get_extract_message(sentence, ner_tags)
            extracted_response = model(extract_prompt).strip()

            # Step 2: Hypothesize the category using extracted information
            hypothesize_prompt = prompt_strategy.get_hypothesize_message(extracted_response, ner_dict)
            hypothesized_response = model(hypothesize_prompt).strip()

            # Step 3: Classify and confirm the final category
            classify_prompt = prompt_strategy.get_classify_message(sentence, hypothesized_response)
            final_response = model(classify_prompt).strip()

            return final_response

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for sentence: {sentence}. Error: {e}")
            time.sleep(5)

    print(f"Failed to classify sentence: '{sentence}' after {max_retries} attempts.")
    return None

'''''


# Classify using the model with Direct prompt strategy
def classify_text_with_direct_prompt(model, sentence, ner_tags, prompt_strategy,categories, max_retries=3):
    valid_categories = set(categories)
    for attempt in range(max_retries):
        try:
            classify_prompt = prompt_strategy.get_direct_classify_message(sentence, ner_tags)
            final_response = model.invoke(classify_prompt).strip()

            for category in valid_categories:
                if category.lower() in final_response.lower():  
                    return category
            
            print(f"Invalid prediction: '{final_response}'. Mapping to 'unknown'.")
            return 'unknown'
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for sentence: {sentence}. Error: {e}")
            time.sleep(5)

    print(f"Failed to classify sentence: '{sentence}' after {max_retries} attempts.")
    return None





def evaluate_model_with_prompt(model, dataset, categories, result_file):

    true_labels = []
    predicted_labels = []
    prompt_strategy = PromptStrategy(categories)

    categories_with_unknown = categories + ['unknown']
    failure_count = 0
    invalid_pred=0

    for _, example in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        sentence = example['sentence']
        ner_tags = example['ner_tags']
        true_label = example['domain']

        predicted_label = classify_text_with_direct_prompt(model, sentence, ner_tags, prompt_strategy, categories)

        if predicted_label is None:
            failure_count += 1
            print(f"Failed classification. Mapping sentence '{sentence}' to 'unknown'.' ")
            predicted_label = 'unknown'
        elif predicted_label == 'unknown':
            print(f"The correct label is:'{true_label}")
            invalid_pred +=1


        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    print(f"Total failures: {failure_count}")
    print(f"Total invalid predictions: {invalid_pred}")

    

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        accuracy = accuracy_score(true_labels, predicted_labels)
        balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        cm = confusion_matrix(true_labels, predicted_labels)
        class_report = classification_report(true_labels, predicted_labels, target_names=categories_with_unknown)

    with open(result_file, 'w') as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Balanced Accuracy: {balanced_acc * 100:.2f}%\n")
        f.write(f"Test Precision: {precision * 100:.2f}%\n")
        f.write(f"Test Recall: {recall * 100:.2f}%\n")
        f.write(f"Test F1 Score: {f1 * 100:.2f}%\n")
        f.write("\nClass-wise Report:\n")
        f.write(class_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    metrics = {'Accuracy': accuracy, 'Balanced Accuracy': balanced_acc, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), [v * 100 for v in metrics.values()], color=['blue', 'purple', 'red', 'green', 'orange'])
    plt.ylabel('Percentage (%)')
    plt.title('Evaluation Metrics')
    plt.savefig('../results/llm/evaluation_metrics_llm.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2%', cmap='Blues', xticklabels=categories_with_unknown, yticklabels=categories_with_unknown)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('../results/llm/confusion_matrix_llm.png')
    plt.show()

    return accuracy, balanced_acc, precision, recall, f1, class_report, cm

test_data_path = 'data/cleaned_test_data.tsv'
test_df = pd.read_csv(test_data_path, sep='\t')
test_df = test_df[['sentence', 'ner_tags', 'domain']]

categories = test_df['domain'].unique().tolist()
print(f"Categories extracted from the dataset: {categories}")

llm = OllamaLLM(model="llama3:8b")


result_file = "../results/llm/evaluation_results_llm.txt"

test_accuracy, balanced_acc, test_precision, test_recall, test_f1, class_report, cm = evaluate_model_with_prompt(llm, test_df, categories, result_file)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
print(f"Test Precision: {test_precision * 100:.2f}%")
print(f"Test Recall: {test_recall * 100:.2f}%")
print(f"Test F1 Score: {test_f1 * 100:.2f}%")

print("\nClass-wise Report:\n", class_report)

print("Confusion Matrix:\n", cm)

