from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os   # Changing the directory where our data is stored
os.chdir('G:\datascience&machinelearning_bootcamp_Data\SpamData\SpamData')

stream = open('Processing\practice_email.txt', encoding='latin-1')
message = stream.read()
stream.close()
#print(message)

# Extracting the message body
stream = open('Processing\practice_email.txt', encoding='latin-1')

is_body = False
lines = []
for line in stream:
    if is_body:
        lines.append(line)
    elif line == '\n':
        is_body = True
stream.close()

email_body = '\n'.join(lines)
#print(email_body)

# Extracting Bodies From All Emails
def email_body_extractor(path):
    for root, dirname, filenames  in os.walk(path):
       for file_name in filenames:

           filepath = join(root, file_name)
           stream = open(filepath, encoding='latin-1')

           is_body = False
           lines = []
           for line in stream:
               if is_body:
                   lines.append(line)
               elif line == '\n':
                   is_body = True
           stream.close()

           email_body = '\n'.join(lines)
           yield file_name, email_body


def dataframe_from_directory(path, classification):
    rows = []
    row_names = []

    for file_name, email_body in email_body_extractor(path):
        rows.append({'MESSAGE': email_body, 'CATEGORY': classification})
        row_names.append(file_name)
    return pd.DataFrame(rows, index=row_names)

SPAM_1_PATH = 'Processing\spam_assassin_corpus\spam_1'
SPAM_2_PATH = 'Processing\spam_assassin_corpus\spam_2'
EASY_NONSPAM_1_PATH = 'Processing\spam_assassin_corpus\easy_ham_1'
EASY_NONSPAM_2_PATH = 'Processing\spam_assassin_corpus\easy_ham_2'

spam_emails = dataframe_from_directory(SPAM_1_PATH, 1)
spam_emails = spam_emails.append(dataframe_from_directory(SPAM_2_PATH, 1))
print(spam_emails.shape)
nonspam_emails = dataframe_from_directory(EASY_NONSPAM_1_PATH, 0)
nonspam_emails = nonspam_emails.append(dataframe_from_directory(EASY_NONSPAM_2_PATH, 0))

data = pd.concat([spam_emails, nonspam_emails])


# Data Cleaning Checking For Missing Values

# Check if any message bodies are null
print(data.MESSAGE.isnull().values.any())
# Check For Empty Email strlen = 0
print((data.MESSAGE.str.len() == 0).any())  # Returns true that is there are empty emails
print((data.MESSAGE.str.len() == 0).sum())  # Number Of empty emails
# Locate Empty Emails
print(data[data.MESSAGE.str.len() == 0].index)  # cmds are the system files included in extracted folder
# Dropping the empty emails
data.drop(['cmds'], inplace=True)   # inplace = true means data = data.drop(['cmds']) appending result with data
print(data[data.MESSAGE.str.len() == 0].index)


# Creating Document IDS for easy tracking
document_ids = range(0, len(data.index))
data['DOC_ID'] = document_ids    #  Adding Column To data

data['FILE_NAME'] = data.index
data = data.set_index('DOC_ID')

'''
# SAVE THE DATAFRAME IN OUR DISK SO THAT WE DONT HAVE TO DO THE BEFORE DATA CLEANING AGAIN
DATA_JSON_FILE = 'G:\datascience&machinelearning_bootcamp_Data\SpamData\SpamData\Processing\my-email-text-data.json'
data.to_json(DATA_JSON_FILE)
'''

# DATA VISUALISATION USING PIE CHARTS
print(data.CATEGORY.value_counts())
amount_of_spam = data.CATEGORY.value_counts()[1]
amount_of_nonspam = data.CATEGORY.value_counts()[0]

category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_nonspam]
plt.pie(sizes, labels=category_names, autopct='%1.0f%%', textprops={'fontsize': 16},explode=[0, 0.1])
plt.show()

# Donut Pie Chart
category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_nonspam]
plt.pie(sizes, labels=category_names, autopct='%1.0f%%', textprops={'fontsize': 16}, pctdistance=0.8,explode=[0, 0.05])
centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')
plt.gca().add_artist(centre_circle)
plt.show()


# Using NLP to modify our data : converting to lowercase,Tokenising,Removing stop words,Stripping out HTML tags,Word Streaming,Removing punctuation

# Download NLTK Resources
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')      # C:\Users\shubh\AppData\Roaming\nltk_data
nltk.download('stopwords')  # stopwords = words that are common and without meaning
stop_words = set(stopwords.words('english'))
# print(stop_words)
stemmer = PorterStemmer()  # Word streaming = converting word to their base words   SnowballSteamer = stemmer for other languages SnowballStemmer('english')

def clean_message(message):
  filtered_words = []
  soup = BeautifulSoup(message, 'html.parser')  # Setting soup to html parser
  cleaned_text = soup.get_text()        # Removing HTML tags
  words = word_tokenize(cleaned_text.lower())     # Converting to lowercase

  for word in words:
      if word not in stop_words and word.isalpha():  # Removing Stopwords And Punctuations
        filtered_words.append(stemmer.stem(word))
  return filtered_words


# Cleaning And Tokenisation to all emails
nested_list = data.MESSAGE.apply(clean_message)  # apply to all emails
#print(nested_list)

# Creating lists from nested_list for spams and non-spams
docs_ids_spam = data[data.CATEGORY == 1].index
docs_ids_nonspam = data[data.CATEGORY == 0].index
nested_list_spam = nested_list.loc[docs_ids_spam]
nested_list_nonspam = nested_list.loc[docs_ids_nonspam]

# Generate Vocabulary and Dictionary For most Frequently used words
stemmed_nested_list = data.MESSAGE.apply(clean_message)
flat_stemmed_list = [item for sublist in stemmed_nested_list for item in sublist]
unique_words = pd.Series(flat_stemmed_list).value_counts()
#print('No Of Unique Words: ',unique_words.shape[0])
#print(unique_words)

# Creating A subset of 2500 most common words
frequent_words = unique_words[0:2500]
#print(frequent_words.head())

# Create A Vocabulary dataframe with word_id 0-2499
words_ids = list(range(0, 2500))
vocab = pd.DataFrame({'VOCAB_WORD':frequent_words.index.values}, index=words_ids)
vocab.index.name = 'WORD_ID'
# SAVE FILE TO OUR DISK
WORDID_FILE = 'G:\datascience&machinelearning_bootcamp_Data\SpamData\SpamData\Processing\my-word-by-id.csv'
vocab.to_csv(WORDID_FILE, index_label=vocab.index.name, header=vocab.VOCAB_WORD.name)

# Generate features and sparse matrix (matrix with words and its occurences in an email)
word_columns_df = pd.DataFrame.from_records(stemmed_nested_list.tolist())
print(word_columns_df.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY, test_size=0.3, random_state=42)
X_train.index.name = X_test.index.name = 'DOC_ID'

# sparse matrix for training data
word_index = pd.Index(vocab.VOCAB_WORD)
def make_sparse_matrix(df, indexed_words, labels):

    no_of_rows = df.shape[0]
    no_of_columns = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []

    for i in range(no_of_rows):
        for j in range(no_of_columns):
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]

                item = {'LABEL':category, 'DOC_ID': doc_id, 'OCCURENCE': 1, 'WORD_ID': word_id}
                dict_list.append(item)
    return pd.DataFrame(dict_list)


sparse_train_df = make_sparse_matrix(X_train, word_index, y_train)
# Combine occurences with pandas groupby() method
train_grouped = sparse_train_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
train_grouped = train_grouped.reset_index()

# Save Training Data File
TRAIN_FILE = 'G:\datascience&machinelearning_bootcamp_Data\SpamData\SpamData\Training\Train-data.txt'
np.savetxt(TRAIN_FILE, train_grouped, fmt='%d')

# Save Test Data File
sparse_test_df = make_sparse_matrix(X_test, word_index, y_test)
# Combine occurences with pandas groupby() method
test_grouped = sparse_test_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
test_grouped = test_grouped.reset_index()
# Saving the File
TEST_FILE = 'G:\datascience&machinelearning_bootcamp_Data\SpamData\SpamData\Testing\Test-data.txt'
np.savetxt(TEST_FILE, test_grouped, fmt='%d')



# Training Our Model

sparse_train_data = np.loadtxt(TRAIN_FILE, delimiter=' ', dtype=int)
sparse_test_data = np.loadtxt(TEST_FILE, delimiter=' ', dtype=int)

# Creating a Full Matrix
def make_full_matrix(sparse_matrix, no_of_words, doc_idx = 0, word_idx=1, cat_idx=2, freq_idx = 3):
  
    column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, 2500))
    index_names = np.unique(sparse_matrix[:, 0])
    full_matrix = pd.DataFrame(index=index_names, columns=column_names)
    full_matrix.fillna(value=0, inplace=True)

    for i in range(sparse_matrix.shape[0]):
        doc_no = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurence = sparse_matrix[i][freq_idx]

        full_matrix.at[doc_no, 'DOC_ID'] = doc_no
        full_matrix.at[doc_no, 'CATEGORY'] = label
        full_matrix.at[doc_no, word_id] = occurence

    full_matrix.set_index('DOC_ID', inplace=True)
    return full_matrix


full_train_data = make_full_matrix(sparse_train_data, 2500)

# Training the naive bayes model

# Gathering Required Information
prob_spam = full_train_data.CATEGORY.sum() / full_train_data.CATEGORY.size  # probability of spam emails
full_train_features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
email_lengths = full_train_features.sum(axis=1)     # Total no of words per email
total_words = email_lengths.sum()               # Total(Overall) words
spam_lengths = email_lengths[full_train_data.CATEGORY == 1]  # no of spam emails
spam_words = spam_lengths.sum()  # Total no of words in spams
nonspam_lengths = email_lengths[full_train_data.CATEGORY == 0]  # no of nonspam emails
nonspam_words = nonspam_lengths.sum()  # Total no of words in nonspams

# Summing the tokens Occuring in Spam
train_spam_tokens = full_train_features.loc[full_train_data.CATEGORY == 1]
summed_spam_tokens = train_spam_tokens.sum(axis=0) + 1          # +1 = Laplace Smoothing avoiding 0/0
# Summing the tokens Occuring in NonSpam
train_nonspam_tokens = full_train_features.loc[full_train_data.CATEGORY == 0]
summed_nonspam_tokens = train_nonspam_tokens.sum(axis=0) + 1

# P(Token | Spam) = Probability that a token occurs given email is spam
prob_tokens_spam = summed_spam_tokens / (spam_words + 2500)
# P(Token | NonSpam) = Probability that a token occurs given email is nonspam
prob_tokens_nonspam = summed_nonspam_tokens / (nonspam_words + 2500)
# P(Token) = Probability that a token occurs
prob_tokens_all = full_train_features.sum(axis=0) / total_words


# Prepare Test Data
full_test_data = make_full_matrix(sparse_test_data, 2500)
X_test = full_test_data.loc[:, full_test_data.columns != 'CATEGORY']
y_test = full_test_data.CATEGORY

# P(spam | X) = { P(X | Spam) * P(Spam) } / P(X)
#             = log(P(X | Spam)) - log(P(X)) + log(P(Spam))

joint_log_spam = X_test.dot(np.log(prob_tokens_spam) - np.log(prob_tokens_all)) + np.log(prob_spam)  # joint probabilities that emails are spams
joint_log_nonspam = X_test.dot(np.log(prob_tokens_nonspam) - np.log(prob_tokens_all)) + np.log(1 - prob_spam)

# Making Predictions = Checking for higher joint probability
prediction = joint_log_spam > joint_log_nonspam

# Metrics And Evaluation
correct_docs = (y_test == prediction).sum()
print('Correct Predictions ', correct_docs)
accuracy = correct_docs/len(X_test)
print('Accuracy: ', accuracy)

# Visualising the results

# chart styling info
yaxis_label = 'P(X | Spam)'
xaxis_label = 'P(X | NonSpam)'
linedata = np.linespace(start=-14000, stop=1, num=1000)

plt.xlim([-14000, 1])
plt.ylim([-14000, 1])
plt.xlabel(xaxis_label, fontsize=16)
plt.ylabel(yaxis_label, fontsize=16)
plt.scatter(joint_log_nonspam, joint_log_spam, color='navy')
plt.show()

# Plotting Decision Boundary
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])
plt.xlabel(xaxis_label, fontsize=16)
plt.ylabel(yaxis_label, fontsize=16)
plt.scatter(joint_log_nonspam, joint_log_spam, color='navy', alpha=0.5, s=35)
plt.plot(linedata, linedata, color='orange')
plt.show()

# Subplot for better understanding

# Chart no 1
plt.subplot(1, 2, 1)
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])
plt.xlabel(xaxis_label, fontsize=16)
plt.ylabel(yaxis_label, fontsize=16)
plt.scatter(joint_log_nonspam, joint_log_spam, color='navy', alpha=0.5, s=35)
plt.plot(linedata, linedata, color='orange')
# Chart No 2
plt.subplot(1, 2, 2)
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])
plt.xlabel(xaxis_label, fontsize=16)
plt.ylabel(yaxis_label, fontsize=16)
plt.scatter(joint_log_nonspam, joint_log_spam, color='navy', alpha=0.5, s=10)
plt.plot(linedata, linedata, color='orange')
plt.show()


# using seaborn for more graph abilities
sns.set_style('whitegrid')
labels = 'Actual Category'
summary_df = pd.DataFrame({yaxis_label: joint_log_spam, xaxis_label: joint_log_nonspam, labels: y_test})
sns.lmplot(x=xaxis_label, y=yaxis_label, data=summary_df, fit_reg=False, scatter_kws={'alpha': 0.5, 's': 25}, hue=labels, markers=['o', 'x'], palette='hls', legend=False)
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])
plt.plot(linedata, linedata, color='black')
plt.legend(('Decision Boundary', 'NonSpam', 'Spam'), loc='lower right', fontsize=16)
sns.plt.show()