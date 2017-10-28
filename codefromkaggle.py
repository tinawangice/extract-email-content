import os, sys, email,re
import numpy as np
import pandas as pd
import csv



emails_df = pd.read_csv('/Users/ting/Desktop/NLP_Email/emails.csv')
print(emails_df.shape)
print(emails_df.head())
print(emails_df.iloc[0,:])

## Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs

# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)
# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]
# Parse content from emails
emails_df['content'] = list(map(get_text_from_email, messages))
# Split multiple email addresses
emails_df['From'] = emails_df['From'].map(split_email_addresses)
emails_df['To'] = emails_df['To'].map(split_email_addresses)

# Extract the root of 'file' as 'user'
emails_df['user'] = emails_df['file'].map(lambda x:x.split('/')[0])

del messages
print(emails_df.head())

emails_df = emails_df.set_index('Message-ID')\
    .drop(['file', 'Mime-Version', 'Content-Type', 'Content-Transfer-Encoding'], axis=1)
# Parse datetime
emails_df['Date'] = pd.to_datetime(emails_df['Date'], infer_datetime_format=True)


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer

def clean(text):
    stop = set(stopwords.words('english'))
    # add elements to set
    stop.update(("to", "cc", "subject", "http", "from", "sent",
                 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    porter = PorterStemmer()
    #rstrip() returns a copy of the string in which all chars have been stripped from the end of the string (default whitespace characters).
    text = text.rstrip()
    #substitude non-letters to letters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if ((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    # stem = " ".join(porter.stem(token) for token in normalized.split())

    return normalized

def generate_wordlist():
    for text in emails_df['content']:
        yield clean(text) + "\n"

with open('/Users/ting/Desktop/NLP_Email/wordlist.dat','w') as f:
    f.writelines(generate_wordlist())

def generate_email_body():
    for text in emails_df['content']:
        text = text.rstrip()
        text = re.sub(r'[^a-zA-Z\.,\?!"\']', ' ', text)
        yield text + "\n"

with open('/Users/ting/Desktop/NLP_Email/emailbody.dat','w') as f:
    f.writelines(generate_email_body())



