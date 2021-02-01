#Import required packages
import warnings
import string
import collections
import pandas as pd
import pyodbc
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

print("***Voice Mail Text Classifcation***")

warnings.filterwarnings('ignore')


def process_text(text, stem=True):
    """Tokenize text and stem words removing punctuation """
    text = text.translate(str.maketrans('','',string.punctuation))
    #print(text)
    tokens = word_tokenize(text)

    if stem:

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
#         lemmatizer = WordNetLemmatizer()
#         tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def vect_texts(call_texts):
    """Vectorize the texts and remove the stop words"""
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                     stop_words=stopwords.words('english'),
                                     max_df=0.5,
                                     min_df=0.1,
                                     lowercase=True,
                                ngram_range=(1, 2))

    tfidf_model = vectorizer.fit_transform(call_texts)
    return tfidf_model


#Commenting because server is down
"""print(pyodbc.drivers())
sqlConn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=xxx;DATABASE=xxx;UID=xxx;PWD=xxx')
query = "SELECT * FROM All_leads"
primary_df = pd.read_sql(query, sqlConn)
primary_df.head(5)"""


#Read "All Leads" Data
print("Reading primary data...")
primary_df = pd.read_excel("All Leads class.xlsx")
pr_df_proc = primary_df[primary_df['Description'].notna()]
pr_df_proc = pr_df_proc.reset_index()


def prepro(text):
    """Function to preprocess Primary data"""
    pro_text = text.strip()
    pro_text = pro_text.replace('\n','')
    pro_text = pro_text.replace('\r', '')
    pro_text = pro_text.replace('\t', '')
    return pro_text


pr_df_proc["Proc_text"] = pr_df_proc["Description"].apply(prepro)

print("Preprocessing and Extracting Features from Primary data...")
features = vect_texts(pr_df_proc["Proc_text"])

#Initialize list of models to be used for cross validation
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    SGDClassifier(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
print("Running 10 fold cross validation on multiple models for primary data...")
CV = 10        #10 fold cross validation
cv_df = pd.DataFrame(index=range(CV * len(models)))     #Intialiaze data frame to store results
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, pr_df_proc["Class"], scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


print("Model accuracy after cross validation for primary data")
print(cv_df.groupby('model_name').accuracy.mean())
print("--------------------------------")


def get_text(text):
    """Function to preprocess secondary data"""
    pro_text = text.split("Name/Reason:")[1]
    pro_text = text.strip()
    pro_text = pro_text.replace('\n','')
    pro_text =pro_text.replace('\r', '')
    pro_text =pro_text.replace('\t', '')
    return pro_text


#Read secondary data
print("Reading secondary data...")

#Commenting because server is down
"""print(pyodbc.drivers())
query = "SELECT * FROM activities_min"
sec_df = pd.read_sql(query, sqlConn)
sec_df.head(5)"""

sec_df = pd.read_csv("activities-min.csv")
sec_df_call = sec_df[sec_df["Deal"].isna()]
sec_df_call = sec_df_call[sec_df_call["Class"].notna()]
sec_df_filter = sec_df_call[sec_df_call["Subject"].str.startswith("Incoming Message for:")]
sec_df_filter["Proc_text"] = sec_df_filter.Subject.apply(get_text)
sec_df_final = sec_df_filter[["Proc_text", "Class"]]
sec_df_final.reset_index().drop("index", axis=1)



df_final = pr_df_proc[["Proc_text", "Class"]]



print("Appending Primary and Secondary data...")
final_df = df_final.append(sec_df_final, ignore_index = True)


print("Preprocessing and Extracting Features from Full data...")
full_features = vect_texts(final_df["Proc_text"])


print("Running 10 fold cross validation on multiple models for full data...")
CV = 10
cv_df_full = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, full_features, final_df["Class"], scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df_full = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])




print("Model accuracy after appending secondary data")
print(cv_df_full.groupby('model_name').accuracy.mean())

print("--------------------------------")


#Split the data randomly for training and testing
print("Training models individually...")
print("Spilting Train and Test data...")
X_train, X_test, y_train, y_test = train_test_split(full_features, final_df["Class"], test_size=0.33, random_state=101)

print("Shape of train data")
print(X_train.shape)
print("Shape of test data")
print(X_test.shape)
print("--------------------")

#Multinomial NB classifier
MNB_model = MultinomialNB()
print("Training Multunomial NB regresssion model...")
MNB_model.fit(X_train, y_train)
pred = MNB_model.predict(X_test)


test_df = pd.DataFrame(y_test)
actual_df = test_df.join(final_df["Proc_text"])
actual_df.reset_index(inplace=True)
actual_df.drop('index', axis=1, inplace=True)
actual_df = actual_df[['Proc_text', 'Class']]

result_df = actual_df.join(pd.DataFrame(pred))
result_df["Match"] = result_df['Class'] == result_df[0]
print("Prediction results with Multinomial NB")
print(result_df.head(5))

print("Save prediction results to 'test.csv'")
result_df.to_csv("test.csv")

#print(confusion_matrix(y_test, pred))
print("Classifiaction report for Multinomial NB")
print(classification_report(y_test, pred))
print("--------------------")

#Multiclass logistic regression
lr_model = LogisticRegression()
print("Training Logistic regresssion model...")
lr_model.fit(X_train, y_train)
pred = lr_model.predict(X_test)

#print(confusion_matrix(y_test, pred))
print("Classification report for Logistic regression")
print(classification_report(y_test, pred))
print("--------------------")

#Randomforest classifier
rf_model = RandomForestClassifier(n_estimators=200,max_depth=3, random_state=0)
print("Training Randomforest classifier...")
rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)


#print(confusion_matrix(y_test, pred))
print("Classification report for Randomforest regression")
print(classification_report(y_test, pred))
print("--------------------")

#Linear SVC
svc_model = LinearSVC()
print("Training Linear Support Vector Classifier...")
svc_model.fit(X_train, y_train)
pred = svc_model.predict(X_test)

#print(confusion_matrix(y_test, pred))
print("Classification report for Linear SVC")
print(classification_report(y_test, pred))
print("--------------------")

print("***Thank you***")

input()
