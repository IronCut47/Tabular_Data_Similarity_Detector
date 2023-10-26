import numpy as np
import os
import tabula
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import sklearn 
import sklearn.feature_extraction.text
import nltk # pip intall nltk (next)--> nltk.download('punkt')
from datasketch import MinHash, MinHashLSH # pip install datasketch
import torch # pip install torch
from transformers import AutoModel, AutoTokenizer # pip install transformers
from sklearn.metrics.pairwise import cosine_similarity

# Flask
app = Flask(__name__)
UPLOAD_FOLDER = (r'D:/Git projects/Tabular/PDF')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, UPLOAD_FOLDER)
content = ''

similarity_scores2 = {
                    "Cosine Similarity": 0,
                    "Longest Common Subsequence": 0,
                    "Smith Waterman": 0,
                    "Needleman Wunsch": 0,
                    "Bert Similarity": 0
                }

# Really good for this application as it maintains structural information
def cosine_similarity_score(document1, document2):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer() # Creates TF-IDF vectorizer
    vectorizer.fit([document1, document2])
    vectors = vectorizer.transform([document1, document2]).toarray() # TF-IDF ---> np.arry

    cosine_similarity = np.dot(vectors[0], vectors[1]) / (
        np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]) # cosine_similarity = (A dot B) / (||A|| * ||B||)
    )

    similarity_scores2['Cosine Similarity'] = cosine_similarity
    return cosine_similarity

# Counts the number of common elements in order between 2 documents
def longest_common_subsequence(document1, document2):
    m, n = len(document1), len(document2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if document1[i - 1] == document2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    similarity_scores2['Longest Common Subsequence'] = dp[m][n]
    return dp[m][n]

# It finds similar regions within documents by maintaining a score which is increase/decreased based on whether the character are similar/disimilar  
def smith_waterman(document1, document2, match=2, mismatch=-1, gap=-1):
    m, n = len(document1), len(document2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if document1[i - 1] == document2[j - 1]:
                dp[i][j] = max(
                    dp[i - 1][j - 1] + match, dp[i - 1][j] + gap, dp[i][j - 1] + gap
                )
            else:
                dp[i][j] = max(
                    dp[i - 1][j - 1] + mismatch, dp[i - 1][j] + gap, dp[i][j - 1] + gap
                )
    similarity_scores2['Smith Waterman'] = dp[m][n]
    return dp[m][n]

# Creates a table/grid row(1st sentence) & column(2nd sentence) and compares them by maintaining score which is increased/decreased based on whether the words/character match or not
def needleman_wunsch(doc1, doc2, match_score=2, mismatch_score=-1, gap_penalty=-2):
    rows, cols = len(doc1) + 1, len(doc2) + 1
    score_matrix = np.zeros((rows, cols))

    for i in range(rows):
        score_matrix[i][0] = i * gap_penalty
    for j in range(cols):
        score_matrix[0][j] = j * gap_penalty

    for i in range(1, rows):
        for j in range(1, cols):
            if doc1[i - 1] == doc2[j - 1]:
                match = score_matrix[i - 1][j - 1] + match_score
            else:
                match = score_matrix[i - 1][j - 1] + mismatch_score

            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty

            score_matrix[i][j] = max(match, delete, insert)
    similarity_scores2['Needleman Wunsch'] = score_matrix[rows - 1][cols - 1]
    return score_matrix[rows - 1][cols - 1]

# Bert model converts documents into fix sized vector --> calculates Cosine similarity  
def bert(document1, document2):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens1 = tokenizer(document1, return_tensors="pt", padding=True, truncation=True)
    tokens2 = tokenizer(document2, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

    similarity_scores2['Bert Similarity'] = cosine_similarity(embeddings1, embeddings2)
    return cosine_similarity(embeddings1, embeddings2)

def similarity_checker(doc1, doc2):
    cosine_similarity_score(doc1, doc2)
    longest_common_subsequence(doc1, doc2)
    smith_waterman(doc1, doc2)
    needleman_wunsch(doc1, doc2)
    bert(doc1, doc2)

def Tabular_Extractor(pdf_path, output_folder):
    df = tabula.read_pdf(pdf_path, stream = True, multiple_tables = True, pages = 'all')

    for i, table in enumerate(df):
        table_string = str(table)
        table_string = table_string.strip('[]').split('\n')  # Removing '[]' and splitting into lines
        table_string = '\n'.join(table_string)  # List to single string
        lines = table_string.strip().split('\n')

        header = lines[0].split()
        data = [line[1:].split() for line in lines[1:]]
        newdf = pd.DataFrame(data, columns=header)

        # Construct the file path for the TXT file in the output folder
        txt_file_path = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(pdf_path))[0]}.txt')

        # Save the extracted table as a TXT file
        with open(txt_file_path, 'w') as f:
            f.write(newdf.to_csv(sep='\t', index=False))

    return txt_file_path


@app.route('/', methods = ['POST', 'GET'])
def main():
    upload = False
    # scores = {}
    if request.method == 'POST':
        document1 = request.files["doc1"]
        document2 = request.files["doc2"]
    
        if document1.filename == "" or document2.filename == "":
            return render_template("index.html", message=0)
        else:
            filename_1 = secure_filename(document1.filename)
            document1_path = os.path.join(app.config["UPLOAD_FOLDER"], filename_1)
            document1.save(document1_path)

            filename_2 = secure_filename(document2.filename)
            document2_path = os.path.join(app.config["UPLOAD_FOLDER"], filename_2)
            document2.save(document2_path)

            output_folder = 'D:/Sem VII/NLP/Mini Project/TXT'

            txt_file_path1 = Tabular_Extractor(document1_path, output_folder)
            txt_file_path2 = Tabular_Extractor(document2_path, output_folder)

            # similarity_score1 = 20
            similarity_checker(
                txt_file_path1, txt_file_path2
            )

            return render_template(
            'testindex.html', scores = similarity_scores2
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True, port = 5000)