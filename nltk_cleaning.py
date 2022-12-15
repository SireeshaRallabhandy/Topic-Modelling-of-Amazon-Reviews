import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from os import path
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

with open("content.txt", 'r') as file:
    content = file.read()

stop_words = set(stopwords.words("english"))

corpus = []

#Remove punctuations
text = re.sub('[^a-zA-Z]', ' ', content)

#Convert to lowercase
text = text.lower()

#remove tags
text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

# remove special characters and digits
text = re.sub("(\\d|\\W)+", " ", text)

##Convert to list from string
text = text.split()

##Stemming
ps = PorterStemmer()
#Lemmatisation
lem = WordNetLemmatizer()
text = [lem.lemmatize(word) for word in text if not word in
        stop_words]
text = " ".join(text)
corpus.append(text)

# print("stemming:", stem.stem(content))
# print("lemmatization:", lem.lemmatize(content, "v"))

# wordcloud = WordCloud(
#     background_color='white',
#     stopwords=stop_words,
#     max_words=100,
#     max_font_size=50,
#     random_state=42
# ).generate(str(corpus))
# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()
# fig.savefig("word1.png", dpi=900)


with open(r'C:\Python\Python Programming\amazon-review-scraper\new.txt', 'w') as fp:
    for item in corpus:
        # write each item on a new line
        fp.write("%s\n" % item)
