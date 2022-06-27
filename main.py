import numpy as np
import pandas as pd
import difflib #kullanıcının kelime hatası yaptığında düzelten kütüphane
from sklearn.feature_extraction.text import TfidfVectorizer #veri setinde olan text değerlerini mantıklı sayılara dönüştürüyor
from sklearn.metrics.pairwise import cosine_similarity #veriler arasında benzerlik skoru oluşturur

dataset = pd.read_csv("movies.csv")

print(dataset.shape) #(satır,sütun)

features = ["genres","keywords","tagline","cast","director"] #programa etki edecek özellikler


#boş değerlerin yerlerini doldurma
for feature in features:
    dataset[feature] = dataset[feature].fillna("")


#kullanılacak değerleri birleştirme
combine = dataset["genres"]+ " "+dataset["keywords"]+ " "+dataset["tagline"]


#text datasını feature vektörüne çevirme
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combine)


#benzerlik skorunu cosine_similarity ile alma
similarity= cosine_similarity(feature_vectors)


#kullanıcıdan film adı alma
movie_name = input("Hangi filme benzer yapımları arıyorsunuz: ")

#film adlarını içeren liste
movie_titles = dataset["title"].tolist()


#inputu film ismi ile eşleştirme
true_movie = difflib.get_close_matches(movie_name, movie_titles)
if true_movie == []:
    print("Eşleşme bulunamadı")
else:
    print(true_movie)
    a = int(input("Kaçıncı sıradaki filmi kastettiğinizi giriniz: "))


if a == 1:
    selected_movie = true_movie[0]
elif a ==2:
    selected_movie = true_movie[1]
elif a ==3:
    selected_movie = true_movie[2]
else:
    print("Yanlıs bir deger girdiniz.")

print(f"Seçilen film: {selected_movie}")



#Belirlenen filmin değerlerini bulma

movie_index = dataset[dataset.title == selected_movie]["index"].values[0]
print(movie_index)


#Benzerlik olan filmlerin listesi - Büyükten küçüğe sıralama
similarity_score = list(enumerate(similarity[movie_index]))

sorted_movies = sorted(similarity_score, key= lambda x:x[1], reverse=True)


i = 1
for movie in sorted_movies:
    index = movie[0]
    title_movie = dataset[dataset.index==index]["title"].values[0]
    if (i<11):
        print(i, ".",title_movie)
        i+=1



