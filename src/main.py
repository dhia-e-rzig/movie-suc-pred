import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from pca import run_pca
from knn import run_knn, plot_data
from random_forest import run_random_forest
from logistic import run_logistic_regression
from server import (
    run,
    _classifier,
    set_directors,
    set_actors,
    set_coutries,
    set_languages,
    set_content_rating,
)
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def fill_nan(df_movie):
    return df_movie.fillna(df_movie.median())


def data_prepocessing():
    df = pd.read_csv(
        "../data/imdb.csv", error_bad_lines=False, quoting=csv.QUOTE_MINIMAL
    )
    # df = df[df['year'] > 2000]
    df_movie = df
    df_movie = fill_nan(df_movie)

    # deleting the useless attributes that won't contribute with the actual fitting of the model.
    df_movie = df_movie.drop(columns="type")
    df_movie = df_movie.drop(columns="fn")
    df_movie = df_movie.drop(columns="tid")
    df_movie = df_movie.drop(columns="url")
    df_movie = df_movie.drop(columns="wordsInTitle")
    df_movie = df_movie.drop(columns="title")

    # apply standard scaling to treat changes in features equally
    # sc = StandardScaler()
    # apply Min Max to treat changes in features equally
    ms = MinMaxScaler()
    scaled_features = ms.fit_transform(df_movie.values)
    df_scaled_features = pd.DataFrame(scaled_features, columns=df_movie.columns).fillna(
        df_movie.mean()
    )
    df_standard = df_movie[list(df_movie.describe().columns)]
    return (df_scaled_features, df_standard)


def classify(row):
    if row["imdbRating"] >= 0 and row["imdbRating"] < 0.6:
        return "BAD"
    elif row["imdbRating"] >= 0.6 and row["imdbRating"] < 0.8:
        return "GOOD"
    elif row["imdbRating"] >= 0.8 and row["imdbRating"] <= 1:
        return "AMAZING"


def split_genres(row):
    return str(row["genres"]).replace("|", " ")


def set_class(row):
    if row["imdb_score"] >= 0 and row["imdb_score"] < 4:
        return 0
    elif row["imdb_score"] >= 4 and row["imdb_score"] < 7:
        return 1
    elif row["imdb_score"] >= 7 and row["imdb_score"] <= 10:
        return 2


def load_metadata_dataset():
    df_metadata = pd.read_csv(
        "../data/movie_metadata.csv", error_bad_lines=False, quoting=csv.QUOTE_MINIMAL
    )

    # startoff by removing useless attributes that won't help with the training of the model(aspect_ratio...)
    df_metadata = df_metadata.drop(columns="aspect_ratio")
    df_metadata = df_metadata.drop(columns="movie_imdb_link")
    df_metadata = df_metadata.drop(columns="color")

    # df_metadata = df_metadata.drop(columns="director_name")
    # df_metadata = df_metadata.drop(columns="actor_1_name")
    df_metadata = df_metadata.drop(columns="actor_2_name")
    df_metadata = df_metadata.drop(columns="actor_3_name")
    df_metadata = df_metadata.drop(columns="plot_keywords")
    df_metadata = df_metadata.drop(columns="movie_title")

    # remove the '|' seperator from the genres attribute
    df_metadata["genres"] = df_metadata.apply(split_genres, axis=1)

    # create an instance of the CountVectorizer to transform the genres attribute into a binary array
    vectorizer = CountVectorizer()
    count_genres = vectorizer.fit_transform(df_metadata["genres"])
    # print(count_genres.toarray())
    genres_feature_names = vectorizer.get_feature_names()
    dfGenres = pd.DataFrame(count_genres.toarray(), columns=genres_feature_names)

    # create a LabelEncoder instance to encode string values
    le = LabelEncoder()

    df_metadata = fill_nan(df_metadata)
    dircetor_names = list(df_metadata["director_name"])
    encoded_directors = le.fit_transform(dircetor_names)
    print(encoded_directors)

    df_metadata = df_metadata.drop(columns="director_name")
    df_metadata["director_name"] = encoded_directors

    set_directors(dircetor_names, encoded_directors)

    # use label encoding to encode first actor names
    df_metadata = fill_nan(df_metadata)
    actor_names = list(df_metadata["actor_1_name"])
    encoded_actor_names = le.fit_transform(actor_names)
    # print(encoded_actor_names)

    df_metadata = df_metadata.drop(columns="actor_1_name")
    df_metadata["actor_1_name"] = encoded_actor_names

    set_actors(actor_names, encoded_actor_names)

    # label encode countries and languages and content rating as well,
    languages = list(df_metadata["language"])
    encoded_languages = le.fit_transform(languages)

    df_metadata = df_metadata.drop(columns="language")
    df_metadata["language"] = encoded_languages
    set_languages(languages, encoded_languages)

    countries = list(df_metadata["country"])
    encoded_countries = le.fit_transform(countries)
    df_metadata = df_metadata.drop(columns="country")
    df_metadata["country"] = encoded_countries
    set_coutries(countries, encoded_countries)

    # do the same for the content rating
    content_rating = list(df_metadata["content_rating"])
    encoded_content_rating = le.fit_transform(content_rating)
    df_metadata = df_metadata.drop(columns="content_rating")
    df_metadata["content_rating"] = encoded_content_rating
    set_content_rating(content_rating, encoded_content_rating)

    # drop the rest of the movies metadata that we wont probably use.
    df_metadata = df_metadata.drop(columns="genres")
    # df_metadata = df_metadata.drop(columns="language")
    # df_metadata = df_metadata.drop(columns="country")
    # df_metadata = df_metadata.drop(columns="content_rating")

    frames = [df_metadata, dfGenres]
    df_movie = pd.concat(frames, ignore_index=False, sort=True)

    # add the class attribute
    df_movie["class"] = df_movie.apply(set_class, axis=1)

    df_movie = fill_nan(df_movie)
    df_movie = df_movie.drop(columns="imdb_score")

    print(df_movie["director_name"].head())

    col_mask = print(df_movie.isna().any(axis=0))
    print(col_mask)

    # return the processed dataset.
    return df_movie


if __name__ == "__main__":
    #     df_movie, df_standard = data_prepocessing()
    #     df_knn = df_movie
    #     df_knn = df_knn.reset_index()
    #     df_knn["class"] = df_knn.apply(classify, axis=1)
    #     classes = list(df_knn["class"])
    #     amazing = classes==['AMAZING']
    #     print(amazing)
    #     df_knn = df_knn.drop(columns="imdbRating")

    df_movie = load_metadata_dataset()
    #run_knn(df_movie)
    #run_logistic_regression(df_movie)
    classifier=run_random_forest(df_movie)

    run(classifier)

