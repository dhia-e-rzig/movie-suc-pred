import asyncio
from aiohttp import web
import json
import numpy as np
import aiohttp_cors

_classifier = None
_directors = []
_actors = []
_countries = []
_languages = []
_content_rating = []

@asyncio.coroutine
def handler(request):
    return web.Response(
        text="Hello!",
        headers={
            "X-Custom-Server-Header": "Custom data",
        })



def predict(data):
    movie_frame = {
        "title": "Der Vagabund und das Kind (1921)",
        "wordsInTitle": "der vagabund und das kind",
        "imdbRating": 8.4,
        "ratingCount": 40550,
        "duration": 3240,
        "year": 1921,
        "nrOfWins": 1,
        "nrOfNominations": 0,
        "nrOfPhotos": 19,
        "nrOfNewsArticles": 96,
        "nrOfUserReviews": 85,
        "nrOfGenre": 3,
        "Action": 0,
        "Adult": 0,
        "Adventure": 0,
        "Animation": 0,
        "Biography": 0,
        "Comedy": 1,
        "Crime": 0,
        "Documentary": 0,
        "Drama": 1,
        "Family": 1,
        "Fantasy": 0,
        "FilmNoir": 0,
        "GameShow": 0,
        "History": 0,
        "Horror": 0,
        "Music": 0,
        "Musical": 0,
        "Mystery": 0,
        "News": 0,
        "RealityTV": 0,
        "Romance": 0,
        "SciFi": 0,
        "Short": 0,
        "Sport": 0,
        "TalkShow": 0,
        "Thriller": 0,
        "War": 0,
        "Western": 0,
    }

    movie_frame = [
        1,
        0,
        19,
        85,
        3,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    movie_frame = np.reshape(data, (1, -1))

    td = _classifier.predict(movie_frame)
    print("\n\n----------------------Prediction Test----------------------\n\n", td)
    return td


async def handle_directors(request):
    # return the encoded director names
    return web.json_response(_directors, status=200)


async def handle_actors(request):
    # return the encoded director names
    return web.json_response(_actors, status=200)


async def handle_countries(request):
    # return the encoded director names
    return web.json_response(_countries, status=200)


async def handle_languages(request):
    # return the encoded director names
    return web.json_response(_languages, status=200)


async def handle_content_rating(request):
    # return the encoded director names
    return web.json_response(_content_rating, status=200)


async def handle(request):
    request = await request.text()
    print(request)
    request_body = json.loads(request)
    movie_data = []

    movie_data.append(int(request_body["wins"]))
    movie_data.append(int(request_body["reviews"]))
    movie_data.append(int(request_body["photos"]))
    movie_data.append(len(request_body["genres"]))

    for i in range(0, 27):
        movie_data.append(0)

    for i in request_body["genres"]:
       movie_data[int(i) + 4] = 1

    movie_data.append(len(request_body["directors"]))

    for i in request_body["directors"]:
        movie_data.append(int(request_body["directors"][i]))
    movie_data.append(len(request_body["actors"]))
    for i in request_body["actors"]:
        movie_data.append(int(request_body["actors"][i]))
    movie_data.append(len(request_body["countries"]))
    for i in request_body["countries"]:
        movie_data.append( int(request_body["countries"][i]))
    movie_data.append(len(request_body["languages"]))
    for i in request_body["languages"]:
        movie_data.append(int(request_body["languages"][i]))
    movie_data.append(request_body["content_rating"])



    print(movie_data)
    response = predict(movie_data)
    return web.json_response(response.tolist(), status=200)


def set_directors(director_names, director_codes):

    for i in range(0, len(director_names)):
        director = {}
        director["id"] = int(director_codes[i])
        director["text"] = str(director_names[i])
        global _directors

        if director not in _directors:
            _directors.append(director)


def set_actors(actor_names, actor_codes):
    for i in range(0, len(actor_names)):
        actor = {}
        actor["id"] = int(actor_codes[i])
        actor["text"] = str(actor_names[i])
        global _actors

        if actor not in _actors:
            _actors.append(actor)


def set_coutries(country_names, country_codes):
    for i in range(0, len(country_names)):
        country = {}
        country["id"] = int(country_codes[i])
        country["text"] = str(country_names[i])
        global _actors

        if country not in _countries:
            _countries.append(country)


def set_languages(language_names, language_codes):
    for i in range(0, len(language_names)):
        language = {}
        language["id"] = int(language_codes[i])
        language["text"] = str(language_names[i])
        global _actors

        if language not in _languages:
            _languages.append(language)


def set_content_rating(content_ratings, rating_codes):
    for i in range(0, len(content_ratings)):
        content_rating = {}
        content_rating["id"] = int(rating_codes[i])
        content_rating["text"] = str(content_ratings[i])
        global _actors

        if content_rating not in _content_rating:
            _content_rating.append(content_rating)


def run(classifier):
    global _classifier
    _classifier = classifier
    app = web.Application()
    app.router.add_post("/predict", handle)
    app.router.add_get("/get-directors", handle_directors)
    app.router.add_get("/get-actors", handle_actors)
    app.router.add_get("/get-countries", handle_countries)
    app.router.add_get("/get-languages", handle_languages)
    app.router.add_get("/get-content-rating", handle_content_rating)


    cors = aiohttp_cors.setup(app, defaults={
        # Allow all to read all CORS-enabled resources from
        # http://client.example.org.
        "*": aiohttp_cors.ResourceOptions(),
    })
    resource = cors.add(app.router.add_resource("/predict"));
    route = cors.add(
        resource.add_route("POST", handler), {
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers=("X-Custom-Server-Header",),
                allow_headers=("X-Requested-With", "Content-Type"),
                max_age=3600,
            )
        })
    # cors.add(route, {
    #     "*":
    #         aiohttp_cors.ResourceOptions(allow_credentials=True),
    #     "http://127.0.0.1":
    #         aiohttp_cors.ResourceOptions(allow_credentials=True),
    # })
    web.run_app(app,host='127.0.0.1', port=8000)
