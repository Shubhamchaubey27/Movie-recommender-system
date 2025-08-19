import asyncio
import aiohttp
from flask import Flask, render_template, request, redirect, url_for
import pickle
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import Optional, Tuple, List, Dict, Any
# from asgiref.wsgi import WsgiToAsgi  # Import for ASGI compatibility

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY", "f2bb76c033335b0589966075a163919c")

# Load the saved files
try:
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    movie_titles_set = set(movies['title'].str.lower().str.strip())
    logging.info(f"Loaded {len(movies)} movies")
except Exception as e:
    logging.error(f"Failed to load pickle files: {e}")
    raise

# Simple in-memory cache
cache = {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def get_movie_id(title: str, session: aiohttp.ClientSession) -> Optional[int]:
    """Get movie ID from TMDB API"""
    cache_key = f"movie_id_{title.lower()}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': API_KEY,
            'query': title,
            'language': 'en-US'
        }

        async with session.get(url, params=params, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()

            if data.get('results'):
                movie_id = data['results'][0]['id']
                cache[cache_key] = movie_id
                logging.info(f"Found movie ID {movie_id} for '{title}'")
                return movie_id
            else:
                logging.warning(f"No results found for movie title: {title}")
                cache[cache_key] = None
                return None

    except Exception as e:
        logging.error(f"Error fetching movie ID for '{title}': {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def get_movie_details(movie_id: int, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
    """Get detailed movie information from TMDB API"""
    cache_key = f"movie_details_{movie_id}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            'api_key': API_KEY,
            'language': 'en-US',
            'append_to_response': 'credits,videos,watch/providers'
        }

        async with session.get(url, params=params, timeout=15) as response:
            response.raise_for_status()
            data = await response.json()

            movie_details = {
                'id': data.get('id'),
                'title': data.get('title'),
                'overview': data.get('overview', 'No description available.'),
                'release_date': data.get('release_date', 'N/A'),
                'runtime': data.get('runtime', 0),
                'genres': [genre['name'] for genre in data.get('genres', [])],
                'vote_average': data.get('vote_average', 0.0),
                'vote_count': data.get('vote_count', 0),
                'poster_path': f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get('poster_path') else "https://via.placeholder.com/200x300?text=No+Image",
                'backdrop_path': f"https://image.tmdb.org/t/p/w1280{data.get('backdrop_path')}" if data.get('backdrop_path') else None,
                'budget': data.get('budget', 0),
                'revenue': data.get('revenue', 0),
                'production_companies': [company['name'] for company in data.get('production_companies', [])],
                'production_countries': [country['name'] for country in data.get('production_countries', [])],
                'spoken_languages': [lang['english_name'] for lang in data.get('spoken_languages', [])],
                'status': data.get('status', 'N/A'),
                'tagline': data.get('tagline', 'N/A'),
            }

            credits = data.get('credits', {})
            cast = credits.get('cast', [])[:10]
            crew = credits.get('crew', [])

            movie_details['cast'] = [
                {
                    'name': person['name'],
                    'character': person.get('character', 'N/A'),
                    'profile_path': f"https://image.tmdb.org/t/p/w185{person['profile_path']}" if person.get('profile_path') else None
                }
                for person in cast
            ]

            directors = [person['name'] for person in crew if person['job'] == 'Director']
            writers = [person['name'] for person in crew if person['job'] in ['Writer', 'Screenplay', 'Story']]
            producers = [person['name'] for person in crew if person['job'] == 'Producer']

            movie_details['directors'] = directors
            movie_details['writers'] = writers[:3]
            movie_details['producers'] = producers[:3]

            videos = data.get('videos', {}).get('results', [])
            trailers = [
                {
                    'name': video['name'],
                    'key': video['key'],
                    'site': video['site']
                }
                for video in videos
                if video['type'] == 'Trailer' and video['site'] == 'YouTube'
            ][:3]

            movie_details['trailers'] = trailers

            watch_providers = data.get('watch/providers', {}).get('results', {})
            us_providers = watch_providers.get('US', {})
            streaming_providers = []
            rent_providers = []
            buy_providers = []

            if 'flatrate' in us_providers:
                streaming_providers = [
                    {
                        'provider_name': provider['provider_name'],
                        'logo_path': f"https://image.tmdb.org/t/p/w92{provider['logo_path']}" if provider.get('logo_path') else None
                    }
                    for provider in us_providers['flatrate']
                ]

            if 'rent' in us_providers:
                rent_providers = [
                    {
                        'provider_name': provider['provider_name'],
                        'logo_path': f"https://image.tmdb.org/t/p/w92{provider['logo_path']}" if provider.get('logo_path') else None
                    }
                    for provider in us_providers['rent'][:5]
                ]

            if 'buy' in us_providers:
                buy_providers = [
                    {
                        'provider_name': provider['provider_name'],
                        'logo_path': f"https://image.tmdb.org/t/p/w92{provider['logo_path']}" if provider.get('logo_path') else None
                    }
                    for provider in us_providers['buy'][:5]
                ]

            movie_details['watch_providers'] = {
                'streaming': streaming_providers,
                'rent': rent_providers,
                'buy': buy_providers
            }

            cache[cache_key] = movie_details
            logging.info(f"Cached movie details for movie ID {movie_id}")
            return movie_details

    except Exception as e:
        logging.error(f"Error fetching movie details for ID {movie_id}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def get_actor_movies(actor_name: str, session: aiohttp.ClientSession) -> Optional[str]:
    """Get a movie from an actor's filmography that exists in our dataset"""
    cache_key = f"actor_{actor_name.lower()}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        search_url = f"https://api.themoviedb.org/3/search/person"
        search_params = {
            'api_key': API_KEY,
            'query': actor_name,
            'language': 'en-US'
        }

        async with session.get(search_url, params=search_params, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()

            if not data.get('results'):
                logging.warning(f"No actor found for: {actor_name}")
                cache[cache_key] = None
                return None

            person_id = data['results'][0]['id']
            actor_found_name = data['results'][0]['name']
            logging.info(f"Found actor: {actor_found_name} (ID: {person_id})")

            credits_url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits"
            credits_params = {
                'api_key': API_KEY,
                'language': 'en-US'
            }

            async with session.get(credits_url, params=credits_params, timeout=10) as credits_response:
                credits_response.raise_for_status()
                credits_data = await credits_response.json()

                cast_movies = credits_data.get('cast', [])
                if not cast_movies:
                    logging.warning(f"No movies found for actor: {actor_found_name}")
                    cache[cache_key] = None
                    return None

                sorted_movies = sorted(
                    cast_movies,
                    key=lambda x: (
                        x.get('popularity', 0),
                        x.get('release_date', '0000-00-00')
                    ),
                    reverse=True
                )

                valid_movies = []
                for movie in sorted_movies:
                    movie_title = movie.get('title', '').strip()
                    if movie_title and movie_title.lower() in movie_titles_set:
                        valid_movies.append(movie_title)

                if valid_movies:
                    selected_movie = valid_movies[0]
                    logging.info(f"Selected movie '{selected_movie}' for actor '{actor_found_name}'")
                    logging.info(f"Other valid movies: {valid_movies[1:6]}")
                    cache[cache_key] = selected_movie
                    return selected_movie
                else:
                    sample_titles = [m.get('title', 'N/A') for m in sorted_movies[:10]]
                    logging.warning(
                        f"No valid movies found for actor '{actor_found_name}'. Sample titles: {sample_titles}")
                    cache[cache_key] = None
                    return None

    except Exception as e:
        logging.error(f"Error fetching actor movies for '{actor_name}': {e}")
        return None

async def fetch_poster(movie_id: Optional[int], session: aiohttp.ClientSession) -> str:
    """Fetch movie poster URL"""
    if movie_id is None:
        return "https://via.placeholder.com/200x300?text=No+Image"

    cache_key = f"poster_{movie_id}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            'api_key': API_KEY,
            'language': 'en-US'
        }

        async with session.get(url, params=params, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()

            poster_path = data.get('poster_path')
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                cache[cache_key] = poster_url
                return poster_url
            else:
                placeholder_url = "https://via.placeholder.com/200x300?text=No+Image"
                cache[cache_key] = placeholder_url
                return placeholder_url

    except Exception as e:
        logging.error(f"Error fetching poster for movie_id {movie_id}: {e}")
        return "https://via.placeholder.com/200x300?text=No+Image"

async def recommend(movie: str) -> Tuple[List[str], List[str], List[Optional[int]]]:
    """Get movie recommendations with movie IDs"""
    if movie not in movies['title'].values:
        raise ValueError("Movie not found in database.")

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_titles = []
    recommended_posters = []
    recommended_ids = []

    async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10),
            timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        tasks = []
        for i, _ in movie_list:
            title = movies.iloc[i].title
            recommended_titles.append(title)
            tasks.append(get_movie_id(title, session))

        movie_ids = await asyncio.gather(*tasks, return_exceptions=True)

        poster_tasks = []
        for movie_id in movie_ids:
            if isinstance(movie_id, Exception) or movie_id is None:
                recommended_ids.append(None)
                poster_tasks.append(asyncio.create_task(
                    asyncio.coroutine(lambda: "https://via.placeholder.com/200x300?text=No+Image")()
                ))
            else:
                recommended_ids.append(movie_id)
                poster_tasks.append(fetch_poster(movie_id, session))

        posters = await asyncio.gather(*poster_tasks, return_exceptions=True)

        for poster in posters:
            if isinstance(poster, Exception):
                recommended_posters.append("https://via.placeholder.com/200x300?text=No+Image")
            else:
                recommended_posters.append(poster)

    return recommended_titles, recommended_posters, recommended_ids

async def get_selected_movie_data(movie_title: str) -> Tuple[str, Optional[int]]:
    """Get poster and movie ID for the selected movie"""
    async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=5),
            timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        movie_id = await get_movie_id(movie_title, session)
        poster_url = await fetch_poster(movie_id, session)
        return poster_url, movie_id

async def process_search(search_query: str) -> Tuple[Optional[str], Optional[Exception], bool]:
    """Process search query - check for movie title or actor name"""
    search_query = search_query.strip()

    for movie_title in movies['title']:
        if movie_title.lower() == search_query.lower():
            logging.info(f"Found exact movie match: {movie_title}")
            return movie_title, None, False

    async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=5),
            timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        actor_movie = await get_actor_movies(search_query, session)
        if actor_movie:
            logging.info(f"Found movie '{actor_movie}' for actor '{search_query}'")
            return actor_movie, None, True

    return None, ValueError(f"No movie or actor found for: {search_query}"), False

@app.route('/', methods=['GET', 'POST'])
def index():
    movie_list = movies['title'].tolist()
    recommendations = None
    selected_movie_data = None
    error = None
    selected_movie = None
    is_actor = False

    if request.method == 'POST':
        search_query = request.form.get('movie', '').strip()

        if not search_query:
            error = "Please enter a movie title or actor name"
            return render_template('index.html', movie_list=movie_list,
                                   recommendations=recommendations, error=error,
                                   selected_movie=selected_movie, is_actor=is_actor,
                                   selected_movie_data=selected_movie_data)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                selected_movie, search_error, is_actor = loop.run_until_complete(
                    process_search(search_query)
                )

                if search_error:
                    raise search_error

                # Get selected movie data (poster and ID)
                selected_poster, selected_movie_id = loop.run_until_complete(
                    get_selected_movie_data(selected_movie)
                )
                selected_movie_data = (selected_movie, selected_poster, selected_movie_id)

                # Get recommendations
                titles, posters, movie_ids = loop.run_until_complete(recommend(selected_movie))
                recommendations = list(zip(titles, posters, movie_ids))

                logging.info(f"Successfully generated {len(recommendations)} recommendations for '{selected_movie}'")

            finally:
                loop.close()

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            error = str(ve)
        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            error = "Something went wrong. Please try again."

    return render_template('index.html', movie_list=movie_list,
                           recommendations=recommendations, error=error,
                           selected_movie=selected_movie, is_actor=is_actor,
                           selected_movie_data=selected_movie_data)

@app.route('/movie/<int:movie_id>')
async def movie_details(movie_id: int):
    """Display detailed information about a specific movie"""
    try:
        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=5),
                timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            movie_info = await get_movie_details(movie_id, session)

            if not movie_info:
                return render_template('error.html',
                                       error="Could not fetch movie details.",
                                       movie_title="Unknown"), 404

            movie_title = movie_info['title']
            try:
                rec_titles, rec_posters, rec_ids = await recommend(movie_title)
                recommendations = list(zip(rec_titles, rec_posters, rec_ids))
            except Exception as e:
                logging.error(f"Error getting recommendations for {movie_title}: {e}")
                recommendations = []

            return render_template('movie_details.html',
                                   movie=movie_info,
                                   recommendations=recommendations)

    except Exception as e:
        logging.error(f"Error in movie_details route for ID {movie_id}: {e}", exc_info=True)
        return render_template('error.html',
                               error="An unexpected error occurred.",
                               movie_title="Unknown"), 500

# Create ASGI application for Uvicorn
# asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    
    # pass  # Run with `uvicorn app:asgi_app --reload`
