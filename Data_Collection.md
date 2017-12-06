---
nav_include: 1
title: Data Collection
notebook: Data_Collection.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

Here, we document our process to collect data from the Spotify API. 

A brief overview of the steps: 

- Get authentication token for accessing the API
- Sample Spotify playlists 

    - Specify 150 predefined search keywords 
    - Use random word generator to generate 50 random words
    - Combine the 2 lists and use get_playlists_by_search_word() to get playlists that match the search words
    - For each keyword, scrape 50 playlists on Spotify that match
 
- Store playlist sample with associated attributes in a json file. Attributes include:
    - Playlist ID
    - Owner ID
    - Whether playlist is collaborative
    - Number of tracks
    - Track IDs
    - Number of follower
- Get track info from the sample of playlists. Information include:
    - Track name
    - Track ID
    - Track ISRC ID
    - Artist names
    - Artist IDs
    - Artist genres
    - Artist popularities
    - Artist number of followers
    - Album name
    - Album ID
    - Album popularity
    - Audio features ('danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature')
    - Whether lyrics is explicit
    - Number of available markets
    - Popularity
- Store all track information in a tracks.json file
Note: A track is often associated with multiple artists

## Define helper functions
These libraries and functions are used to scrape the Spotify API.







```python
def get_auth_spotipy () -> Spotify:
    """
    Function that returns authorized Spotify client
    """ 
    os.environ['SPOTIPY_CLIENT_ID'] = 'c4cd8ee33b624ca6b224debdef35ba58'
    os.environ['SPOTIPY_CLIENT_SECRET'] = '3d5127677713483d99ded163e45198c6'

    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp


def dump_data(data_to_json, file):
    """
    Function to save scraped data to json file
    example: file = '../data/playlists_5088.json'
    """
    with open(file,'w') as fd:
        json.dump(data_to_json, fd)


def load_data(file):
    """
    Function to load json file
    """
    with open(file, 'r') as fd:
        data_from_json = json.load(fd)
        return data_from_json


def generate_search_word(predefined, number):
    """
    Funciton to generate a list of search words, including a set of predefined words and randomly generated words
    """
    rw = RandomWords()
    random_words = rw.random_words(letter=None, count=number)
    search_words = predefined + random_words
    return list(set(search_words))


def get_playlists_by_search_word(search_words):
    """
    Function to get playlists from the Spotify API using a list of search words
    """
    all_playlists = []

    for word in search_words:
    #     time.sleep(1)
        try:
            print(word)
            playlists = sp.search(word, type='playlist', limit=50)

            for i, playlist in enumerate(playlists['playlists']['items']):
                playlist_info = {}
                user = playlist['owner']['id']

                try:
                    current_playlist = sp.user_playlist(user, playlist_id=playlist['id'])

                    tracks = current_playlist['tracks']['items']
                    track_ids = set()
                    for track in tracks:
                        if track['track']:
                            track_ids.add(track['track']['id'])

                    playlist_info['track_ids'] = list(track_ids) # convert the set of track_ids of a playlist into a list

                    playlist_info['num_followers'] = current_playlist['followers']['total']
                    playlist_info['collaborative'] = current_playlist['collaborative']
                    playlist_info['id'] = current_playlist['id']
                    playlist_info['owner_id'] = current_playlist['owner']['id']
                    playlist_info['num_tracks'] = current_playlist['tracks']['total']

                    all_playlists.append(playlist_info)

                except:
                    continue
        except:
            continue

    return all_playlists


def get_tracks_in_playlists(playlists):
    """
    Function to extract track features from a list of playlists
    """
    all_tracks = {}
    track_ids_set = set()
    len_playlists = len(playlists)
    for i, playlist in enumerate(playlists):
        sys.stdout.write('\r{0}% completed.'.format((float(i+1)/len_playlists)*100))
        sys.stdout.flush()

        track_ids = playlist['track_ids']
        track_ids = [i for i in track_ids if i] # remove NaNs in track_ids

        for track_id in track_ids:
            try:
                if track_id in track_ids_set:
                    continue
                track_ids_set.add(track_id)
                track_info = {}

                # Features related to the TRACK itself
                track = sp.track(track_id)
                track_info['id'] = track_id
                track_info['name'] = track['name']
                
                if 'explicit' in track:
                    track_info['explicit'] = track['explicit']
                if track['external_ids']:
                    track_info['isrc'] = track['external_ids']['isrc']
                
                track_info['num_available_markets'] = len(track['available_markets'])

                # Get audio_features of a track
                audio_feature_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                                     'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
                track_audio_feature = sp.audio_features(track_id)[0]
                if track_audio_feature:
                    for key in audio_feature_keys:
                        track_info[key] = track_audio_feature[key]
                track_info['popularity'] = track['popularity']
                
                
                # Features related to the ALBUM
                album = sp.album(track['album']['id'])
                track_info['album_name'] = album['name']
                track_info['album_id'] = album['id']
                track_info['album_genres'] = album['genres']
                track_info['album_popularity'] = album['popularity']
                
                # Features related to the ARTIST
                track_info['artists_names'] = []
                track_info['artists_ids'] = []
                track_info['artists_popularities'] = []
                track_info['artists_num_followers'] = []
                track_info['artists_genres'] = []
                
                for artist in track['artists']:
                    current_artist = sp.artist(artist['id'])
                    track_info['artists_names'].append(current_artist['name'])
                    track_info['artists_ids'].append(current_artist['id'])
                    track_info['artists_popularities'].append(current_artist['popularity'])
                    track_info['artists_num_followers'].append(current_artist['followers']['total'])
                    track_info['artists_genres'].extend(current_artist['genres'])
                
                
                track_info['avg_artist_popularity'] = np.mean(track_info['artists_popularities'])
                track_info['std_artist_popularity'] = np.std(track_info['artists_popularities'])
                track_info['avg_artist_num_followers'] = np.mean(track_info['artists_num_followers'])
                track_info['std_artist_num_followers'] = np.std(track_info['artists_num_followers'])
                
                if (track_info['artists_genres']):
                    # count the most freqent artist genre

                    counter = collections.Counter(track_info['artists_genres'])
                    track_info['mode_artist_genre'] = counter.most_common()[0][0]
                
                all_tracks[track_id] = track_info
                
            except:
                continue
    return all_tracks  


def missing_tracks(tracks_db, playlists):
    """
    Function that checks if tracks in playlists are missing from tracks database(tracks_db)
    Returns a set of missing track ids 
    """
    missing_counts = 0
    missing_tracks = []
    # Loop over each playlist
    for index, playlist in enumerate(playlists):
        # get the list of track ids for playlist
        track_ids = playlist['track_ids']
        
        # check if tracks in playlist are in the track database
        for track_id in track_ids:
            # check if the track_id is in the tracks_db
            if track_id in tracks_db.keys():
                continue
            else:
                missing_counts += 1
                missing_tracks.append(track_id)
    print('tracks that are missing : {}'.format(missing_counts))
    return set(missing_tracks)


def get_tracks_by_track_ids(track_ids):
    """
    Function to extract track features from a list of track ids
    """
    all_tracks = {}
    num_track_ids = len(track_ids)
    for i, track_id in enumerate(track_ids):
        try:
            time.sleep(1)
            sys.stdout.write('\r{0}% completed.'.format((float(i+1)/num_track_ids)*100))
            sys.stdout.flush()

            track_info = {}

            # Features related to the TRACK itself
            track = sp.track(track_id)
            track_info['id'] = track_id
            track_info['name'] = track['name']

            if 'explicit' in track:
                track_info['explicit'] = track['explicit']
            if track['external_ids']:
                track_info['isrc'] = track['external_ids']['isrc']

            track_info['num_available_markets'] = len(track['available_markets'])

            # Get audio_features of a track
            audio_feature_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                                 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
            track_audio_feature = sp.audio_features(track_id)[0]
            if track_audio_feature:
                for key in audio_feature_keys:
                    track_info[key] = track_audio_feature[key]
            track_info['popularity'] = track['popularity']


            # Features related to the ALBUM
            album = sp.album(track['album']['id'])
            track_info['album_name'] = album['name']
            track_info['album_id'] = album['id']
            track_info['album_genres'] = album['genres']
            track_info['album_popularity'] = album['popularity']

            # Features related to the ARTIST
            track_info['artists_names'] = []
            track_info['artists_ids'] = []
            track_info['artists_popularities'] = []
            track_info['artists_num_followers'] = []
            track_info['artists_genres'] = []

            for artist in track['artists']:
                current_artist = sp.artist(artist['id'])
                track_info['artists_names'].append(current_artist['name'])
                track_info['artists_ids'].append(current_artist['id'])
                track_info['artists_popularities'].append(current_artist['popularity'])
                track_info['artists_num_followers'].append(current_artist['followers']['total'])
                track_info['artists_genres'].extend(current_artist['genres'])

            track_info['avg_artist_popularity'] = np.mean(track_info['artists_popularities'])
            track_info['std_artist_popularity'] = np.std(track_info['artists_popularities'])
            track_info['avg_artist_num_followers'] = np.mean(track_info['artists_num_followers'])
            track_info['std_artist_num_followers'] = np.std(track_info['artists_num_followers'])

            if (track_info['artists_genres']):
                # count the most freqent artist genre
                counter = collections.Counter(track_info['artists_genres'])
                track_info['mode_artist_genre'] = counter.most_common()[0][0]
        except:
            continue

        all_tracks[track_id] = track_info
            
    return all_tracks  

sp = get_auth_spotipy()
```


## Define search queries (search words)
To broadly and randomly sample playlists from the Spotify API, we combine 150 predefined words and 50 randomly generated words as our search words to query the API. 

Our 150 predefined search words are mostly inspired from https://insights.spotify.com/us/2015/12/18/trending-playlist-words-2015/, https://www.digitaltrends.com/music/best-playlists-on-spotify/2/, and https://gizmodo.com/these-are-the-25-most-popular-spotify-playlists-510275721.



```python
predefined = ['your', 'my', 'are', 'the', 'is', 'a', 'can', 'love', 'hate', 'holiday', 'work', 'workday',
              'weekend', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'beautiful',
              'fall', 'summer', 'spring', 'winter', 'classics',  'throwback', 'car', 'morning', 'shower', 'current', 
              'jesus', 'party', 'gym','late', 'night', 'old', 'chill', 'country', 'new', 'feel', 'good', 'workout', 
              'slow','hood','tropical', 'EDM', 'wedding', 'sex', 'honeymoon', 'senior', 'cool', 'house', 'jam',
              'today', 'top', 'hits', 'dance', 'mix', 'teen', 'cardio', 'hangouts', 'hot', 'ultimate',
              'hip hop', 'mega', 'upbeat', 'acoustic', 'deep', 'girls', 'baby', 'indie', 'punk', 'rock', 'family',
              'funk', 'jazz', 'instrumental', 'rap', 'beats', 'future', 'happy', 'emotional', 'great', 'magic',
              'finds', 'escape', 'fresh', 'high', 'low', 'buzz', 'kill', 'mood', 'blue' 'dirty', 'soul', 'pop',
              'beach', 'dream', 'shuffle', 'date', 'romantic', 'prom', 'college', 'kids', 'sleep', 'serenade',
              'calm', 'light', 'heavy', 'soft', 'strong', 'drama', 'confession', 'blink', 'sad', 'heart',
              'trend', 'trending', 'max', 'folk', 'blues', 'contemporary', 'electric', 'R&B', 'alternative', 
              'easy', 'metal', 'reggae', 'southern', 'cozy', 'darling', 'like', 'you', 'I', 'club', 'mind', 
              'waltz', 'glow', 'crazy', 'women', 'men', 'vibes', 'wave', 'trip', 'crave', 'him', 'break', 
              'true', 'different', 'her']

search_words = generate_search_word(predefined=predefined, number=50)

dump_data(search_words, '../data/200_search_words.json')
```


## Query the Spotify API 
With these search words, we scrape the Spotify API for playlists that match these search queries. For each of the search words, we scraped 50 matched playlists. Therefore, with 200 search words, we scraped 9511 unique playlists. We save the matched playlists in a json file.



```python
keywords = load_data('../data/200_search_words.json')

all_playlists = get_playlists_by_search_word(keywords)

dump_data(all_playlists, '../data/playlists_from_200_search_words.json')
```


Next, we scrape the songs that are associated with these playlists. We only get the unique songs (i.e. if 2 playlists have the same song, we only scrape the song once). Note: because it takes a very long time to scrape, we  divided the list of playlists into 5 sections and scraped them in parallel. These json files containing track information are merged into a single track database at the conclusion of our data collection process.



```python
all_playlists = load_data('../data/playlists_from_200_search_words.json')

tracks_2000 = get_tracks_in_playlists(all_playlists[:2000])
tracks_4000 = get_tracks_in_playlists(all_playlists[2000:4000])
tracks_6000 = get_tracks_in_playlists(all_playlists[4000:6000])
tracks_8000 = get_tracks_in_playlists(all_playlists[6000:8000])
tracks_9000 = get_tracks_in_playlists(all_playlists[8000:9000])
tracks_last = get_tracks_in_playlists(all_playlists[9000:])

dump_data(tracks_2000, '../data/tracks_2000.json')
dump_data(tracks_4000, '../data/tracks_2000_4000.json')
dump_data(tracks_6000, '../data/tracks_4000_6000.json')
dump_data(tracks_8000, '../data/tracks_6000_8000.json')
dump_data(tracks_9000, '../data/tracks_8000_9000.json')
dump_data(tracks_last, '../data/tracks_last.json')
```


After the first scrape, we checked to see if we have missing tracks in our database. For those tracks that are missing, we re-scraped the Spotify API based on their track IDs and store all track data into our tracks.json database.



```python
tracks = {}
tracks_2000 = load_data('../data_archive/tracks_2000.json')
tracks_4000 = load_data('../data_archive/tracks_2000_4000.json')
tracks_6000 = load_data('../data_archive/tracks_4000_6000.json')
tracks_8000 = load_data('../data_archive/tracks_6000_8000.json')
tracks_9000 = load_data('../data_archive/tracks_8000_9000.json')
tracks_last = load_data('../data_archive/tracks_9000.json')

tracks.update(tracks_2000)
tracks.update(tracks_4000)
tracks.update(tracks_6000)
tracks.update(tracks_8000)
tracks.update(tracks_9000)
tracks.update(tracks_last)

missing_2000 = missing_tracks(tracks_2000, all_playlists[1000:2000])
missing_4000 = missing_tracks(tracks_4000, all_playlists[2000:4000])
missing_6000 = missing_tracks(tracks_6000, all_playlists[4000:6000])
missing_8000 = missing_tracks(tracks_8000, all_playlists[6000:8000])
missing_9000 = missing_tracks(tracks_9000, all_playlists[8000:9000])
missing_last = missing_tracks(tracks_last, all_playlists[9000:])

missing_tracks_2000 = get_tracks_by_track_ids(missing_2000)
missing_tracks_4000 = get_tracks_by_track_ids(missing_4000)
missing_tracks_6000 = get_tracks_by_track_ids(missing_6000)
missing_tracks_8000 = get_tracks_by_track_ids(missing_8000)
missing_tracks_9000 = get_tracks_by_track_ids(missing_9000)
missing_tracks_last = get_tracks_by_track_ids(missing_last)

tracks.update(missing_tracks_2000)
tracks.update(missing_tracks_4000)
tracks.update(missing_tracks_6000)
tracks.update(missing_tracks_8000)
tracks.update(missing_tracks_9000)
tracks.update(missing_tracks_last)

dump_data(tracks, '../data_archive/tracks.json')
```


Check one last time to confirm that there is no missing tracks from the playlists



```python
all_playlists = load_data('../data_archive/playlists_from_200_search_words.json')
tracks_db = load_data('../data_archive/tracks.json')
missing_tracks(tracks_db, all_playlists)
```


    tracks that are missing : 505





    {None}



Note: While there are still 505 tracks missing, these are entries without track IDs. We will discard these entries in subsequent data preprocessing step. 
