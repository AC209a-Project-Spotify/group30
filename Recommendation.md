---
nav_include: 6
title: Recommendation
notebook: Recommendation.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}


## Steps of recommendation:
- Get the user specified genre: E.g. `pop`
- Get the user specified number of tracks: E.g. `N`
- Get all tracks of this genre from the tracks database
- Sort the filtered tracks based on `popularity`
- Combinatorically generate **`N+2 choose N`** different playlists as the recommendation candidates
- Use our fitted regression model to predict the `num_followers` of each recommendation candidate
    - We used the **Meta Linear Regression** model **with main predictors and interaction terms** for recommendation here. This model has top test $R^2$ score among all our models.
- Return the playlist that has the highest predicted value of `num_followers` as the final recommendation

## Strategy of validating the recommendation:
- Find the **most similar** playlist from the playlists database
    - **Definition of Similairy:** || set(recommended_playlist_tracks)  $ \quad \cap \quad $  set(existing_playlist_tracks) ||
- Compare the predicted `num_followers` of the recommendation playlist and its most similar playlist

## Summary of findings: 
We find that the meta linear regression model has a good validation result when the user specified genre is `pop`, the genre that most playlists are associated with. The validation result would be off when the user specified genre is a genre that does not often appear in the playlists. This suggests that future work to improve recommendation is to gather balanced number of playlists/tracks among all genres.







```python
"""
Function to get track features and return a playlist dictionary with track features
""" 
def extract_track_features(tracks, playlists):
    processed_playlists = copy.deepcopy(playlists)
    
    missing_counts = 0
    # Loop over each playlist
    for index, playlist in enumerate(processed_playlists):
        track_feature_keys = ['acousticness', 'album_id', 'album_name', 'album_popularity','artists_genres', 
                              'artists_ids', 'artists_names', 'artists_num_followers', 'artists_popularities',
                              'avg_artist_num_followers', 'avg_artist_popularity', 'danceability', 'duration_ms',
                              'energy', 'explicit', 'instrumentalness', 'isrc', 'key', 'liveness', 
                              'loudness', 'mode', 'genre', 'name', 'num_available_markets',
                              'popularity', 'speechiness', 'std_artist_num_followers', 'std_artist_popularity',
                              'tempo', 'time_signature', 'valence']
        
        # new entries of audio features for each playlist as a list to append each track's audio feature
        for track_feature_key in track_feature_keys:
            playlist['track_' + track_feature_key] = []
        
        # append each tracks' audio features into the entries of the playlist
        selected_tracks = tracks[tracks['trackID'].isin(playlist['track_ids'])]
        for j, track in selected_tracks.iterrows():
            # append each track's audio feature into the playlist dictionary
            for track_feature_key in track_feature_keys:
                if track_feature_key in list(selected_tracks.columns):
                    playlist['track_' + track_feature_key].append(track[track_feature_key])
        processed_playlists[index] = playlist
    print('tracks that are missing : {}'.format(missing_counts))
    return processed_playlists

"""
Function to build playlist dataframe from playlists dictionary with track features
"""
def build_playlist_dataframe(playlists_dictionary_list):
    
    # features to take the avg and std
    features_avg = ['track_acousticness', 'track_avg_artist_num_followers', 'track_album_popularity',
                    'track_avg_artist_popularity', 'track_danceability', 'track_duration_ms', 
                    'track_energy', 'track_explicit', 'track_instrumentalness','track_liveness', 
                    'track_loudness', 'track_mode', 'track_num_available_markets',
                    'track_std_artist_num_followers', 'track_std_artist_popularity',
                    'track_popularity', 'track_speechiness', 'track_tempo', 'track_valence'
                   ]          
                      
    # features to take the mode, # of uniques
    features_mode = ['track_key','track_time_signature', 'track_genre']

    # features as is
    features = ['collaborative', 'num_followers', 'num_tracks']

    processed_playlists = {}

    for index, playlist in enumerate(playlists_dictionary_list):
        playlist_info = {} 
    #     playlist_info['id'] = playlist['id']

        for key in playlist.keys():
            if key in features_avg: # take avg and std
                playlist_info[key + '_avg'] = np.mean(playlist[key])
                playlist_info[key + '_std'] = np.std(playlist[key])
                if key in set(['track_popularity', 'track_album_popularity', 'track_avg_artist_popularity']):
                    playlist_info[key + '_max'] = max(playlist[key])
            elif key in features_mode: # take mode
                if playlist[key]:
                    if key == 'track_artists_genres':
                        flatten = lambda l: [item for sublist in l for item in sublist]
                        flattened_value = flatten(playlist[key])
                        if flattened_value:
                            counter = collections.Counter(flattened_value)
                            playlist_info[key + '_mode'] = counter.most_common()[0][0]
                            playlist_info[key + '_unique'] = len(set(flattened_value))
                    else:
                        counter = collections.Counter(playlist[key])
                        playlist_info[key + '_mode'] = counter.most_common()[0][0]
                        playlist_info[key + '_unique'] = len(set(playlist[key]))
            elif key in features:
                playlist_info[key] = playlist[key]

        processed_playlists[index] = playlist_info
    df = pd.DataFrame(processed_playlists).T
    df.rename(columns = {'track_genre_mode': 'genre'}, inplace = True)
    
    return df

def load_data(file):
    with open(file, 'r') as fd:
        data_from_json = json.load(fd)
        return data_from_json
```




```python
tracks_db = load_data('../../data_archive/tracks.json')
tracks_df = pd.read_csv('../../data/tracks.csv')
playlists_db = load_data('../../data_archive/playlists_from_200_search_words.json')
playlists_df = pd.read_csv('../../data/playlists_with_id.csv')

X_train_main = pd.read_csv('../../data/X_train_main.csv', index_col = 0)
X_train_int = pd.read_csv('../../data/X_train_int.csv', index_col = 0)
```




```python
def generate_playlists(genre, num_tracks):
    num_tracks_pool = num_tracks + 2
    
    # Filter tracks by popularity in ascending order
    tracks_filtered = tracks_df[tracks_df['genre'] == genre].sort_values('popularity', ascending = False).iloc[:num_tracks_pool]
    
    # generate combinations of track ids from the filtered tracks
    combination_tracks_ids = list(combinations(list(tracks_filtered['trackID']), num_tracks))
    
    # build the list of dictionaries consisting of tracks ids of size user_input_tracks
    playlist_track_ids = [{'track_ids':track_ids, 'num_tracks':num_tracks_pool} for track_ids in combination_tracks_ids]
    
    # extract the audio features for each playlists in the list of dictionaries
    processed_playlists = extract_track_features(tracks_df, playlist_track_ids)
    
    # build playlists from processed_playlists
    gen_playlists_df = build_playlist_dataframe(processed_playlists)
    
    return gen_playlists_df, playlist_track_ids

def preprocess_playlist_candidates(candidates_playlists_df):
    '''
    1. Take One-hot encoding for categorical variables;
    2. Add columns to match X_train_main
    3. Add interaction terms
    4. Standardize the numerical predictors
    '''
    # ======= One-hot encoding =======
    categorical_predictors = ['genre', 'track_time_signature_mode', 'track_key_mode']
    df_encoded = pd.get_dummies(candidates_playlists_df, prefix = categorical_predictors, columns = categorical_predictors)
    
    
    # ======= Add columns to match X_train_main =======
    df_encoded_full = pd.DataFrame()
    cur_columns = set(list(df_encoded.columns))
    for col in X_train_main.columns:
        if col in cur_columns:
            df_encoded_full = pd.concat([df_encoded_full, df_encoded[col]], axis = 1)
        else:
            df_encoded_full[col] = 0
    
    
    # ======= Add interaction terms =======
    audio_features_avg = ['track_acousticness_avg', 'track_album_popularity_avg', 'track_danceability_avg',
                    'track_duration_ms_avg', 'track_energy_avg', 'track_explicit_avg', 
                    'track_instrumentalness_avg', 'track_liveness_avg', 'track_loudness_avg', 'track_mode_avg', 
                    'track_speechiness_avg', 'track_tempo_avg', 'track_valence_avg']
    genres = ['genre_blues', 'genre_classical', 'genre_country', 'genre_dance',
           'genre_edm', 'genre_elect', 'genre_folk', 'genre_funk', 'genre_hip hop',
           'genre_house', 'genre_indie', 'genre_indie pop', 'genre_jazz',
           'genre_korean pop', 'genre_mellow', 'genre_metal', 'genre_other',
           'genre_pop', 'genre_pop christmas', 'genre_pop rap', 'genre_punk',
           'genre_r&b', 'genre_rap', 'genre_reggae', 'genre_rock', 'genre_soul',]

    cross_terms = audio_features_avg + genres
    
    df_encoded_int = deepcopy(df_encoded_full)
    for feature in audio_features_avg:
        for genre in genres:
            df_encoded_int[feature+'_X_'+genre] = df_encoded_int[feature] * df_encoded_int[genre]

    
    # ======= Standardize the numerical predictors =======
    df_recommendation_int = deepcopy(df_encoded_int)
    for col in X_train_int.columns:
        if not np.logical_or((df_recommendation_int[col]==0), ((df_recommendation_int[col]==1))).all():
            mean_train = X_train_int[col].mean()
            std_train = X_train_int[col].std()
            df_recommendation_int[col] = (df_recommendation_int[col] - mean_train) / std_train

    return df_recommendation_int
    
    
def recommended_playlist(genre, num_tracks=30):
    gen_playlists_df, playlist_track_ids = generate_playlists(genre, num_tracks)
    df_recommendation_int = preprocess_playlist_candidates(gen_playlists_df)
    
    # Load meta model
    meta_model = joblib.load('../../fitted_models/meta_reg_int.pkl')
    prefix = '../../fitted_models/'
    suffix = '.pkl'
    models_int = ['sim_lin_int', 'ridge_cv_int', 'lasso_cv_int', 'RF_best_int', 'ab_best_int']

    # Record each single model's predicted results
    meta_X_recommendation_int = np.zeros((df_recommendation_int.shape[0], len(models_int)))
    for i, name in enumerate(models_int):
        model_name = prefix + name + suffix
        model = joblib.load(model_name) 
        meta_X_recommendation_int[:, i] = model.predict(df_recommendation_int)

    predicted_log_num_followers = meta_model.predict(meta_X_recommendation_int)
    predicted_num_followers = np.exp(predicted_log_num_followers) - 1

    recommendation_playlist = playlist_track_ids[np.argmax(predicted_num_followers)]
    recommendation_playlist_pred_num_followers = max(predicted_num_followers)
    display_recommendation_df = get_recommendation_tracks_display_info(recommendation_playlist)
    
    print('The recommended playlist is:')
    display(display_recommendation_df)
    return recommendation_playlist, recommendation_playlist_pred_num_followers

def get_recommendation_tracks_display_info(recommendation):
    display_info_list = []
    recommend_track_ids = recommendation['track_ids']
    for track_id in recommend_track_ids:
        display_info = {}
        track_info = tracks_db[track_id]
        display_info['track name'] = track_info['name']
        display_info['artists names'] = track_info['artists_names']
        display_info_list.append(display_info)
    return pd.DataFrame(display_info_list)

def validate_recommendation_playlist(recommendation_playlist, recommendation_playlist_pred_num_followers):
    # go through the playlist db to find the most similar existing playlist to our recommendation
    dissimilarity_list = []
    for playlist in playlists_db:
        dissimilarity_list.append(len(set(recommendation_playlist['track_ids']) - set(playlist['track_ids'])))
    
    print('min # different tracks: {}'.format(min(dissimilarity_list)))
    most_similar_playlist_id = playlists_db[dissimilarity_list.index(min(dissimilarity_list))]['id']
    most_similar_playlist = playlists_df[playlists_df['id'] == most_similar_playlist_id]
    most_similar_playlist_num_followers = most_similar_playlist['num_followers'].values[0]
    
    processed_most_similar_playlist = preprocess_playlist_candidates(most_similar_playlist)
    
    # Load meta model
    meta_model = joblib.load('../../fitted_models/meta_reg_int.pkl')
    prefix = '../../fitted_models/'
    suffix = '.pkl'
    models_int = ['sim_lin_int', 'ridge_cv_int', 'lasso_cv_int', 'RF_best_int', 'ab_best_int']

    # Record model's predicted results on validation set as the train set for the meta regressor
    meta_X_processed_most_similar_playlist = np.zeros((processed_most_similar_playlist.shape[0], len(models_int)))
    for i, name in enumerate(models_int):
        model_name = prefix + name + suffix
        model = joblib.load(model_name) 
        meta_X_processed_most_similar_playlist[:, i] = model.predict(processed_most_similar_playlist)

    most_similar_predicted_log_num_followers = meta_model.predict(meta_X_processed_most_similar_playlist)
    most_similar_predicted_num_followers = (np.exp(most_similar_predicted_log_num_followers) - 1)[0]

    print('\nPredicted num_followers: {}'.format(recommendation_playlist_pred_num_followers))
    print('The most similar playlist\'s num_followers: {}'.format(most_similar_predicted_num_followers))
    
```




```python
recommendation, recommendation_pred_num_followers = recommended_playlist('pop', 10)
validate_recommendation_playlist(recommendation, recommendation_pred_num_followers)
```


    tracks that are missing : 0
    The recommended playlist is:



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artists names</th>
      <th>track name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Camila Cabello, Young Thug]</td>
      <td>Havana</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[ZAYN, Sia]</td>
      <td>Dusk Till Dawn - Radio Edit</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Ed Sheeran]</td>
      <td>Perfect</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Selena Gomez, Marshmello]</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Becky G, Bad Bunny]</td>
      <td>Mayores</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[Sam Smith]</td>
      <td>Too Good At Goodbyes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[Charlie Puth]</td>
      <td>Attention</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[Ed Sheeran]</td>
      <td>Shape of You</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[Maroon 5, SZA]</td>
      <td>What Lovers Do (feat. SZA)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[Lauv]</td>
      <td>I Like Me Better</td>
    </tr>
  </tbody>
</table>
</div>


    min # different tracks: 2
    
    Predicted num_followers: 14741.997686715062
    The most similar playlist's num_followers: 12219.764844058263




```python
recommendation, recommendation_pred_num_followers = recommended_playlist('rock', 12)
validate_recommendation_playlist(recommendation, recommendation_pred_num_followers)
```


    tracks that are missing : 0
    The recommended playlist is:



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artists names</th>
      <th>track name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Imagine Dragons]</td>
      <td>Thunder</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Imagine Dragons]</td>
      <td>Thunder</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Imagine Dragons]</td>
      <td>Believer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Imagine Dragons]</td>
      <td>Believer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Imagine Dragons]</td>
      <td>Whatever It Takes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[Twenty One Pilots]</td>
      <td>Stressed Out</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[Twenty One Pilots]</td>
      <td>Heathens</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[Twenty One Pilots]</td>
      <td>Ride</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[AC/DC]</td>
      <td>Highway to Hell</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[Eagles]</td>
      <td>Hotel California - Remastered</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[Red Hot Chili Peppers]</td>
      <td>Under The Bridge</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[AC/DC]</td>
      <td>Back In Black</td>
    </tr>
  </tbody>
</table>
</div>


    min # different tracks: 7
    
    Predicted num_followers: 14808.540723599252
    The most similar playlist's num_followers: 302338.3862527031




```python
recommendation, recommendation_pred_num_followers = recommended_playlist('jazz', 1)
validate_recommendation_playlist(recommendation, recommendation_pred_num_followers)
```


    tracks that are missing : 0
    The recommended playlist is:



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artists names</th>
      <th>track name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[The Cinematic Orchestra, Patrick Watson]</td>
      <td>To Build A Home</td>
    </tr>
  </tbody>
</table>
</div>


    min # different tracks: 0
    
    Predicted num_followers: 27894.382630047396
    The most similar playlist's num_followers: 37845.34328748115

