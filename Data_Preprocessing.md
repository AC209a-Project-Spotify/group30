---
nav_include: 2
title: Data Preprocessing
notebook: Data_Preprocessing.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

Here, we preprocess Spotify data. By the end of this preprocessing stage, we have 2 csv files containing data ready to be used for EDA and for building our predictive model.

Overview of the steps:
- Load tracks.json and playlists_from_200_search_words.json files
- Extract track features (This is for each playlist. The data is stored in a list of python dictionaries where a dictionary contains information for one playlist. In each dictionary, each key is a feature and each value is a list of feature values where each entry corresponds to one track in the playlist.)
- Build playlists dataframe (feature engineering).
    - For each playlist, there are playlist-level and track-level variables. We chose to take playlist-level variables as they are. For track-level numeric variables, we chose to calculate the **average and standard deviation**, and for track-level categorical variables, we chose to take the **mode and count the number of unique** occurrences. Since Spotify playlists/tracks are not directly labeled with genre, we defined the **genre of a playlist** to be the most freqently occurring artist genre among all its tracks.
        - Playlist-level predictors: number of tracks
        - Track-level predictors: average, standard deviation of all numerical track audio features (e.g. danceability, tempo), popularities (e.g. track, album), and number of available market; and mode and unique counts of track artist genre, key and time signature
- Build tracks dataframe 
    - For each track, we added a new column `genre`, which is based on the **mode** of its artists genres.
    
- Save playlists and tracks dataframes to csv. They are ready for EDA and model building.

## Define helper functions

These libraries and functions are used to preprocess the data scpared from the Spotify API.







```python
def load_data(file):
    """
    Function to load json file
    """
    with open(file, 'r') as fd:
        data_from_json = json.load(fd)
        return data_from_json
    
def extract_track_features(tracks_db, playlists):
    """
    Function to get track features and return a playlist dictionary with track features
    """ 
    processed_playlists = deepcopy(playlists)
    
    missing_counts = 0
    # Loop over each playlist
    for index, playlist in enumerate(processed_playlists):
        # get the list of track ids for each playlist
        track_ids = playlist['track_ids']
        track_feature_keys = ['acousticness', 'album_id', 'album_name', 'album_popularity','artists_genres', 
                              'artists_ids', 'artists_names', 'artists_num_followers', 'artists_popularities',
                              'avg_artist_num_followers', 'avg_artist_popularity', 'danceability', 'duration_ms',
                              'energy', 'explicit', 'instrumentalness', 'isrc', 'key', 'liveness', 
                              'loudness', 'mode', 'mode_artist_genre', 'name', 'num_available_markets',
                              'popularity', 'speechiness', 'std_artist_num_followers', 'std_artist_popularity',
                              'tempo', 'time_signature', 'valence']
        
        # new entries of audio features for each playlist as a list to append each track's audio feature
        for track_feature_key in track_feature_keys:
            playlist['track_' + track_feature_key] = []
        
        # append each tracks' audio features into the entries of the playlist
        for track_id in track_ids:
            # check if the track_id is in the scrapped_tracks
            if track_id in tracks_db.keys():
                # append each track's audio feature into the playlist dictionary
                for track_feature_key in track_feature_keys:
                    if track_feature_key in tracks_db[track_id].keys():
                        playlist['track_' + track_feature_key].append(tracks_db[track_id][track_feature_key])
            else:
                missing_counts += 1
        processed_playlists[index] = playlist
    print('tracks that are missing : {}'.format(missing_counts))
    return processed_playlists


def build_playlist_dataframe(playlists_dictionary_list):
    """
    Function to build playlist dataframe from playlists dictionary with track features
    """
    
    if playlists_dictionary_list[7914]['id'] == '4krpfadGaaW42C7cEm2O0A':
        del playlists_dictionary_list[7914]
        
    # features to take the avg and std
    features_avg = ['track_acousticness', 'track_avg_artist_num_followers', 'track_album_popularity',
                    'track_avg_artist_popularity', 'track_danceability', 'track_duration_ms', 
                    'track_energy', 'track_explicit', 'track_instrumentalness','track_liveness', 
                    'track_loudness', 'track_mode', 'track_num_available_markets',
                    'track_std_artist_num_followers', 'track_std_artist_popularity',
                    'track_popularity', 'track_speechiness', 'track_tempo', 'track_valence'
                   ]                
                      
    # features to take the mode, # of uniques
    features_mode = ['track_artists_genres','track_key','track_time_signature']

    # features as is
    features = ['collaborative', 'num_followers', 'num_tracks']

    processed_playlists = {}

    for index, playlist in enumerate(playlists_dictionary_list):
        playlist_info = {} 
        playlist_info['id'] = playlist['id']

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
    
    # Drop all observations (playlists) with missingness
    df_full = df.dropna(axis=0, how='any')
    df_full.reset_index(inplace=True, drop=True)
    
    # Define our genre labels
    predefined_genres =['pop rap', 'punk', 'korean pop', 'pop christmas', 'folk', 'indie pop', 'pop', 
                    'rock', 'rap' , 'house', 'indie', 'dance', 'edm', 'mellow', 'hip hop',  
                    'alternative', 'jazz', 'r&b', 'soul', 'reggae', 'classical', 'funk', 'country',
                    'metal', 'blues', 'elect']
    # Create a new column genre_category
    df_full['genre'] = None
    
    # Label genres
    genres = df_full['track_artists_genres_mode']
    for g in reversed(predefined_genres):
        df_full['genre'][genres.str.contains(g)] = g

    # Label all observations that did not match our predefined genres as 'other'  
    df_full['genre'].fillna('other', inplace=True)
    df_full.drop('track_artists_genres_mode', axis=1, inplace=True)
    
    return df_full
    

def build_track_dataframe(tracks_db):
    """
    Function to build track dataframe
    """
    df = pd.DataFrame(tracks_db).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'trackID'}, inplace=True)
    df.drop('album_genres', axis=1, inplace=True) # drop album genre because it's null for all tracks
    
    # Define our genre labels
    predefined_genres =['pop rap', 'punk', 'korean pop', 'pop christmas', 'folk', 'indie pop', 'pop', 
                    'rock', 'rap' , 'house', 'indie', 'dance', 'edm', 'mellow', 'hip hop',  
                    'alternative', 'jazz', 'r&b', 'soul', 'reggae', 'classical', 'funk', 'country',
                    'metal', 'blues', 'elect']
    
    # Drop all observations (tracks) with missingness
    df_full = df.dropna(axis=0, how='any')
    df_full.reset_index(inplace=True, drop=True)
    
    # Create a new column genre_category
    df_full['genre'] = None
    
    # Label genres
    genres = df_full['mode_artist_genre']
    for g in reversed(predefined_genres):
        df_full['genre'][genres.str.contains(g)] = g

    # Label all observations that did not match our predefined genres as 'other'  
    df_full['genre'].fillna('other', inplace=True)
    df_full.drop('mode_artist_genre', axis=1, inplace=True)
    
    return df_full
```


## Preprocess the playlists

Upon extracting relevant track features and performing necessary calculations on the extracted track features, we stored the playlist dataframe as a csv file for easy access in the later parts of this project.



```python
playlists = load_data('../../data_archive/playlists_from_200_search_words.json')
tracks_db = load_data('../../data_archive/tracks.json')

playlists_with_track_features = extract_track_features(tracks_db, playlists)

playlists_df = build_playlist_dataframe(playlists_with_track_features)

playlists_df.head()
```


    tracks that are missing : 505





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
      <th>collaborative</th>
      <th>id</th>
      <th>num_followers</th>
      <th>num_tracks</th>
      <th>track_acousticness_avg</th>
      <th>track_acousticness_std</th>
      <th>track_album_popularity_avg</th>
      <th>track_album_popularity_max</th>
      <th>track_album_popularity_std</th>
      <th>track_artists_genres_unique</th>
      <th>...</th>
      <th>track_std_artist_num_followers_std</th>
      <th>track_std_artist_popularity_avg</th>
      <th>track_std_artist_popularity_std</th>
      <th>track_tempo_avg</th>
      <th>track_tempo_std</th>
      <th>track_time_signature_mode</th>
      <th>track_time_signature_unique</th>
      <th>track_valence_avg</th>
      <th>track_valence_std</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>37i9dQZF1DX1N5uK98ms5p</td>
      <td>3000606</td>
      <td>52</td>
      <td>0.180999</td>
      <td>0.17112</td>
      <td>71.6731</td>
      <td>96</td>
      <td>13.1364</td>
      <td>60</td>
      <td>...</td>
      <td>921166</td>
      <td>1.78425</td>
      <td>3.08155</td>
      <td>116.689</td>
      <td>25.1949</td>
      <td>4</td>
      <td>1</td>
      <td>0.456071</td>
      <td>0.184214</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>37i9dQZF1DX5drguwUcl5X</td>
      <td>69037</td>
      <td>75</td>
      <td>0.144201</td>
      <td>0.160799</td>
      <td>68.44</td>
      <td>100</td>
      <td>15.5111</td>
      <td>70</td>
      <td>...</td>
      <td>1.53959e+06</td>
      <td>2.11486</td>
      <td>3.17182</td>
      <td>114.454</td>
      <td>24.115</td>
      <td>4</td>
      <td>2</td>
      <td>0.555027</td>
      <td>0.19144</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>37i9dQZF1DX9bAf4c66TGs</td>
      <td>385875</td>
      <td>38</td>
      <td>0.1166</td>
      <td>0.117615</td>
      <td>72.4211</td>
      <td>94</td>
      <td>16.1923</td>
      <td>44</td>
      <td>...</td>
      <td>2.05042e+06</td>
      <td>2.12676</td>
      <td>2.15179</td>
      <td>115.813</td>
      <td>22.7593</td>
      <td>4</td>
      <td>1</td>
      <td>0.526526</td>
      <td>0.201783</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>37i9dQZF1DX9nq0BqAtM4H</td>
      <td>69344</td>
      <td>40</td>
      <td>0.134162</td>
      <td>0.247197</td>
      <td>57.025</td>
      <td>82</td>
      <td>18.0838</td>
      <td>97</td>
      <td>...</td>
      <td>308030</td>
      <td>0.0375</td>
      <td>0.172753</td>
      <td>126.491</td>
      <td>29.5215</td>
      <td>4</td>
      <td>2</td>
      <td>0.501825</td>
      <td>0.188804</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>1dCUPq7sB98i1jgQmo9d7e</td>
      <td>15612</td>
      <td>26</td>
      <td>0.171635</td>
      <td>0.229736</td>
      <td>53.4615</td>
      <td>54</td>
      <td>0.498519</td>
      <td>5</td>
      <td>...</td>
      <td>12787</td>
      <td>3.34629</td>
      <td>3.18413</td>
      <td>126.678</td>
      <td>33.242</td>
      <td>4</td>
      <td>1</td>
      <td>0.658846</td>
      <td>0.184523</td>
      <td>pop</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>





```python
print('Number of observations with missing values: ', sum(playlists_df.isnull().any()))
```


    Number of observations with missing values:  0




```python
playlists_df.to_csv('../../data/playlists.csv', index=False)
```


## Preprocess the tracks 



```python
tracks_df = build_track_dataframe(tracks_db)
```




```python
tracks_df.head()
```





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
      <th>trackID</th>
      <th>acousticness</th>
      <th>album_id</th>
      <th>album_name</th>
      <th>album_popularity</th>
      <th>artists_genres</th>
      <th>artists_ids</th>
      <th>artists_names</th>
      <th>artists_num_followers</th>
      <th>artists_popularities</th>
      <th>...</th>
      <th>name</th>
      <th>num_available_markets</th>
      <th>popularity</th>
      <th>speechiness</th>
      <th>std_artist_num_followers</th>
      <th>std_artist_popularity</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000C3ZY8325A4yktxnnwCl</td>
      <td>0.952</td>
      <td>3ypgq6ExA3JN8s2biuRK5e</td>
      <td>Soft Ice</td>
      <td>36</td>
      <td>[drift]</td>
      <td>[4Uqu4U6hhDMODyzSCtNDzG]</td>
      <td>[Poemme]</td>
      <td>[531]</td>
      <td>[47]</td>
      <td>...</td>
      <td>When the Sun Is a Stranger</td>
      <td>62</td>
      <td>26</td>
      <td>0.0469</td>
      <td>0</td>
      <td>0</td>
      <td>134.542</td>
      <td>3</td>
      <td>0.0835</td>
      <td>other</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000EWWBkYaREzsBplYjUag</td>
      <td>0.815</td>
      <td>5WGfEM0WaAyoJa6AOSfx7T</td>
      <td>Red Flower</td>
      <td>59</td>
      <td>[chillhop]</td>
      <td>[0oer0EPMRrosfCF2tUt2jU]</td>
      <td>[Don Philippe]</td>
      <td>[1300]</td>
      <td>[56]</td>
      <td>...</td>
      <td>Fewerdolr</td>
      <td>62</td>
      <td>40</td>
      <td>0.0747</td>
      <td>0</td>
      <td>0</td>
      <td>76.43</td>
      <td>4</td>
      <td>0.56</td>
      <td>other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000hI2Lxs4BxqJyqbw7Y10</td>
      <td>0.108</td>
      <td>5RIqRVn99mfdZSVmgjBrfj</td>
      <td>Las 35 Baladas de Medina Azahara</td>
      <td>0</td>
      <td>[latin metal, rock en espanol, spanish new wav...</td>
      <td>[72XPmW6k6HZT6K2BaUUOhl]</td>
      <td>[Medina Azahara]</td>
      <td>[43172]</td>
      <td>[47]</td>
      <td>...</td>
      <td>Tu Mirada</td>
      <td>0</td>
      <td>0</td>
      <td>0.029</td>
      <td>0</td>
      <td>0</td>
      <td>72.474</td>
      <td>4</td>
      <td>0.41</td>
      <td>metal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000uWezkHfg6DbUPf2eDFO</td>
      <td>0.00188</td>
      <td>3MBXzJXHFBslpPUcxNB3jn</td>
      <td>Dancehall Days</td>
      <td>36</td>
      <td>[reggae rock]</td>
      <td>[0hDJSg859MdK4c9vqu1dS8]</td>
      <td>[The Beautiful Girls]</td>
      <td>[48518]</td>
      <td>[56]</td>
      <td>...</td>
      <td>Me I Disconnect From You</td>
      <td>39</td>
      <td>21</td>
      <td>0.0298</td>
      <td>0</td>
      <td>0</td>
      <td>134.008</td>
      <td>4</td>
      <td>0.362</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000x2qE0ZI3hodeVrnJK8A</td>
      <td>0.339</td>
      <td>2N0AgtWbCmVoNUl2GN1opH</td>
      <td>Dreamboat Annie</td>
      <td>62</td>
      <td>[album rock, art rock, classic rock, dance roc...</td>
      <td>[34jw2BbxjoYalTp8cJFCPv]</td>
      <td>[Heart]</td>
      <td>[413139]</td>
      <td>[70]</td>
      <td>...</td>
      <td>(Love Me Like Music) I'll Be Your Song</td>
      <td>62</td>
      <td>32</td>
      <td>0.0306</td>
      <td>0</td>
      <td>0</td>
      <td>134.248</td>
      <td>4</td>
      <td>0.472</td>
      <td>rock</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>





```python
tracks_df.to_csv('../../data/tracks.csv', index=False)
```

