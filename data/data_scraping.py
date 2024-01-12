# Working with the Spotify_Youtube dataset
# Copyrighted content used for research 
# and educational purposes only

import pandas as pd
import numpy as np
import os
from pytube import YouTube

def load_data(path):
    """
    Loads the Spotify_Youtube dataset
    """
    # Load the data
    data = pd.read_csv(path)

    # Drop bad rows
    data = data.drop_duplicates()
    data = data.dropna()

    # Drop the unnecessary columns
    drop = ['Unnamed: 0', 'Url_spotify', 'Album_type', 'Uri', 'Title', 'Channel', 'Views', 'Likes', 'Comments', 'Description', 'official_video', 'Stream']
    data = data.drop(drop, axis=1)

    # Reset the index
    data = data.reset_index(drop=True)

    return data

def get_stats(data: pd.DataFrame):
    """
    Returns the statistics of the dataset
    """
    # Get the number of non-licenced tracks
    non_licenced = len(data[data['Licensed'] == False])

    stats = {
        'Number of tracks': len(data),
        'Number of artists': len(data['Artist'].unique()),
        'Number of albums': len(data['Album'].unique()),
    }        

    # Return the statistics
    return stats

def zip_data(data: pd.DataFrame, instances):
    """
    Scrapes YouTube URLs from the dataset and appends path to .mp3 files
    """
    # Keep random instances number of rows
    data = data.sample(n=instances, random_state=1)

    # Set the destination
    destination = f'data/{instances}_mp3/'

    # Iterate over the URLs
    for row in data.iterrows():
        try:
            yt = YouTube(row[1]['Url_youtube'])
            stream = yt.streams.filter(only_audio=True).first()
            stream.download(output_path=f'{destination}/mp3')

            # Rename to .mp3
            os.rename(f'{destination}/mp3/{stream.default_filename}', f'{destination}/mp3/{row[0]}.mp3')

            # add the path to the .mp3 file to the dataset
            data.loc[row[0], 'Path'] = f'{destination}mp3/{row[0]}.mp3'
        except:
            # If the URL is invalid, drop the row
            data = data.drop(row[0], axis=0)

            print(f'Error scraping {row[1]["Url_youtube"]}')

    # Save data to .csv 
    data.to_csv(f'{destination}/{instances}_mp3.csv', index=True)

    return data

def main ():
    # Load the data
    load_path = 'data/Spotify_Youtube.csv'
    data = load_data(load_path)

    # Zip the data
    data = zip_data(data, 70)

    # Get the statistics
    stats = get_stats(data)
    print('Statistics of dataset:')
    for key, value in stats.items():
        print(f'{key}: {value}')
    
if __name__ == '__main__':
    main()