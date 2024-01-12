# Working with the Spotify_Youtube dataset
# Copyrighted content used for research 
# and educational purposes only

import pandas as pd
import numpy as np
import os
from pytube import YouTube

def load_data(path, instances=1):
    """
    Loads the Spotify_Youtube dataset
    """
    # Load the data
    data = pd.read_csv(path)

    # Drop the duplicates
    data = data.drop_duplicates()

    # Drop the rows with NaN values
    data = data.dropna()

    # Only keep instances number of rows
    data = data.iloc[:instances]

    # Specify columns to drop
    drop = ['Url_spotify', 'Album_type', 'Uri', 'Title', 'Channel', 'Views', 'Likes', 'Comments', 'Description', 'official_video', 'Stream']

    # Drop the columns
    data = data.drop(drop, axis=1)

    # Reset the index
    data = data.reset_index(drop=True)

    # Return the data
    return data

def get_stats(data: pd.DataFrame):
    """
    Returns the statistics of the dataset
    """
    # Get the number of non-licenced tracks
    non_licenced = len(data[data['Licensed'] == False])

    stats = {
        'Number of tracks': len(data),
        'Number of non-licenced tracks': non_licenced,
        'Number of licenced tracks': len(data) - non_licenced,
        'Number of artists': len(data['Artist'].unique()),
    }        

    # Return the statistics
    return stats

def scrape_data(data: pd.DataFrame, path):
    """
    Scrapes YouTube URLs from the dataset and appends path to .mp3 files
    """
    # Get the YouTube URLs
    urls = data['Url_youtube'].values

    # Iterate over the URLs
    for i, url in enumerate(urls):
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        destination = path
        stream.download(output_path=destination)
        
        # Rename to index.mp3
        os.rename(f'{destination}{stream.default_filename}', f'{destination}{i}.mp3')

def main ():
    # Load the data
    load_path = 'data/Spotify_Youtube.csv'
    data = load_data(load_path, instances=500)

    # Get the statistics
    stats = get_stats(data)

    # Print the statistics
    for stat in stats:
        print(f'{stat}: {stats[stat]}')

    # Scrape the data
    mp3_path = 'data/mp3/'
    scrape_data(data, mp3_path)
    
if __name__ == '__main__':
    main()