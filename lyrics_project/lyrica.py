import json
import lyricsgenius as lg
import apikey

genius = lg.Genius(access_token=apikey.access_token, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
musician = input("Artist's Name: ")
song = input("What's the name of the song:")
artist = genius.search_artist(musician, max_songs=3, sort="popularity")
lyrica = genius.search_song(song.capitalize())
print(artist, lyrica.lyrics)

