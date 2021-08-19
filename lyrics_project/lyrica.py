import json
import lyricsgenius as lg
import apikey

genius = lg.Genius(access_token=apikey.access_token, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
musician = input("Artist's Name: ")
artist = genius.search_artist(musician.capitalize(), max_songs=7, sort="popularity")
print(artist)

