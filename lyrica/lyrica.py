import json
import lyricsgenius as lg
import apikey

genius = lg.Genius(access_token=apikey.access_token, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
name = input("Artist's Name (ignore if none): ")
song = input("What's the name of the song:")
# singer = genius.search_artist(name, max_songs=5, sort='popularity')
lyrica = genius.search_song(song.capitalize(), artist=name)
print(lyrica.lyrics)


# Help Docs:
# https://melaniewalsh.github.io/Intro-Cultural-Analytics/Data-Collection/LyricsGenius.html
# https://medium.com/analytics-vidhya/getting-lyrics-of-songs-via-python-lyricsgenius-23e5dd5992e3