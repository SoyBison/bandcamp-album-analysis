
import requests
from bs4 import BeautifulSoup
import os
import re
import multiprocessing as mp
from functools import partial
import pickle
import numpy as np
from PIL import Image
from time import sleep
from tqdm import tqdm

BCAMPURL = 'https://{ARTIST}.bandcamp.com'


def load_artist_tags(loc='artist_tags'):
    with open(loc, 'r') as f:
        x = f.read().split('\n')
    return x


def add_artist_tag(tag, loc='artist_tags'):
    with open(loc, 'a') as f:
        f.write(tag + '\n')


def cowdog(earl, loc='artist_tags', loops=3, n=0):  # 'earl' like 'url'
    # It's called 'cowdog' because it does the wrangling
    
    site = requests.get(earl)
    soup = BeautifulSoup(site.text, 'html.parser')
    footer = soup.find_all('li', attrs={'class': re.compile('recommended-album footer')})
    urls = {album.find('a', class_='go-to-album album-link')['href'] for album in footer}
    artists = {re.findall('(?<=//)[a-z0-9-_~]*(?=.)', tag)[0] for tag in urls}

    if not os.path.exists(loc):
        add_artist_tag(re.findall('(?<=//)[a-z0-9-_~]*(?=.)', earl)[0])

    knowns = load_artist_tags(loc)
    for tag in artists:
        if tag not in knowns:
            add_artist_tag(tag, loc)

    if n < loops:
        for album in urls:
            cowdog(album, loc, loops, n+1)


def get_album_covers(tag, loc='./covers/'):
    url = BCAMPURL.replace('{ARTIST}', tag)
    lib = requests.get(url)
    soup = BeautifulSoup(lib.text, 'html.parser')
    albums = soup.find('div', class_='leftMiddleColumns')
    try:
        albums = albums.find_all('a', class_=None)
    except AttributeError:
        return
    album_locs = []
    for a in albums:
        try:
            album_locs.append(url + a['href'])
        except KeyError:
            pass

    for album in album_locs:
        fname = loc + str(hash(album))
        if os.path.exists(fname):
            continue
        alb = requests.get(album)
        soup = BeautifulSoup(alb.text, 'html.parser')
        try:
            art = soup.find('div', id='tralbumArt')
            imgloc = art.find('a')['href']
        except (KeyError, AttributeError):
            continue
        img = requests.get(imgloc, stream=True)
        img.raw.decode_content = True
        im = np.array(Image.open(img.raw))

        titlesec = soup.find('h2', class_='trackTitle')
        artistsec = soup.find('span', itemprop='byArtist')
        tags = soup.find_all('a', class_='tag')
        tags = [tag.text.strip() for tag in tags]
        album_title = re.findall('(?<=/)[a-z-_~0-9]*$', album)[0]
        data_dict = {'cover': im,
                     'title': titlesec.text.strip(),
                     'artist': artistsec.text.strip(),
                     'tags': tags,
                     'album': album_title,
                     'url': album,
                     'store': tag}
        if not os.path.exists(loc):
            os.mkdir(loc)
        with open(fname, 'bw+') as jar:
            pickle.dump(data_dict, jar)


def album_cover_scrape(cover_loc='./covers/', artist_loc='artist_tags'):
    pool = mp.Pool()
    worker = partial(get_album_covers, loc=cover_loc)
    artists = load_artist_tags(artist_loc)
    pool.map(worker, tqdm(artists), chunksize=1)
    return True


if __name__ == '__main__':
    done = False
    while not done:
        try:
            done = album_cover_scrape()
        except requests.exceptions.ConnectionError:
            sleep(5)
