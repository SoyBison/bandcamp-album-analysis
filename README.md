# Bandcamp Album Analysis

This is a project to look into the question:

<big>

**How do indie musicians use visual signs to indicate their subgenre?**

</big>

The main functions are in `bandcamp_webtools.py`, and do basic web scraping. Analysis will be done in the jupyter notebooks `Proposal.ipynb`, `Prelims.ipynb`, and `Analysis.ipynb`. These will also be uploaded as blog posts to [my website](https://www.coeneedell.com).

The Exploratory analysis (corresponding to `Proposal.ipynb`) can also be found at [this blog post.](http://www.coeneedell.com/post/bandcamp_proposal)

## Core Concepts

### Web Scraping and Data Collection

The data collection process consists of two stages. The first stage is about collecting targets. Currently, I have implemented one algorithm for this, but I have plans to implement another. The first algorithm, stored as `cowdog` (because it wrangles the artists), takes in a starting position, a bancamp release page, and records all of the "recommended releases" shown at the bottom of the page. It then runs recursively on those artists, up to a set recursion depth. I intend to implement another algorithm that goes through the new and notable, and genre-defining releases from the bandcamp homepage for a certain genre, and runs the `cowdog` function on each of these to a set depth. (Probably depth 2). This proposed method will make the dataset more generalizable, but less thorough.

The second stage of data collection, which is stored in the `album_cover_scrape` function, goes through the stored artist list and downloads album covers, and stores them with identifying information (tags, artist name, url, etc.). These are stored with a file name decided by the hash of the release url.

### Color Decomposition

After scraping a large corpus of tagged cover art from bandcamp, we can decompose the images into their dominant colors. The dataset used for `Proposal.ipynb` and the exploratory data analysis was of 1100+ albums from 400+ artists. The artists were acquired in a 4-step (recursion depth set to 4) region around [Chillhop Records](https://chillhop.bandcamp.com/). For the work in `Prelims.ipynb`, the data set was collected in a 4-step region around Chillhop Records, [Peggy Sue](https://peggysue.bandcamp.com), [BS0 Music](https://bs0music.bandcamp.com), [Daupe](https://daupe.bandcamp.com), and [Tops](https://tops.bandcamp.com), which were chosen to be a somewhat representative sample of bandcamp's offerings, in addition to the previously collected Chillhop Records. The final Analysis will likely use the breadth-first collection discussed before.

The album covers are decomposed into colorgrams, using the `colorgram.py` package. An n-colorgram is a list of the n most dominant colors for an image. For `Proposal.ipynb`, 6-colorgrams are used, but for `Prelims.ipynb`, the images are decomposed into 40-colorgrams, to make the objects more accessible to Latent Dirichlet Analysis.

### Unsupervised Statistical Learning

In `Proposal.ipynb`, only very basic analysis tools are used. The most dominant colors are plotted on an HLS scale (Hue, Luminosity, Saturation), to make it easier to visualize the dominant colors. In addition, the brightnesses are plotted against one another for different genres to visualize the pattern between "Chillhop" as a subgenre of "Hip-Hop" and how the introduction of a generally less-serious outlook changes the way these artists portray their work.

In `Prelims.ipynb`, more complicated statistical techniques are used, especially techniques like Latent Dirichlet Allocation, which are generally utilized in a text analysis setting. In a way, this is an attempt to treat color usage in album covers as a sort of natural language.

## Additional Information

This project is currently ongoing, if you have any questions feel free to contact me through github or my website [email form](https://www.coeneedell.com/#contact). This project uses python, with a standard array of data science packages, plus the very excellent colorgram generating package [`colorgram.py`](https://pypi.org/project/colorgram.py/).
