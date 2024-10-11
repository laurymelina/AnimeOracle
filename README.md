# CapstoneJulyNYC

Overview:
My capstone is a recommendation system for anime. There is a plethora of different anime to watch and at times it can be overwhelming to decide what the next one to watch will be. By using this recommendation system we will cater to each individual person, using an array of filter inputs to recommend anime that are in line with their criteria. If the user wants more information on a recommended anime the app will send them to the corresponding information pages in MyAnimeList.net.

Data:
My data was acquired from MyAnimeList.net (MAL). MAL is the world's largest anime and manga database and community created by an anime fan, for anime fans. They provide users with a quick and no-hassle way to catalog their anime and manga collections, as well as a platform to communicate with like minded fans, and keep up to date with important industry news. Each month they display 150 million page views to over 12 million unique visitors. Around 28% of their users are from the US, with Indonesia, the Phillipenes, and Brazil being the next highest user countries. They are used in over 200 countries worldwide.

My data currently consists of three main datasets from Kaggle, which are around 2 years old:
1. Anime.csv is a list of all of the anime plus key information like the anime id, url, title, synopsis, number of episodes, genre, etc.
2. User_Anime.csv is a file that contains the relationships between users and animes, specifically around their rating of different anime as well as their watch status of the anime.
3. Anime_Anime.csv is a file that contains the relationships between pairs of anime. It specifies the relationships between different anime including if one is the Prequel, Sequel, etc. of the other.


Flowchart:

This is the initial expected flowchart for my project:

![flowchart drawio (1)](https://github.com/user-attachments/assets/9aa827c9-0bdd-4033-b409-7d920d0c5264)
