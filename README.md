Movie Recommendation System

Project Overview
This is a Python-based content-driven Movie Recommendation System that suggests similar movies using Natural Language Processing techniques. The system reads movie and rating data from CSV files and recommends movies based on genre similarity using CountVectorizer and cosine similarity from Scikit-Learn. This approach does not train a machine learning model; instead, it applies NLP-based similarity on text features.

Features
• Reads movie information from CSV files
• Uses text processing to compare movie genres
• Recommends movies using cosine similarity
• Provides top similar movies based on content features
• Fully implemented in Python using Scikit-Learn for NLP operations

Technologies Used
• Programming Language: Python
• Libraries: Pandas, NumPy, Scikit-Learn
• Dataset Format: CSV files (movies.csv, ratings.csv)

Project Files
• movie_recommender.py – Main Python program
• movies.csv – Dataset containing movie titles, genres, etc.
• ratings.csv – Dataset containing rating information

How to Run
1. Install Python 3
2. Install required libraries: pandas, numpy, scikit-learn
3. Run the program using: python movie_recommender.py
4. Provide the dataset CSV files when prompted
5. The system will recommend similar movies based on genre similarity

Dataset Source
The dataset contains publicly available movie metadata and rating information suitable for academic and learning use.

Skills Demonstrated
• Python programming
• CSV file handling
• Natural Language Processing (CountVectorizer)
• Cosine similarity for recommendation systems
• Practical implementation of content-based filtering

Developer
Dino Sajan
Portfolio: https://github.com/dinosajan
