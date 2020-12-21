# Final project-CSE896
Here is our final project for the CSE 896 graduate Deep Learning Course.
# Project goals
Visual discovery is a key element in social media as we speak today especially in any commercial image that is aimed at evoking a real response. With independent movies being churned out especially for streaming platforms like Netflix some posters are specifically being created for social media platforms. The challenge today is to be minimalist yet deliver the main imagery through powerful representation. With that said our goal was to explore whether a powerful convolutional architecture backed with a traditional machine learning pipeline employing graph embedding approaches could be leveraged to learn this morphology along with contextual labels to recommend similar movies that perhaps evoke the same feeling.<br>
To that end we deployed an app on AWS that recommends similar movies based on an uploaded image and learns from live user feed back which metric to use for this recommendation. The feedback is provided to a multi armed bandit with its arms being cosine similarity, Manhattan, and Euclidean distance. Similar ideas like this are being explored by ad tech companies like Ad theorent to recommend targeted ad to geospatial groups
Perhaps, if not a book, we may eventually be able to judge a movie by its cover!
# Instructions on local testing
Clone the repository to your machine.<br>
Download the vgg16_1.h5 file from below into the and app/static/data folder (it is too large to host on Github):<br>
https://od.lk/f/MzlfMjUzMTQwOTBf<br>
Install any necessary depencies.<br>
Inside the app folder open the application.py file (I used the Spyder IDE).<br>
The app should be accessible through your browser, probably at http://127.0.0.1:5000.
# Link to google slide show
https://drive.google.com/file/d/1NTJlm3S5E5WUFryhYDiwGoCE5aMG11ZW/view?usp=sharing
# Packages used
Keras back end for VGG-16 and Flask for the app. All bandit codes were written from scratch.
