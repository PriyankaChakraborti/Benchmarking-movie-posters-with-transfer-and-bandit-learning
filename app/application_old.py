 # import modules
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.models import load_model, Model
from keras.preprocessing import image
import os
from scipy.spatial import distance
import pickle
import requests
from PIL import Image
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from urllib.request import urlopen
import urllib.request
import random
import glob
from tensorflow import Graph


arm_prob=[0.33, 0.33, 0.33]#assign some starting probabilities to the bandit , these could be randomly assigned as well
n_experiment=100
epsilon_decay=50 #probability of exploration
img_shape = (100,100,3)
N_episodes = 1
num_imgs_6 = 48475
genres = ['Action','Comedy','Drama','Horror','Romance','Thriller']

global arm, start_int, df, feature_array, img_height, img_width, learning_rate, model, graph

start_int = 0





#######################################################        
# Function to calculate hamming distance
def hamming_dist(y_true,y_pred):
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    return K.mean(K.sum(K.abs(y_true-y_pred),axis=1))

# Code to load the model
def generate_models():
    learning_rate = 10**(-6)
    img_height,img_width=img_shape[0],img_shape[1]
    model=load_model('static/data/vgg16_1.h5', custom_objects={"hamming_dist": hamming_dist})



########################################################
# use this to pandas data frame of movie_name,movie_layer and movie prediction
def make_prediction(image_uploaded,img_shape):
    '''send in uploaded image as numpy array,check with single image first'''
    global start_int, df, feature_array, img_height, img_width, learning_rate,  model, graph
    
    # Load model
    if start_int == 0:
        learning_rate = 10**(-6)
        img_height,img_width=img_shape[0],img_shape[1]
        
        graph = Graph()
        with graph.as_default():
            model=load_model('static/data/vgg16_1.h5', custom_objects={"hamming_dist": hamming_dist})
        
        
        
        # Load the feature_array
        pickle_in = open("static/data/feature_array.pickle","rb")
        feature_array = pickle.load(pickle_in)
        
        # Load the main dataframe
        df = pd.read_csv('static/data/full_poster_6.csv')
        
    start_int = 1
    
    # Get feature vector for uploaded image
    img = image.load_img(image_uploaded, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    del img
    x = x.astype('float32')
    x/=255.
    
    # Add new dimension
    x = np.expand_dims(x, axis=0)
    
    with graph.as_default():
        
        class_pred_prob = model.predict(x)
        
        output_layer = model.layers[-2].output
        input_layer = model.layers[0].input
        output_fn = K.function(input_layer, output_layer)
        class_layer = output_fn([x])
        
    class_pred = np.where(class_pred_prob>0.5,1,0)[0]
    
    # If nothing fit here it returns the one with the highest value
    if np.sum(class_pred) == 0:
        class_pred = np.zeros(6)
        class_pred_val = np.argmax(class_pred_prob,axis=1)[0]
        class_pred[int(class_pred_val)] = 1
    
    class_pred = [int(pred) for pred in class_pred]
    del x
    
    # Create new model, removing the last layer
    #feature_model = Model(model.inputs, model.layers[-2].output)
    #del model
    
    # Compile model
    #feature_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9), metrics=[hamming_dist,'accuracy'])
    
    #class_layer=feature_model.predict(x)
    #del feature_model
    
    #K.clear_session()
    
    return class_pred, class_layer

def make_recommendations(class_pred,class_layer,bandit_arm, df_rec, feature_array):
    
    # Make copy of dataframe
    temp_df = df_rec.copy(deep=True)
    del df_rec
    
    #df_rec=load_dataframe(class_pred)
    for index,row in temp_df.iterrows():
        curr_index = int(row['label_index']) # Holds index of feature_array for current movie
        #print(curr_index)
        curr_feature = feature_array[curr_index,:]
        
        if bandit_arm == 'minkowski':
            d_metric=distance.minkowski(class_layer,curr_feature,1)
        elif bandit_arm == 'euclidean':
            d_metric=distance.euclidean(class_layer,curr_feature)
        elif bandit_arm == 'cosine':
            d_metric=distance.cosine(class_layer,curr_feature)
            
        # Add calculated distance to dataframe
        temp_df.loc[index,'distance']=d_metric
        
    #del feature_array
    # sort movie by distance
    temp_df = temp_df.sort_values(by='distance', ascending=True)
    
    # Save these posters to the disk
    pred = temp_df.head()
    indices = pred['label_index'].tolist()
    
    #print(temp_df.head(10))
    
    return temp_df[:10]

def file_to_localfile(file):
    filename = os.path.join(application.config['UPLOAD_FOLDER'], 'target_poster' + str(random.randint(0,9999)) + '.jpg')
    
    file.save(filename)
    return filename

def url_to_localfile(url, verbose=False):
    
    response=requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    image = soup.find('div', class_ = "poster")
    
    file_name = os.path.join(application.config['UPLOAD_FOLDER'], 'target_poster' + str(random.randint(0,9999)) + '.jpg')
    
    urllib.request.urlretrieve(image.img['src'],file_name)
    
    return file_name

def random_to_localfile():
    
    # Calculate number of files in sample_posters folder
    
    num_posters = len(glob.glob("static/sample_posters/*.jpg"))
    
    # Randomly choose the poster to load
    img_index = random.randint(0,num_posters)
    
    # Locate chosen poster
    filename = glob.glob("static/sample_posters/*.jpg")[img_index]
    img = Image.open(filename)
    
    # Save chosen poster using target_poster filename
    filename = os.path.join(application.config['UPLOAD_FOLDER'], 'target_poster' + str(random.randint(0,9999)) + '.jpg')
    img.save(filename)
    del img
    
    return filename


def grab_movie_posters(df):
    found=[]
    not_found=[]
    count = 0
    saved_count = 1
    filenames = []
    for index,row in df.iterrows():
        if row['imdb'] == '-':
            continue
        if count >= 5:
            break
        count += 1
        
        movie_url=row['link']
        r = requests.get(movie_url)
        soup = BeautifulSoup(r.text, 'html.parser')  
        results = soup.find_all('div', attrs={'class':'poster'})  
        # As long as we can grab poster add to list
        
        if results:
            found.append([row['link'],row['movie'],row['year'],row['imdb']])
            
            # Also save the poster image
            response=requests.get(row['link'])
            soup = BeautifulSoup(response.text, "html.parser")
            
            image = soup.find('div', class_ = "poster")
            
            file_name = os.path.join(application.config['UPLOAD_FOLDER'], 'rec_poster_' + str(random.randint(0,9999)) + '.jpg')
            filenames.append(file_name)
            
                
            urllib.request.urlretrieve(image.img['src'],file_name)
            del image
            saved_count += 1
            
        else:
            not_found.append([row['movie'],row['year'],row['imdb']])
            
    return found,not_found, filenames
    
def add_image(ax_,movie_url,xy, imzoom):
    
    r = requests.get(movie_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all('div', attrs={'class':'poster'})
    first_result = results[0] 
    imgurl = first_result.find('img')['src']
    
    f = urlopen(imgurl)
    arr_img = plt.imread(f, format='jpg')
    
    imagebox = OffsetImage(arr_img, zoom=imzoom)
    imagebox.image.axes = ax_
    ab = AnnotationBbox(imagebox, xy, xybox=(0., 0.), boxcoords="offset points", pad=-0.5)  # hide box behind image
    ax_.add_artist(ab)
    return ax_

            
def make_plot(_l):
    
    df = pd.DataFrame.from_records(_l,columns=['link','movie','year','imdb'])
    
    plot_path = 'static/plot/poster_plot' + str(random.randint(0,9999)) + '.jpg'
    
    # Convert year and IMDB rating to numbers instead of strings
    years = [int(year) for year in df['year']]
    ratings = [float(rating) for rating in df['imdb']]
    
    # Create scatter plot using matploblib
    plt.clf()  # Clear figure if it previously existed
    plt.scatter(x=years, y=ratings, alpha=0.5)
    plt.title('Movie ratings vs Year')
    plt.xlabel('Year')
    plt.ylabel('IMDB Rating')
    fig = plt.gcf()
    fig.savefig(plot_path, dpi=96, bbox_inches='tight')  # save figure
    
    return plot_path



class Bandit(object):
    def __init__(self,num_arms,epsilon_decay,arm_prob):
        self.decay=epsilon_decay
        self.num_arms=num_arms
        self.counts = [0] * self.num_arms  # example: number of views
        self.values = [0.] * self.num_arms # example: number of clicks / views
        self.prob=[item for item in arm_prob]
        
    def get_action(self,force_explore=False):
        epsilon=self.get_epsilon()
        rand_num=np.random.rand()
        
        if rand_num<epsilon or force_explore:
            action_random=np.random.randint(self.num_arms)  # explore random arm of the bandit
            #print('Exploring and using: ' + str(action_random))
            return action_random
        else:
            action_greedy=np.argmax(self.values) #explore the arm that made the most rewards
            #print('Not exploring and using: ' + str(action_greedy))
            return action_greedy  
        
    def get_epsilon(self):
        """Produce epsilon"""
        total = np.sum(self.counts)
        return float(self.decay) / (total + float(self.decay))  
    def get_reward(self,action):
        rand = np.random.random()  # [0.0,1.0)
        #more rewards will be given for unexplored actions
        curr_reward = 1 if (rand < self.prob[action]) else 0
        return curr_reward
    def update_q(self,action,reward):
        """Update an arm with some reward value"""
        #print('')
        #print('Action index is: ' + str(action))
        #print('Provided reward is: ' + str(reward))
        
        # Read from disk before updating
        try:
            pickle_in = open("static/data/values.pickle","rb")
            values = pickle.load(pickle_in)
            #print(values)
            self.values = values
            pickle_in = open("static/data/counts.pickle","rb")
            self.counts = pickle.load(pickle_in)
            pickle_in = open("static/data/actions_taken.pickle","rb")
            actions_taken = pickle.load(pickle_in)
            pickle_in = open("static/data/reward_list.pickle","rb")
            reward_list = pickle.load(pickle_in)
        except:
            actions_taken = []
            reward_list = []
            pass
        
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        actions_taken.append(action)
        reward_list.append(reward)
        
        # Running product
        new_value = value + (1/n) * (reward - value)
        self.values[action] = new_value
        
        
        # Save to disk before exiting
        pickle_out = open('static/data/values.pickle','wb')
        pickle.dump(self.values, pickle_out)
        pickle_out.close()
        pickle_out = open('static/data/counts.pickle','wb')
        pickle.dump(self.counts, pickle_out)
        pickle_out.close()
        pickle_out = open('static/data/actions_taken.pickle','wb')
        pickle.dump(actions_taken, pickle_out)
        pickle_out.close()
        pickle_out = open('static/data/reward_list.pickle','wb')
        pickle.dump(reward_list, pickle_out)
        pickle_out.close()

# Definition to download and save posters
def scrape_image(link):
    
    r = requests.get(link)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all('div', attrs={'class':'poster'})
    first_result = results[0] 
    imgurl = first_result.find('img')['src']
    
    f = urlopen(imgurl)
    image = plt.imread(f, format='jpg')

    file_name = 'chosen_poster.jpg'
    filename = file_to_localfile(urllib.request.urlretrieve(image.img['src'],os.path.join(dir_path, file_name)))
    del image
    
    return filename


### main flask app ###
application = Flask(__name__)

arms={0:'minkowski',1:'euclidean',2:'cosine'}
eg = Bandit(3,epsilon_decay,arm_prob)
ids = {}
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@application.route('/')
def homepage():
    return render_template('homepage.html')

@application.route('/select', methods=['POST','GET'])
def selectpage():
    global arm
    
    # Delete all previous downloaded posters
    files = glob.glob(r'static/uploads/*')
    for items in files:
        os.remove(items)
    # Delete all previous plots
    files = glob.glob(r'static/plot/*')
    for items in files:
        os.remove(items)
    
    if request.method=='POST':
        if request.form['submit_button'] == 'Good Suggestions':
            reward = 1
        elif request.form['submit_button'] == 'Bad Suggestions':
            reward = -1 # do something else
        elif request.form['submit_button'] == 'Okay Suggestions':
            reward = 0
        eg.update_q(arm,reward)
        
    return render_template('select.html')

#### if storing files on aws use this #######################
DATA_FOLDER='static/data'
application.config['DATA_FOLDER']=DATA_FOLDER

@application.route('/static/data/<filename>')
def load_data_file(filename):
    temp=send_from_directory(application.config['DATA_FOLDER'], filename,as_attachment=True)
    if filename.split('.')[-1]=='pickle':
        pickle_in = open(temp,"rb")
        return pickle.load(pickle_in)
    elif filename.split('.')[-1]=='csv':
        return pd.read_csv(temp)
    elif filename.split('.')[-1]=='dat':
        return np.memmap(temp, dtype='float32', mode='r', shape=(num_imgs_6,img_shape[0],img_shape[1],img_shape[2]))
    elif filename.split('.')[-1]=='jpg': 
        return temp
#######################################################  

@application.route('/results', methods=['POST','GET']) 
def resultspage():
    global arm, df, feature_array
    
    # get form submit data
    filename = None
    try:
        file = request.files['image']
        target_name = file_to_localfile(file)
        #print('Image Uploaded')
    except:
        try:
            url = request.form['url']
            target_name = url_to_localfile(url)
            #print('URL Supplied')
        except:
            target_name = random_to_localfile()
            #print('Random File Requested')
            
    
    prediction,layer=make_prediction(target_name,img_shape)
    
    # bandit will choose arm
    arm=eg.get_action()
    
    #get the name of the arm,i.e,name of recommender metric
    arm_name=arms[arm]
    
    
    
        
    df_rec = df
    #del df
    
    for index,pred in enumerate(prediction):
        if pred == 0:
            # Ignore prediction if it is zero
            continue
        
        else:
            genre = genres[index] # collect name of current genre
            # Filter to include only those movies including pos prediction
            df_rec = df_rec[df_rec[genre]==1] 
            
    found,not_found,filenames=grab_movie_posters(make_recommendations(prediction,layer,arm_name, df_rec, feature_array))
    
    #del feature_array
    del layer
    del df_rec
    del not_found
    
    #plot the results
    plot_path = make_plot(found)
    del found
    
    #script, div = components(plot)
    return render_template("results.html", target_name=target_name,
                           rec_poster_1=filenames[0],rec_poster_2=filenames[1],
                           rec_poster_3=filenames[2],rec_poster_4=filenames[3],
                           rec_poster_5=filenames[4], plot_path=plot_path)
    
@application.route('/state', methods=['GET','Post'])
def get_results():
    
    action_list = ['minkowski', 'euclidean', 'cosine']
    reward_list = ['Good','Neutral','Bad']
    
    # Load current counts for each action
    try:
        pickle_in = open("static/data/values.pickle","rb")
        values = pickle.load(pickle_in)
        pickle_in = open("static/data/counts.pickle","rb")
        counts = pickle.load(pickle_in)
        pickle_in = open("static/data/actions_taken.pickle","rb")
        actions_taken = pickle.load(pickle_in)
        
    except: # Run if these have not yet been created
        return render_template('select.html')
    
    # Plot how many times each metric is used
    plt.clf()  # Clear figure if it previously existed
    # Calculate number of times minkowski used
    minkowski_num = 0
    euclidean_num = 0
    cosine_num = 0
    for val in actions_taken:
        if int(val)==0:
            minkowski_num += 1
        if int(val)==1:
            euclidean_num += 1
        if int(val)==2:
            cosine_num += 1
    num_list = [minkowski_num, euclidean_num, cosine_num]
    plt.bar(x=list(range(len(num_list))), height=num_list, width=0.8)
    plt.title('Frequency for Similarity Metrics')
    plt.xlabel('Type of Similarity Metric')
    plt.ylabel('Number of Times Used')
    plt.xticks(ticks=list(range(len(action_list))), labels=action_list)
    fig = plt.gcf()
    value_path = 'static/plot/value_plot' + str(random.randint(0,9999)) + '.jpg'
    fig.savefig(value_path, dpi=96, bbox_inches='tight')  # save figure
    
        # Make plot of rewards given over time
    pickle_in = open("static/data/reward_list.pickle","rb")
    reward_list = pickle.load(pickle_in)
    reward_list = [int(reward) for reward in reward_list]
    #print(reward_list)
    feedback_index = list(range(len(reward_list)))
    plt.clf()  # Clear figure if it previously existed
    plt.plot(feedback_index, reward_list)
    plt.title('Reward Given Over Time')
    plt.xlabel('Number of Times Feedback Provided')
    plt.ylabel('Reward Given')
    #plt.yticks(ticks=list(range(len(action_list))), labels=reward_list)
    fig = plt.gcf()
    reward_path = 'static/plot/reward_plot' + str(random.randint(0,9999)) + '.jpg'
    fig.savefig(reward_path, dpi=96, bbox_inches='tight')  # save figure
    
    # Plot Value for each arm
    feedback_index = list(range(len(action_list)))
    plt.clf()  # Clear figure if it previously existed
    plt.bar(x=feedback_index, height=values, width=0.8)
    plt.title('Current Value for each Similarity Metric')
    plt.xlabel('Similarity Metric')
    plt.ylabel('Value for Metric')
    plt.xticks(ticks=list(range(len(action_list))), labels=action_list)
    fig = plt.gcf()
    arm_choice_path = 'static/plot/arm_choice_plot' + str(random.randint(0,9999)) + '.jpg'
    fig.savefig(arm_choice_path, dpi=96, bbox_inches='tight')  # save figure
    

    
    return render_template('state.html', arm_choice_path=arm_choice_path, value_path=value_path, reward_path=reward_path)
    
if __name__ == '__main__':
   application.run(debug = False, threaded=False)
   