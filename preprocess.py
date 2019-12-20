
import pandas as pd
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle


target_size_1 = (112,112)
genre_num = 6

pd.set_option('display.max_columns', 30)

dir_path = '_CSV'
#dir_path = 'CSV_Files_Drama1-6000'
img_path = '_Posters'

genres_18= ['Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Family','Fantasy','History',         'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

genres_6 = ['Action','Comedy','Drama','Horror','Romance','Thriller']

genres_5 = ['Action','Comedy','Drama','Horror','Romance']

genres_4 = ['Action','Comedy','Drama','Horror']

# Concatenate all csv files together
count = 0
for csv_path in glob.glob(dir_path + '/*.csv'):
    new_csv = pd.read_csv(csv_path)
    if count==0:
        full_poster_18 = new_csv
    else:
        full_poster_18 = pd.concat([full_poster_18,new_csv])
    count += 1
    
# Remove any rows missing genre information
full_poster_18.dropna(subset=['genre'], inplace=True)
# Drop any rows pertaining to the same movie
full_poster_18.drop_duplicates(subset=['movie','year'], inplace=True)
# Add label_index column to dataframe
full_poster_18["label_index"] = np.nan

# One-Hot encode the genres
for genre in genres_18:
    full_poster_18[genre] = [int(genre in cat) for cat in full_poster_18.genre.values]
    

if genre_num == 18:
    # Remove rows corresponding to having nothing
    full_poster_18 = full_poster_18[(full_poster_18.Action != 0) | (full_poster_18.Adventure != 0) | (full_poster_18.Animation != 0) | (full_poster_18.Biography != 0) | (full_poster_18.Comedy != 0) | (full_poster_18.Crime != 0) | (full_poster_18.Drama != 0) | (full_poster_18.Family != 0) | (full_poster_18.Fantasy != 0) | (full_poster_18.History != 0) | (full_poster_18.Horror != 0) | (full_poster_18.Musical != 0) | (full_poster_18.Mystery != 0) | (full_poster_18.Romance != 0) | (full_poster_18['Sci-Fi'] != 0) | (full_poster_18.Thriller != 0) | (full_poster_18.War != 0) | (full_poster_18.Western != 0)]
    print(full_poster_18.shape)

    # Calculate number of movies belonging only to each genre
    genre_dict = dict.fromkeys(genres_18, 0)
    for index, row in full_poster_18.iterrows():
        # Means there is only one per genre
        if int(np.sum(row[genres_18])) == 1:
            # Check each genre
            for genre in genres_18:
                if int(row[genre]) == 1:
                    genre_dict[genre] += 1
                    continue
    print('Below are the number of movies exclusively belonging to each genre for the 18 label space:')
    print(genre_dict)
    print('')

########## 6 GENRES ##############
if genre_num == 6:
# Remove rows corresponding to having nothing in genres_6 genres
    full_poster_6 = full_poster_18[(full_poster_18.Action != 0) | (full_poster_18.Comedy != 0) | (full_poster_18.Drama != 0) | (full_poster_18.Horror != 0) | (full_poster_18.Romance != 0) | (full_poster_18.Thriller != 0)]
    print('The shape of the dataframe is: ' + str(full_poster_6.shape))

    # Calculate number of movies belonging only to each genre
    genre_dict = dict.fromkeys(genres_6, 0)
    for index, row in full_poster_6.iterrows():
        # Means there is only one per genre
        if int(np.sum(row[genres_6])) == 1:
            # Check each genre
            for genre in genres_6:
                if int(row[genre]) == 1:
                    genre_dict[genre] += 1
                    continue
    print('Below are the number of movies exclusively belonging to each genre for the 6 label space:')
    print(genre_dict)
    print('')

if genre_num == 5:
    full_poster_5 = full_poster_18[(full_poster_18.Action != 0) | (full_poster_18.Comedy != 0) | (full_poster_18.Drama != 0) | (full_poster_18.Horror != 0) | (full_poster_18.Romance != 0)]
    print(full_poster_5.shape)

    # Calculate number of movies belonging only to each genre
    genre_dict = dict.fromkeys(genres_5, 0)
    for index, row in full_poster_5.iterrows():
        # Means there is only one per genre
        if int(np.sum(row[genres_5])) == 1:
            # Check each genre
            for genre in genres_5:
                if int(row[genre]) == 1:
                    genre_dict[genre] += 1
                    continue
    print('Below are the number of movies exclusively belonging to each genre for the 5 label space:')
    print(genre_dict)
    print('')

# Note for the 4 genre case we only keep cases where they belong to only a single genre
if genre_num == 4:
    full_poster_4 = full_poster_18[
        ((full_poster_18.Action==1)&(full_poster_18.Comedy==0)&(full_poster_18.Drama==0)&(full_poster_18.Horror==0)) | 
        ((full_poster_18.Action==0)&(full_poster_18.Comedy==1)&(full_poster_18.Drama==0)&(full_poster_18.Horror==0)) | 
        ((full_poster_18.Action==0)&(full_poster_18.Comedy==0)&(full_poster_18.Drama==1)&(full_poster_18.Horror==0)) | 
        ((full_poster_18.Action==0)&(full_poster_18.Comedy==0)&(full_poster_18.Drama==0)&(full_poster_18.Horror==1))]
    print('The shape of the dataframe is: ' + str(full_poster_4.shape))

    # Calculate number of movies belonging only to each genre
    genre_dict = dict.fromkeys(genres_4, 0)
    for index, row in full_poster_4.iterrows():
        # Means there is only one per genre
        if int(np.sum(row[genres_4])) == 1:
            # Check each genre
            for genre in genres_4:
                if int(row[genre]) == 1:
                    genre_dict[genre] += 1
                    continue
    print('Below are the number of movies exclusively belonging to each genre for the 5 label space:')
    print(genre_dict)
    print('')
    
def preprocess_image(genre_num, target_size_1, img_path, full_poster):
    
    count_total = 0
    num_imgs_total = len(glob.glob(img_path + '/*.jpg'))
    # Randomize order where we will store data
    np.random.seed(42) # Seeded so we always get same splits
    
    if genre_num == 18:
        count = 0
        num_imgs = 80762 # Manually found that this number of images in folder are also in csv file and valid
        labels = np.zeros((num_imgs, len(genres_18)))
        data = np.memmap('Data/data_18.dat', dtype='float32', mode='w+', shape=(num_imgs,target_size_1[0],target_size_1[1],3))
        idx = np.random.permutation(num_imgs)
    elif genre_num == 6:
        count = 0
        num_imgs = 68054 # Manually found that this number of images in folder are also in csv file and valid
        labels = np.zeros((num_imgs, len(genres_6)))
        data = np.memmap('Data/data_6.dat', dtype='float32', mode='w+', shape=(num_imgs,target_size_1[0],target_size_1[1],3))
        idx = np.random.permutation(num_imgs)
    elif genre_num == 5:
        count = 0
        num_imgs = 81266 # Manually found that this number of images in folder are also in csv file and valid
        labels = np.zeros((num_imgs, len(genres_5)))
        data = np.memmap('Data/data_5.dat', dtype='float32', mode='w+', shape=(num_imgs,target_size_1[0],target_size_1[1],3))
        idx = np.random.permutation(num_imgs)
    elif genre_num == 4:
        count = 0
        num_imgs = 81266 # Manually found that this number of images in folder are also in csv file and valid
        labels = np.zeros((num_imgs, len(genres_4)))
        data = np.memmap('Data/data_4.dat', dtype='float32', mode='w+', shape=(num_imgs,target_size_1[0],target_size_1[1],3))
        idx = np.random.permutation(num_imgs)
    
    print('Starting to go through images...')
    for picture in glob.glob(img_path + '/*.jpg'):
        
        # Display progress
        if count_total % 1000 == 0:
            print('Processing item ' + str(count_total) + ' out of ' + str(num_imgs_total))
        count_total += 1
        
        temp_name=picture.split('/')[-1][:-4]
        temp_title = temp_name[:-5]
        temp_year = int(temp_name[-4:])
        
        curr_array_index = idx[count]
        
        df=full_poster.loc[(full_poster['year']==temp_year) & (full_poster['movie']==temp_title)]
        
        # Test to see if the dataframe is empty (if so do not continue processing)
        if df.shape[0] == 0:
            continue
            
        # Load the image
        temp_image=image.load_img(picture,target_size = target_size_1)
        
        if genre_num == 18:
            new_label = df[genres_18].squeeze().tolist()
            labels[count,:] = new_label
            data[count,:,:,:] = temp_image
            data[count,:,:,:] = data[count,:,:,:]/255. # Normalize the data before saving
            full_poster['label_index'].loc[(full_poster['year']==temp_year) & (full_poster['movie']==temp_title)] = curr_array_index
            count += 1
            
        if genre_num == 6:
            new_label = df[genres_6].squeeze().tolist()
            labels[count,:] = new_label
            data[count,:,:,:] = temp_image
            data[count,:,:,:] = data[count,:,:,:]/255. # Normalize the data before saving
            full_poster['label_index'].loc[(full_poster['year']==temp_year) & (full_poster['movie']==temp_title)] = curr_array_index
            count += 1
            
        if genre_num == 5:
            new_label = df[genres_5].squeeze().tolist()
            labels[count,:] = new_label
            data[count,:,:,:] = temp_image
            data[count,:,:,:] = data[count,:,:,:]/255. # Normalize the data before saving
            full_poster['label_index'].loc[(full_poster['year']==temp_year) & (full_poster['movie']==temp_title)] = curr_array_index
            count += 1
            
        if genre_num == 4:
            new_label = df[genres_4].squeeze().tolist()
            labels[count,:] = new_label
            data[count,:,:,:] = temp_image
            data[count,:,:,:] = data[count,:,:,:]/255. # Normalize the data before saving
            full_poster['label_index'].loc[(full_poster['year']==temp_year) & (full_poster['movie']==temp_title)] = curr_array_index
            count += 1
            
        
    del data
    print(' In total ' + str(count_total) + ' posters were processed')
    print(' For the selected number of genres (' + str(genre_num) + '), ' + str(count) + ' posters were processed')
    return  np.asarray(labels), full_poster

if genre_num == 18:
    full_poster = full_poster_18
elif genre_num == 6:
    full_poster = full_poster_6
elif genre_num == 5:
    full_poster = full_poster_5
elif genre_num == 4:
    full_poster = full_poster_4
    
labels, full_poster = preprocess_image(genre_num, target_size_1, img_path, full_poster)

print('Images and labels extracted...')
# For csv files keep only those rows where we were able to match a poster
full_poster = full_poster[np.isfinite(full_poster['label_index'])]
print('The shape of the dataframe for ' + str(genre_num) + ' genres is: ' + str(full_poster.shape[0]))

# Save the labels and dataframe
if genre_num == 18:
    full_poster.to_csv('Data/full_poster_18.csv', index=False)
    pickle_out = open('Data/labels_18_ML.pickle','wb')
elif genre_num == 6:
    full_poster.to_csv('Data/full_poster_6.csv', index=False)
    pickle_out = open('Data/labels_6_ML.pickle','wb')
elif genre_num == 5:
    full_poster_5.to_csv('Data/full_poster_5.csv', index=False)
    pickle_out = open('Data/labels_5_ML.pickle','wb')
elif genre_num == 4:
    full_poster_4.to_csv('Data/full_poster_4.csv', index=False)
    pickle_out = open('Data/labels_4_ML.pickle','wb')
pickle.dump(labels, pickle_out)
pickle_out.close()

print('Everything saved!') 
