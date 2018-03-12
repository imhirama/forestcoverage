
#Imports
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.grid_search import GridSearchCV
from random import sample
import pandas as pd
import matplotlib.pyplot as plt 
import time
import sys
import logging
import csv
import warnings



#Hide further depreciation warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

#Optional: set vocal settings
#logging.basicConfig(
            #format="%(message)s",
            #level=logging.INFO,
            #stream=sys.stdout)




################ Prepare Data ############################

#Import data for four wilderness areas
area1 = pd.read_csv("forest_area_1.csv",
                   delimiter = ',', header=0)
                   
area2 = pd.read_csv("forest_area_2.csv",
                   delimiter = ',', header=0)                   
                   
area3 = pd.read_csv("forest_area_3.csv",
                   delimiter = ',', header=0)

area4 = pd.read_csv("forest_area_4.csv",
                   delimiter = ',', header=0)

print "Area 1:", area1.shape
print "Area 2:", area2.shape
print "Area 3:", area3.shape
print "Area 4:", area4.shape
                   

area1.head()



#Choose the dataset to be used for current iteration
data = area2                 

#Set dependent variable: Cover Type
dep_vars = ['Cover_Type']                   
dep_data = data[dep_vars]

#Set independent variables: all except Cover Type
indep_data = data
del indep_data["Cover_Type"]



################ Tuning Method 1: For-Loop ############################

#Use for varying a single parameter

#Set up parameters and result lists for for-loop testing
results =[]
times = []
levels = [1,10,50]

#For-loop testing. Set parameter being tested as T. Here, epochs (n_iter) is varied. 
for T in levels:
    print 'Starting T = ', T
    start_time = time.time()  

    #Train/Test Split                
    X_train, X_test, Y_train, Y_test = train_test_split(indep_data, dep_data, test_size=.5, random_state=2016)
    Y_train = Y_train.as_matrix()
    Y_test = Y_test.as_matrix()
    
    X_trainn = preprocessing.normalize(X_train, norm='l2')
    X_testnn = preprocessing.normalize(X_test, norm='l2')

    X_trainn = preprocessing.scale(X_trainn)
    X_testnn = preprocessing.scale(X_testnn)
    
    #Build model
    clsfr = Classifier(
    	layers=[  
           Layer("Rectifier", units=30),  
           Layer("Rectifier", units=30),
           Layer("Softmax")], 
           learning_rate=.01, 
           learning_rule='sgd',
           random_state=100,
           n_iter= T, # <---- Parameter being varied 
           #learning_momentum=T,
           #valid_size=T,
           verbose = False)
    #Fit model
    model1=clsfr.fit(X_trainn, Y_train)
    
    #Predictions
    y_hat=clsfr.predict(X_testnn)
    
    #Print scores
    print 'sgd test %s' % accuracy_score(y_hat,Y_test)
    end_time = time.time()
    times.append((end_time - start_time))
    results.append(accuracy_score(y_hat,Y_test))
    print 'time: ', (end_time - start_time)
    print



# Plot results (accuracy & runtime)
get_ipython().magic(u'matplotlib inline')

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(levels, results, 'g')
ax2.plot(levels, times, 'b')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy', color='g')
ax2.set_ylabel('Run Time (Seconds)', color='b')

plt.title('Number of Epoch vs. Accuracy and Run Time')

plt.show


# <h3>Grid Search</h3>

# <i><b>Note:</b> To save time building the notebook, in the example below only three parameters were varied, with a limited number of levels. Normally, all parameters could be varied.</i>



################ Tuning Method 2: Grid Search ############################

#Note: another similar option that could save time is RandomSearch

#Prepare data
Y_train = Y_train.reshape(len(Y_train),)

#Run grid search
print
print
print "Starting Grid Search"
gs = GridSearchCV(clsfr, param_grid={
    'learning_rate': [.005,.01],#[.03,.05,.075,.1]
    'hidden0__units': [10,100], #range(10,100,10),
    'hidden1__units': [10,100], #range(10,100,10),
    'hidden1__type': ["Rectifier"],#,"Tanh"],
    'hidden0__type': ["Rectifier"],#,"Tanh"]
    'n_iter': [5], #range(5,200,20)
    'valid_size':[.25]
    },
    scoring = 'accuracy', verbose = 1, cv = 2)
gs.fit(X_trainn, Y_train)



#Display best parameters and result
print("Best: %f using %s" % (gs.best_score_, gs.best_params_))



##### Output Grid Search results to a csv file

#Prepare array
arrayofdata = []
labels = gs.grid_scores_[0][0].keys()
labels.append('mean')
arrayofdata.append(labels)

#Add results to array
for line in gs.grid_scores_:
    rowdata = []
    for key in gs.grid_scores_[0][0].keys():
        rowdata.append(line[0][key])
    rowdata.append(line[1])
    arrayofdata.append(rowdata)

#Write to csv
with open('gridsearch_results.csv', 'wb') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile)
    for row in arrayofdata:
        thedatawriter.writerow(row)    
        
print "Done writing results to CSV file."

