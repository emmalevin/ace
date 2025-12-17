
'''
Code to compute the KDE approximated PDF of our target variable (surface pressure in the Gulf of Mexico).
To do so we use all the training data.
'''


# For a given iteration of the algorithm

# list of time indices already in the training dataset 

# list of 10 highest acquisition function values
# list of time indices associated with the 10 highest acquisition function values

# loop through each possible point (EXCEPT THOSE ALREADY IN THE TRAINING DATASET) 
    # run inference on the point using the trained model
    # compute the acquisition function
    # if the acquisition function value is greater than the 10th highest value, add the time index to the list and remove the lowest value
    # making sure list always has 10 values


# then I have my list of 10 highest acquisition function values and the list of time indices associated with them
# add these points to the list of points in the training dataset 