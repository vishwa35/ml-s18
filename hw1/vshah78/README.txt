README.txt
-------------------------------------------

This code is to be run on Python 2.7 (I used 2.7.13)

Required libraries (you can do pip install ____ to install them provided you have pip, python's library manager)
scikit-learn
pandas
numpy
scipy
matplotlib
time

To produce learning curves for default parameters, go to the main method lines 371 and 383 and set cv=False

To run grid search cross validation to find the best hyper parameters, train the model with those, and the produce learning curves, go to the main method lines 371 and 383 and set cv=True

Then run the below command
python ml1.py

