Project 1 for IT348
by Dane Iwema

Welcome to my Project1 Naive Classifier!

to run the program type
    
    python3 BayesClassifier.py
    
into the terminal and it should start the program.

When you start the program the first thing you will see is the menu in the format:

1: Train Classifier
2: K-fold Validation
3: Confusion Matrix
4: Exit

Option:

Here you will input the number associated with the option, it will handle wrong inputs and reprint the menu untill
an option is selected.

Option 1: Train Classifier

    Here is where you will be prompted with for the meta data file and the training data file. These will be used
    to build and train the classifier using the Naive Bayes Classifier algorithm. The data structure I ended up using
    is a dictionary of lists of dictionaries. I did it this way to easily access each target class as the first dictionary,
    so then when an acc or unacc etc. appeared then I can easily use that directly from the file to access that index in
    the dictionaryl. then I used a list to iterate through all the other classes so for the car example 0=buying, 1=maint . . .
    untill 6=class. I used this for pythons list comprehension features and because we never actually see the words buying
    or maint to even use them as an index. Each of those lists hold dictionaries of the possible features of each class, and 
    the index here leads to the number of appearances in the file. So using it classifier["acc"][6]["acc"] will get me the total
    appearances of acc lines of data, or classifier["good"][3]["4"] will give me the total appearances of 4 person vehicles in good
    condition. 

Option 2: K-fold Validation

    In this option of the menu it will prompt the user again for the meta data file and the training data file. After that the user
    will also be prompted for a number that will be used as the k value for the k-fold Validation of the data pertaining to the model.
    k can be any number and splits the data into k seperate files. these files will be saved at the same location you are running the
    program and will be deleted after use. I chose to create new files since I did not want to clutter the ram so that you can hold as
    much data on the disk as long as you have double the space. This design was chosen to be as efficient as possible so that reading
    the data was only O(2n) where I need to read through each data entry twice, first time to add it to the new created file and to count
    its values, and the second time to use a validation to see if it made a correct prediction. Feel free to delete the files after they
    are not being used. the program will always just create more files or replace the them when a new k value is chosen. After performing
    the validation the accuracy of each fold and the average accuracy will be printed to the screen and then the menu will reappear allowing
    for another choice.

Option 3: Confusion Matrix

    This option will ask for 3 files this time in this order: Meta data, training data, and test data. From here it will build a model
    and train based off of the training data provided and then validate it using the test data. The output to the screen will be the 
    confusion matrix to get the true positives, true negatives, false positives, and false negatives for each prediction. The accuracy,
    precision, recall, and f1 are also printed.

Option 4: Exit

    This exits the program.
