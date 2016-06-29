# coding: utf-8

def getdata(studyname):
    import numpy as np
    import pandas as pd
    from os import listdir
    from numpy import genfromtxt
    if studyname == 'Utrecht':
        datadir = "/home/vincent/estep/data/utrecht"
        multivar = True #False # Are this Multivariate time series TRUE or FALSE
        multiclass = False
        Nfilesperelement = 4 #there are four files per patient
        labloc = "/home/vincent/estep/data/utrecht_labels.csv" # labels in column 1 (0), row 1 (1), or name of file
        idbloc = 2 # Id in column 1 (0), row 1 (1), seperate file (2), not applicable (3)
        labint = True # Is the label an integer?
        timecol = False # time series per column (True) or per row (False)
    elif studyname == 'UCR':
        datadir = "/home/vincent/estep/data/UCR_TS_Archive_2015/50words"
        multivar = False # Are this Multivariate time series TRUE or FALSE
        multiclass = False
        Nfilesperelement = 0 #all elements are in one merged file
        labloc = 0 # Classifcation labels in column 1 (0), row 1 (1), seperate file (2)
        idbloc = 3 # Id in column 1 (0), row 1 (1), seperate file (2), not applicable (3)
        labint = True # Is the label an integer?
        timecol = False # time series per column (True) or per row (False)
    else:
        raise ValueError(str(studyname) + ' is not a valid studyname')

    #============================================================================
    # Identify number of files and filetype based on datadir
    filenames = listdir(datadir)
    Nfiles = len(filenames) # number of files
    # Investigate what format the first file has by trying out a variety of reading attempts
    path = datadir + '/' + filenames[1]
    delimiter = [None,','] #possible delimiter values
    skiprows=[0,1]
    ntests = len(delimiter)*len(skiprows)
    df = pd.DataFrame(index=range(ntests),columns=['delimiter','skiprows','nrow','ncol','first cell'])
    cnt = 0
    for di in delimiter:
        for si in skiprows:
            try:
                F1 = np.loadtxt(fname=path,delimiter=di,skiprows=si)
                df['delimiter'][cnt] = di
                df['skiprows'][cnt] = si
                df['nrow'][cnt] = F1.shape[0]
                df['ncol'][cnt] = F1.shape[1]
                df['first cell'][cnt] = F1[0,1]
            except:
                df['delimiter'][cnt] = di
                df['skiprows'][cnt] = si
                df['nrow'][cnt] = 0
                df['ncol'][cnt] = 0
                df['first cell'][cnt] = 0
            cnt = cnt + 1
    # df is now a dataframe with information to help identify how the data should be loaded
    # load one file based on the extracted information on fileformat
    form = df[df.nrow == max(df.nrow)] # extraction procedure that resulted in the largest number of rows is the best
    if form.shape[0] > 1:
        form = df[df.ncol == max(df.ncol)] # extraction procedure that resulted in the largest number of columns

    if (form['delimiter'] == ',').bool():
        F2 = np.loadtxt(fname=path,delimiter=',',skiprows=int(form['skiprows']))
    else:
        F2 = np.loadtxt(fname=path,delimiter=None,skiprows=int(form['skiprows']))
    # Extract labels y and data X, and standardize shape of matrix
    # Extract data based on newly gained insight into fileformat and data structure
    if labint == True:
        labtype = 'int'
    else:
        labtype = 'str'
    if type(labloc) == str:
        #y = genfromtxt(labloc, delimiter=',',skip_header=1) # do we want numpy array or pd dataframe?
        y = pd.read_csv(labloc, sep=',',header=0)
        # within files
        if timecol == False:
            X = F2.transpose()
        else:
            X = F2
    elif type(labloc) == int:
        if labloc == 0:
            y = np.array(F2[:,0], dtype=labtype)
            X = F2[:,1:]
        elif labloc == 1:
            y = np.array(F2[0,:], dtype=labtype)
            X = F2[1:,:].transpose()

    return(X, y)
    # Reformat labels to be useful for Keras (in progress):
    if type(labloc) == str and multiclass == True and y.shape[0] != X.shape[0]:
        #filter y relevant for this filename, this is relevant for the london labels
        y = y[y['filename'] == filenames[1].strip('.csv')] #TO DO: make code less specific to London, e.g by
