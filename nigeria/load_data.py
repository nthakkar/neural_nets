'''Functions and tools for loading and preprocessing the data.
The functions take the data from the .csv file and manipulate it as a pandas dataframe and as
a numpy array.'''

from __future__ import print_function

import numpy as np
import pandas as pd

def ReadHeader(fname='headers.csv'):

	'''Read the header file, output is a dictionary which maps column ID to content description.'''

	header = {}

	## Loop through the file
	skip = set([0,1])
	header_file = open(fname,'r')
	for i,line in enumerate(header_file):

		## Skip lines in set skip
		if i in skip:
			continue

		## Otherwise, parse the line
		l = line.split(',')

		## And store it in the dictionary
		header[l[0].strip().lower()] = l[1]
	header_file.close()

	## two entries are missing from the header.csv file.
	## Here, I put in filler for those two to avoid issues later.
	header['var1'] = 'var1'
	header['caseid'] = 'case id'
	return header

def ReadData(fname='data.csv'):

	'''Function to read the data. Output is a pandas dataframe'''

	## Start by constructing the headers for the pandas data frame
	## Open the file, read the first line, strip the \n and \r, and parse.
	data = open(fname,'r')
	columns = data.readline().rstrip().lower().split(',')
	data.close()

	## Now use pandas to handle the rest
	df = pd.read_csv(fname,skiprows=0)
	df.columns = columns
	return df

def CleanData(df):

	'''Perform some cleaning operations to make the data easier to work with.'''

	## Compress the "i don't know" and "missing" categories
	df['h8'].replace([8.,9.],np.nan,inplace=True)
	df['h0'].replace([8.,9.],np.nan,inplace=True)
	df['h9'].replace([8.,9.],np.nan,inplace=True)


	## Make all the known ones equal
	df['h8'].replace([1.,2.,3.],1.,inplace=True)
	df['h0'].replace([1.,2.,3.],1.,inplace=True)
	df['h9'].replace([1.,2.,3.],1.,inplace=True)

	## Convert age to years to make it smaller numbers
	df['hw1'] = df['hw1']/12.

def LoadData(dataset='data.csv',header_file='headers.csv',resample=False):

	'''Wrapper for the functions above.

	The resample option uses the statistical weights in column v005 and samples (with replacement) a 
	fraction of the data provided by the user.'''

	header = ReadHeader(header_file)
	df = ReadData(dataset)

	## Resample using the weights column
	if resample:
		df = df.sample(frac=resample,replace=True,weights=df['v005'])

	## Simplify the vaccine columns of interest.
	CleanData(df)

	return header, df

def PortData(dataset='data.csv',header_file='headers.csv',
			predictors=['sstate','v106','v190','hw1','h9'],y=['h0']):

	'''Function to interface pandas with network class. 
	predictors is the subset of the data we're using based on the analysis in basic_data_analysis.py'''

	## Get the dataset as a pandas dataframe
	## Cut out the portion of the data we want,
	## including dropping nan's.
	header, df = LoadData(dataset,header_file)
	df = df[predictors+y].dropna()

	## Restructure the df to have dummy variables for each 
	## catagorical variable value. This is done automatically
	## in the statsmodels implementation, but here we have to do it
	## explicitly.
	dummy_predictors = []

	## Loop through each of the predictors
	for predictor in predictors:
		name = df[predictor].dtype.name
		type_check = (name.startswith('int')) or (name.startswith('float'))

		## If the type isn't catagorical, then there's no need to change anything.
		if type_check:
			dummy_predictors.append(predictor)
			continue

		## But if it is, we expand our data frame to have a 1 or 0 column for each 
		## catagory.
		else:
			## Get the different catagories in alphabetical order
			catagories = df[predictor].value_counts().sort_index(axis=0).index.tolist()
			for catagory in catagories:
				## We convert from bool to float for ease later.
				df[catagory] = 1.*(df[predictor]==catagory) 
				dummy_predictors.append(catagory)

	dummy_df = df[dummy_predictors+y]

	## Split the dataset into training (60%), testing (20%), 
	## and validation (20%). This is done with random resampling.
	train_xy, validate_xy, test_xy = np.split(dummy_df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

	## Convert to numpy
	train_x, train_y = train_xy[dummy_predictors].as_matrix(), train_xy[y].as_matrix()
	validate_x, validate_y = validate_xy[dummy_predictors].as_matrix(), validate_xy[y].as_matrix()
	test_x, test_y = test_xy[dummy_predictors].as_matrix(), test_xy[y].as_matrix()


	## Reshape appropriately
	## Just do it one by one
	## reshaping and recasting each.
	## Training
	N = len(dummy_predictors)
	training_inputs = [np.reshape(x, (N, 1)) for x in train_x]
	training_results = [OneHot(y) for y in train_y]
	training_data = zip(training_inputs, training_results)

	## Validation
	validation_inputs = [np.reshape(x, (N, 1)) for x in validate_x]
	validation_data = zip(validation_inputs, validate_y)

	## Test
	test_inputs = [np.reshape(x, (N, 1)) for x in test_x]
	test_data = zip(test_inputs, test_y)

	return (training_data, validation_data, test_data)

def OneHot(j,n=2):

	'''Returns a vector of len = n with j = 1 and all others = 0'''

	e = np.zeros((n,1))
	e[int(j)] = 1
	return e

if __name__ == "__main__":

	df = ReadData()

	#print df.isnull().sum()['hw1']
	hist = df['hw1'].value_counts()
		
	#print len(df[df['hw1']==0.0]['b5'])#.value_counts()

	print(df[['v133','h8']].corr())








