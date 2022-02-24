from ADA.settings import MEDIA_ROOT
from collections import Counter
import os

import pandas as pd

import operator
import math

import numpy as np
from numpy import mean, std

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer,OrdinalEncoder
from category_encoders import TargetEncoder

from scipy.stats import chi2_contingency
import scipy.stats as stats

import json

RESPONSE = ""
FAIL = "F"
PASS = "P"

def insert_transformer(pipeline, step):
    no_of_steps = len(pipeline.steps)
    pipeline.steps.insert(no_of_steps ,step)

    return pipeline

def preprocessing_pipeline(data, dataset, training = False):
    ## IMPROVE CODE BY PASSING IN ALL OF DATASET ATTRIBUTES 
    global RESPONSE
    RESPONSE = dataset.get('response')
    print("FAIL ", "PASS " , dataset.get('FAIL'), dataset.get('PASS'))
    data[RESPONSE].replace({dataset.get('PASS'): "P", dataset.get('FAIL'): "F"}, inplace=True)

    if(training): # binarize y labels and encode features
        feature_list = json.loads(dataset.get('features')) if isinstance(dataset.get('features'),str) else dataset.get('features')
        preprocessing_pipe2 = Pipeline([  ('encoder', Encode_data(y=data[RESPONSE]))]) 
        cleaned_data = preprocessing_pipe2.fit_transform(data[feature_list])
        
    else:
        print(" Dont preprocess again ")
        new_path = os.path.join(MEDIA_ROOT, "def_regrouped.csv")
        data_regrouped = pd.read_csv(new_path)
        print("reading ", data_regrouped.shape)

        preprocessing_pipe2 = Pipeline([
        ('data preparation', Data_preparation(level = dataset.get('level') , data_regrouped =data_regrouped )),
        ('lower cardinality', Lower_cardinality(kneed= False)),
        ])
        
        if(dataset.get('pearson') or dataset.get('remove_association')):
            insert_transformer(preprocessing_pipe2, ['pearson test',pearson_test(y=data[RESPONSE], tf = True)]) if dataset.get('pearson') else None
            insert_transformer(preprocessing_pipe2, ['full association', Remove_fullAssociation()]) if dataset.get('remove_association') else None
        
        insert_transformer(preprocessing_pipe2, ['encoder', Encode_data(y=data[RESPONSE])])

    
        print('==== PIPELINE steps ' , len(preprocessing_pipe2.steps) , preprocessing_pipe2.steps )

        print(">>before", data.shape)

        cleaned_data = preprocessing_pipe2.fit_transform(data)
    
    print(">>after",cleaned_data.columns)
    return preprocessing_pipe2, cleaned_data

class Data_preparation():
    
    """ Prepare data by filter by the levels of the features (ex. Primary or Secondary)

    Parameters
    ----------
    data_regrouped      :       DataFrame
        The dataframe imported from anothere excel sheet that describes the levels of each features
        whether it is primary or secondary

    level               :        String 
        Filters the dataset according to either P for Primary or S for Secondary level features

    """
        
    def __init__(self,level = 'P', data_regrouped = None):
        self.data_regrouped = data_regrouped
        self.level = level
        
    def transform(self, X, y=None, **fit_params):
        typeds = ["DEF"]
        self.data_regrouped = self.data_regrouped[self.data_regrouped['GRP'].isin(typeds) & self.data_regrouped['P/S'].isin([self.level])]
        keep_cols = list(self.data_regrouped.Columns)
        
        keep_cols = list(set(X.columns).intersection(set(keep_cols)))
        keep_cols.append(RESPONSE)
        new_data = X[keep_cols]  
        print("Keep primary columns data shape " , new_data.shape)
        index_list = np.arange(1,len(keep_cols)+1, 1)
        self.keep_cols = pd.DataFrame({'Filtered Features': keep_cols}).sort_values(by='Filtered Features')
        self.keep_cols.set_axis(index_list, axis='index',  inplace=True)

        # new_path = os.path.join(MEDIA_ROOT, "def_regrouped.csv")
        # new_data.to_csv(new_path)

        return new_data
    
    def fit(self, X, y=None, **fit_params):
        return self



class Lower_cardinality():
    
    """ Prepares the data by removing features with cardinalities of 1 and ID features (cardinalities are equal to the samples)
    ----------
    max_cat : int, optional
        the maximum number of categories all the features should have in the dataset. features that
        have more than the max number will be removed
    kneed : bool, optional 
        if kneed is True, then the kneed algorithm is used to eliminate high cardinality features. if kneed is False, 
        kneed algorithm is not used to eliminate the features  
        
    """

    
    def __init__(self, max_cat = None, kneed = True):
            self.max_cat = max_cat
            self.kneed = kneed

            
    def transform(self, X, y=None, **fit_params):
        remove_cols= ['ACTUAL_DATE',RESPONSE ]

        if(self.max_cat):  ##if there is a "preset" max # of cats
            max_cat = self.max_cat
        else:
            max_cat = X.shape[0]
            
        collist = []
                
        for c in X.columns: 
            no_of_cat = len(X[c].value_counts().index)
            if no_of_cat > 1 and no_of_cat < max_cat: ##columns w more than 1 category and not ID variables

                collist.append(c)

                X.loc[:, c] = X.loc[:, c].astype('str')

        for i in remove_cols:
            if i in collist:
                collist.remove(i)

    
        if(self.max_cat or self.kneed):   ##if there is a "preset" max # of cats, no need to use kneed
            data_category = self.filter_n_cat(X[collist])
        else:  ## do not use kneed 
            data_category = X[collist]
        
        print("lower cardinality data shape ", data_category.shape)
        return data_category


    def fit(self, X, y=None, **fit_params):

        return self

    def filter_n_cat(self, data_category):
        
        ncat_df = pd.DataFrame(data_category.nunique(), columns=['N_cat'])
        ncat_df = ncat_df.sort_values(by= ['N_cat'])
        r = len(data_category.columns)
        lis = ncat_df['N_cat']

        kneedle = KneeLocator(range(0,r), lis, S=1.0, curve="convex", direction="increasing", online="True")
        knee_val = kneedle.knee
        print("Columns are filter down to after Kneed Algorithm: ", knee_val)
        
        x_columns = ncat_df[:knee_val].index
        filter_cat = data_category[x_columns]
    
        return filter_cat
        


class pearson_test():

    """ Does a pearson test for all features and filter out irrelevant features that do not meet the 2 conditions of 
         (1) have less than 20% of expected numbers that are less than 5 and (2) p value must be less than the alpha

        ----------
        y : Series
            the response column

        tf : bool, optional 
            if tf is True, then the output dataset is filtered by relevant features considered by pearson test.
            if tf is False, then the output dataset remains the same when it was inputted
    """
    
    def __init__(self, y = None, tf = True):
        self.y = y
        self.tf = tf
        
    def cramers_v(self, chi2 , table):
        minDim = min(table.shape) -1
        n = np.sum(np.array(table))
        
        #calculate Cramer's V 
        V = np.sqrt((chi2/n) / minDim)

        return V
    
    def transform(self, X , y= None, **fit_params): 
        Y =  self.y

        p_values = []
        cramer_values = []
        col_lis = []
        fisher_values = []
        removed_cols = []

        for cat_col in X.columns :
            table = pd.crosstab( X[cat_col], Y) 

            try:


                chi2, p, dof, expected = chi2_contingency(table, correction=False)

                prob = 0.95
                alpha = 1.0 - prob
                if p <= alpha:
                    N  = len(expected[0]) * len(expected)
                    condition = len(expected[expected<=5])
                    percent = condition / N
                    
                    
                    if(X[cat_col].nunique() == 2):    ## FOR SMALLER SAMPLES IN 2X2 contigency tables, fisher is calculated
                        oddsratio, pvalue = stats.fisher_exact(table)

                        col_lis.append(cat_col)
                        fisher_values.append(pvalue)
                        p_values.append(p)
                        
                        
                    elif(percent <= 0.2):    ## if 20% of the expected numbers are less than 5, still considered relevant
                        
                        col_lis.append(cat_col)
                        p_values.append(p)
                        fisher_values.append(-1)

        

                    else:                   ## features that did not meet the two conditions and are removed
                        removed_cols.append(cat_col)


            except:
                continue

        # p_values = pd.DataFrame(data = {'Pearson': p_values, 'Fisher': fisher_values} , index = col_lis)
        p_values = pd.DataFrame(data = {'Pearson': p_values} , index = col_lis)
        p_values = p_values.sort_values(by = ['Pearson'], ascending= True)

#         display(p_values.head(50))
        
        self.p_values = p_values
        self.removed_cols = removed_cols
        chi_cols = p_values.index
        self.chi_cols = p_values.reset_index()

        
        chi_X = X[chi_cols]
                
        print("NUMBER OF PEARSON FILTERED FEATURES : " , len(chi_cols))
        print("PEARSON FILTERED FEATURES : " , col_lis)
        print("After pearson data shape", chi_X.shape)
        
        
        if(self.tf):
#             plotdata = pd.DataFrame(chi_X.nunique(), columns = ['n_cat'])
#             plotdata.sort_values(by='n_cat', ascending = False).plot(kind="bar",figsize=(30,20),fontsize=20)
            return chi_X
        else:
#             pl  = pd.DataFrame(X.nunique(), columns = ['n_cat'])
#             pl.sort_values(by='n_cat', ascending = False).plot(kind="bar",figsize=(30,20),fontsize=20)
            return X
        

    def fit(self, X, y=None, **fit_params):

        return self

   

class Remove_fullAssociation():
    
    """ Determining pairs of fully associated features and removing one
    ----------
   

    """

    def __init__(self):
        print("removing fully associated features")
        self.kept_list = []
        self.remove_list = []
        self.pair_list = []
        self.chart = pd.DataFrame()
    
    def transform(self, X, y=None, **fit_params):

        grid = []
        for x in X.columns:
            row = []
            for y in X.columns:
                row.append(self.theil_u(X[x], X[y]))
            grid.append(row)

        assocMap = pd.DataFrame(grid, index = X.columns, columns = X.columns)
        keptlist, removelist, cleaned_data_X = self.remove_full_assoc_attrv3(assocMap, X)

        
        self.chart = pd.DataFrame({'Pairs':  np.arange(1,len(keptlist)+1, 1), 'Feature A': keptlist, 'Feature B' : removelist })


        return cleaned_data_X
    
    def fit(self, X, y=None, **fit_params):

        return self
    
    
    def conditional_entropy(self, x,y):
        # entropy of x given y
        y_counter = Counter(y)
        xy_counter = Counter(list(zip(x,y)))
        total_occurrences = sum(y_counter.values())
        entropy = 0
        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurrences
            p_y = y_counter[xy[1]] / total_occurrences
            entropy += p_xy * math.log(p_y/p_xy)
        return entropy

    def theil_u(self, x,y):
        s_xy = self.conditional_entropy(x,y)
        x_counter = Counter(x)
        total_occurrences = sum(x_counter.values())
        p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
        s_x = stats.entropy(p_x)
        if s_x == 0:
            return 1
        else:
            return (s_x - s_xy) / s_x

    def remove_full_assoc_attrv3(self, assocMap, data):
     

        x_cols = assocMap.columns
        y_cols = assocMap.columns

        for x in assocMap.columns:

            if x in self.remove_list: ##if attribute is alrd removed , dont need to compare with it
                continue

            for y in assocMap.columns:

                if y in self.remove_list:  ##if attribute is alrd removed , dont need to compare with it
                    continue

                if(x != y):
                    if(assocMap[x][y] == 1 and assocMap[y][x] == 1):
                            self.pair_list.append((x,y))
                            self.kept_list.append(x)
                            self.remove_list.append(y)
                            


       
        data = data.drop(columns = self.remove_list)
        print("DATA ASSOC ", data.columns)
        return self.kept_list, self.remove_list, data


class Encode_data():
    
    """ Part of the data pipeline to encode the data / label encode the response column and target encode 
    the rest of the features 
    ----------
    label_encoder : LabelEncoder
        label encoder that encodes Pass/Fail values to 0/1
    t_encoder : TargetEncoder 
        target encoder encodes all categorical values according to the response variable 

    """


    def __init__(self, y = None):
        self.label_encoder = LabelBinarizer()
        self.t_encoder = TargetEncoder()
        self.y = y
        
        
    def transform(self, X, y=None, **fit_params):
    
        ### for label encoding y
        y = list(self.y)
        # print('y to be transformed using label encoder: ', y)
        le = self.label_encoder.fit([PASS, FAIL])
        self.label_encoder.classes_ = [PASS, FAIL]
        self.y = self.label_encoder.transform(y)
        # print(self.y)

        tdata_ord = self.t_encoder.fit_transform(X, self.y)
        tdata_category =  pd.DataFrame(tdata_ord, columns = X.columns)

        X = tdata_category
        
        ## drop rows that have all NULL values
        X = X.dropna(axis=0, how='all')

        Y = pd.DataFrame(self.y, columns = [RESPONSE])

        data_encoded = pd.merge(X, Y, left_index=True, right_index=True)
        
        print("Encoded data shape", data_encoded.shape)
        return data_encoded


    def fit(self, X, y=None, **fit_params):
        return self