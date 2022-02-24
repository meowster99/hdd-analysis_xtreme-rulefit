from pickle import TRUE
from tracemalloc import StatisticDiff
from sklearn import metrics
from sklearn.metrics import roc_auc_score,brier_score_loss, f1_score
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso
import pandas as pd
from numpy import mean
import statistics
import itertools
from typing import Dict
import math
import re
import copy
from matplotlib import pyplot
import time
from sklearn.exceptions import ConvergenceWarning
import warnings
from collections import Counter

class Rule:
    """ An object modeling a logical rule and add factorization methods.
    It is used to simplify rules and deduplicate them.
    Parameters
    ----------
    rule : str
        The logical rule that is interpretable by a pandas query.
    args : object, optional
        Arguments associated to the rule, it is not used for factorization
        but it takes part of the output when the rule is converted to an array.
    """
    def __init__(self, rule, args=None, support=None):
        self.rule = rule
        self.args = args
        self.support = support
        self.terms = [t.split(' ') for t in self.rule.split(' and ')]
        self.agg_dict = {}
        self.factorize()
        self.rule = str(self)
    
    def factorize(self) -> None:
        for feature, symbol, value in self.terms:
            if (feature, symbol) not in self.agg_dict:
                if symbol != '==':
                    self.agg_dict[(feature, symbol)] = str(float(value))
                else:
                    self.agg_dict[(feature, symbol)] = value
            else:
                if symbol[0] == '<':
                    self.agg_dict[(feature, symbol)] = str(min(
                        float(self.agg_dict[(feature, symbol)]),
                        float(value)))
                elif symbol[0] == '>':
                    self.agg_dict[(feature, symbol)] = str(max(
                        float(self.agg_dict[(feature, symbol)]),
                        float(value)))
                else:  # Handle the c0 == c0 case
                    self.agg_dict[(feature, symbol)] = value

class Rule_Filter():
    """
        This rule filter class creates the Logistic Regression model and filters to the relevant rules

        Parameters
        ----------
            
            X: DataFrame
                the dataframe of the extracted rules where each datapoint will have 1 or 0 
                where 1 = the data point belongs to the rule and 0 = does not belong to the rule
            Y : Series
                the data of the labels in the response variable
            rulefit: Rulefit
                Rulefit instance model
            prediction_task: String
                Default is classification
            max_rules: Int
                the maximum number of rules 
            coefs_array: List
                list of coefficients of each rule of every iteration
            rule_count_array: List
                list of number of rules of every iteration
            nonzero_combo_arr: List
                list of strings of the combinations (basically features of the rules) of every iteration
            object_rules_arr : List             
                list of Rules object with positive coefficients of every iteration
            alpha_arr_iterations: List
                list of alphas of every iteration 
            og_cv_scores: 2D Array
                array of the list of metric scores of every iteration       
            alpha_df: DataFrame
                a whole summary of each iteration and its information including the
                alpha, coefficients, metric scores, number of rules, the strings of rule and 
                the rule objects
            variance_df: DataFrame
                    showcases the information of every alpha such as
                   the variance of the combinations, number of unique rules, number of iterations
            linear_model_info : Dict
                    the dictionary form of variance_df
            alpha_info_list: List
                list of Alpha objects containing information of
                each alpha such as the variance of the combinations,
                number of unique rules, number of iterations
            list_of_metrics: List
                list of metrics to evaluate the logistic regression
                'f1_macro','f1','roc','log_loss','brier_score'
            iterations_w_rules_alpha_dict: Dict
                maps the unique alpha and the count of iterations that have rules
    """
    def __init__(self, model , maxrules, X , Y, prediction_task='classification' , alphas = None, metric = False):
        self.linear_model_info = {'alpha_arr2':[], 'vari_arr':[] , 'no_of_combos_per_alpha': [] , 'no_of_iterations' : []}
        self.rulefit = model
        self.coefs_array = []
        self.rule_count_array= []
        self.nonzero_combo_arr = []
        self.object_rules_arr = []
        self.iterations_w_rules_alpha_dict = {}
        self.alpha_arr_iterations = []
        self.alpha_df = pd.DataFrame()
        self.alpha_info_list = []
        self.list_of_metrics = ['f1_macro','f1','roc','log_loss','brier_score']
        self.og_cv_scores = np.vstack(np.zeros(len(self.list_of_metrics)))
        self.prediction_task = prediction_task
        self.max_rules = maxrules
        if alphas == None:
            self.alphas = self.generate_alphas()
        else:
            self.alphas = alphas
        self.X = X
        self.y = pd.DataFrame(Y) if not isinstance(Y, pd.core.series.Series) else Y
        self.metric = metric
        
    def generate_alphas(self):
        """
        Generate a list of alphas to run through the best alpha for the logistic regression
        ----------
        Returns
            alphas: List
            list of alphas - a hyperparameter of logistic regression
        """
        if self.prediction_task == 'regression':
            alphas = _alpha_grid(self.X, self.y)
        elif self.prediction_task == 'classification':
            #get 30 alphas in linear space between 0.1 to 10 
            alphas = [alpha for alpha in np.linspace(0.1, 10, num=30)]  
        return alphas


    def switcher_metric(self, case):
        """
        Direct to the right metric function to evaluate the predicted values
        """
        return {
            "f1_macro":f1_score,
            "f1":f1_score,
            "roc":roc_auc_score,
            "brier_score":brier_score_loss,
            "log_loss": metrics.log_loss,
        }.get(case, metrics.log_loss)  # you can pass

    def get_metric_scores(self, metric_arr, y_test,predicted_y, y_pred_proba):
        """
        Evaluate the model with metrics
       
        Returns
        -------
            score_arr: array
                an array of the metric score 
        """
        score_arr = np.array([])
        for m in metric_arr:
            if m == 'f1_macro':
                score = self.switcher_metric(m)(y_test, predicted_y,average='weighted')
            elif m == 'f1':
                score = self.switcher_metric(m)(y_test, predicted_y)
            else:
                score = self.switcher_metric(m)(y_test, y_pred_proba)
            score_arr = np.append(score_arr, score)
        return score_arr


    def train_logreg_range_of_iterations(self, range_of_iterations= [150], max_range = None):
        """
          This function trains logistic regression within a range of interations to check
          which range if optimal. Default is one range of iteration of 100 iterations 
          
          Parameters
          ----------
              max range: Int
                  the maximum iterations that user wants to reach
                  then a range of iterations of a 100 steps are created to reach the max
                  number of iterations. Default is None
        """
        if(max_range):
            print("max range: ", max_range)
            multiplier = len(np.arange(100, max_range, 100))
            range_of_iterations.extend([100] * multiplier) 
            print(range_of_iterations)
        else:
            print("no max range set")
        for iterations_per_alpha in range_of_iterations:  # for every 100 iterations 
            start_time = time.time()

            with warnings.catch_warnings(): # ignore convergence warning
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                for alpha in self.alphas: # goes through each alpha to train the logistic regression
                    self.train_logreg(alpha = alpha, iterations_per_alpha = iterations_per_alpha, dynamic=False) #TODO: toggle dynamic tuning of LR hyperparameters
  
            print('Training time taken: ', time.time() - start_time)
            try:
                # create the alpha dataframe with information of each iteration 
                if(self.metric):
                    alpha_df = pd.DataFrame({"alpha": self.alpha_arr_iterations ,
                                             "coefs": self.coefs_array, 
                                             "f1 scores": self.og_cv_scores[1] ,
                                             "f1 macro": self.og_cv_scores[0] , 
                                             "roc": self.og_cv_scores[2], 
                                             "log loss": self.og_cv_scores[3],
                                             "brier loss": self.og_cv_scores[4], 
                                             "no_of_rules":self.rule_count_array ,
                                             "combinations" : self.nonzero_combo_arr,
                                            'og_rules':self.object_rules_arr })
                else:
                     alpha_df = pd.DataFrame({"alpha": self.alpha_arr_iterations ,
                                             "coefs": self.coefs_array, 
                                             "no_of_rules":self.rule_count_array ,
                                             "combinations" : self.nonzero_combo_arr,
                                            'og_rules':self.object_rules_arr })
                self.alpha_df = alpha_df[alpha_df.no_of_rules != 0].reset_index(drop=True)
                ## the whole results table -- unique combinations for each alpha 
                results = pd.DataFrame()
                ## remove duplicate combinations
                results = self.remove_duplicate_combos()
                ## show results per alpha
                if iterations_per_alpha > 1: ## iterations per alpha is set by user
                    self.variance_df = self.results_per_alpha(results,iterations_per_alpha)
                    self.best_alpha = self.find_best_alpha()
                else:
                    self.variance_df = self.results_per_alpha(results,iterations_per_alpha)
            except Exception as e:
                print(e)
            
    def train_logreg(self, iterations_per_alpha = 1 , alpha = 0.1, cv = 3, random_state=0, dynamic = False):
        """
        Train logistic regression model with one alpha 

        Parameters
        ----------
            iterations_per_alpha: Int
                number of iterations to train logistic regression model 
            alpha: Float
                the hyperparameter of the logistic regression model
            cv: Int
                split dataset into cv consecutive folds 
            random_state: Int
                random state of the logistic regression model
        """
        print("ALPHA >>>>> " , alpha ) 
        norules_iteration = 0  # number of iterations with no rules
        kf = StratifiedKFold(cv)
        
        # DYNAMIC SETTING
        if dynamic:
            solver_list = ['liblinear', 'saga']
            penalty_list = ['l1', 'l2']
            params = dict(solver=solver_list, penalty=penalty_list)
            crossvalidator = RepeatedStratifiedKFold(n_repeats=2, n_splits=2)
            model = LogisticRegression(C=1/alpha)
            grid_search = GridSearchCV(model, params, scoring='neg_log_loss', cv=crossvalidator, n_jobs=1)
            grid_search.fit(self.X,self.y)
            print('Best params: ', grid_search.best_params_)
            print('Best score:', grid_search.best_score_)
        else:
            print("No tuning of Logisitc Regression >>>>>>>")
        for i in range(0,iterations_per_alpha):
            if self.prediction_task == 'regression':
                m = Lasso(alpha=alpha, random_state=random_state)
            else: ## classification task uses logistic regression
                if dynamic:
                    m = LogisticRegression(**grid_search.best_params_) # dynamic
                else:
                    m = LogisticRegression(C=1/alpha, solver='liblinear', penalty='l2') # https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when
            if(self.metric):
                ## initialise all metric scores 
                mse_cv = 0
                roc_cv = 0
                log_loss_cv = 0
                brier_score_cv = 0
                f1macro_score_cv = 0
                ## the array of all metric scores (metric's score board)
                og_score_arr = np.vstack((np.zeros(len(self.list_of_metrics))))
            ## train and test according to splits 
            for train_index, test_index in kf.split(self.X,self.y):   
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                m.fit(X_train, y_train)
                predicted_y = m.predict(X_test)
                y_pred_proba = m.predict_proba(X_test)[::,1]
                if self.prediction_task == 'regression':
                    t = (predicted_y - y_test)  
                    mse_cv += np.mean(t ** 2)  ## mse does not work for logistic regression, averaging the 0 or 1s being predicted
                else:
                    if(self.metric):
                        # get metric scores for the predicted y 
                        score_arr= self.get_metric_scores(self.list_of_metrics, y_test,predicted_y, y_pred_proba)
                        ## add into the metric's score board
                        og_score_arr= og_score_arr + np.vstack(score_arr)   
            m.fit(self.X, self.y)
            coef_ = m.coef_.flatten()
            #percentage of data points to which the decision rule applies 
            support = np.sum(self.X[:, -len(self.rulefit.extracted_rules):], axis=0) / self.X.shape[0]  
            ## allow rules with coefficient above 0 -> rules above 0 determines the rule's ability to affect class 1 ('Fail')
            coef_zero_threshold = 1e-6 / np.mean(np.abs(self.y))
            rule_count = np.sum(coef_ > coef_zero_threshold) 
            if rule_count > self.max_rules:
                print("rule_count > max_rules")
                break
            elif rule_count == 0: ## when there are no rules with coefficient above 0 in the iteration
                norules_iteration+=1
                continue
            ## global array of alphas for every iteration 
            self.alpha_arr_iterations.append(alpha)
            nonzero_rules = []
            rules = []
            for r, w, s in zip(self.rulefit.extracted_rules, coef_[-len(self.rulefit.extracted_rules):], support):
                if w > coef_zero_threshold: ## if coefficient of rule is larger than the zero threshold
                    rules.append(r) ## add rules into the rules array
                    nonzero_rules.append(Rule(r, args=[w], support=s))   ## add Rules Objects into nonzero rules array
            ## replace feature placeholders to feature names of every Rule Object
            self.rulefit.rulefit_model.rules_ = [
                    self.replace_feature_name(rule, self.rulefit.rulefit_model.feature_dict_) for rule in nonzero_rules]
            ## array of importance of the rules  
            importance_array = []
            combo_arr = []
            non_zero_coefs = []
            ## iterate through all NON ZERO rules 
            for idx in range(0, len(self.rulefit.rulefit_model.rules_)):
                r = self.rulefit.rulefit_model.rules_[idx]
                coef = r.args[0]
                non_zero_coefs.append(coef)
                combo = []                
                ## extract the features of each rule => creating combinations
                for feature, sym in r.agg_dict:
                    combo.append(feature)
                combo_arr.append(combo)                
                ## refer to https://christophm.github.io/interpretable-ml-book/rulefit.html for feature importance formula
                importance = abs(coef) * (r.support * (1 - r.support)) ** (1 / 2)
                importance_array.append(importance)
            combination_df = pd.DataFrame({'combinations': combo_arr, 'coef': non_zero_coefs, 'importance':importance_array})
            combination_df = combination_df.sort_values(by = 'importance', ascending = False)
            arranged_combos = combination_df.combinations.to_list()  ##arranged list is sorted by importance
            ## add information from the iteration into the global lists 
            self.coefs_array.append(non_zero_coefs)  ## coefs of each rule of all iterations
            self.nonzero_combo_arr.append(arranged_combos)    ##extracted combinations of each iteration
            self.rule_count_array.append(len(arranged_combos)) ## number of rules of each iteration             
            if(self.metric):
                ## average metric score board of the iteration
                og_avg_score = og_score_arr/cv
                if (np.all(self.og_cv_scores == 0)):  ## for the first iteration
                    self.og_cv_scores = np.sum([self.og_cv_scores, og_avg_score], axis = 0)
                else:
                    self.og_cv_scores = np.concatenate([self.og_cv_scores, og_avg_score], axis = 1)
            ## add the list of Rule Objects of the iteration into a global list
            self.object_rules_arr.append(self.rulefit.rulefit_model.rules_ )     
        iterations_w_rules = iterations_per_alpha - norules_iteration
        self.iterations_w_rules_alpha_dict[alpha] = iterations_w_rules  ## dictionary of the number of iterations with rules for each alpha
        
    def replace_feature_name(self, rule: Rule, replace_dict: Dict[str, str]) -> Rule:  
        """ Imodel Library Function
            To replace each feature placeholder to the original feature name in the Rule Object
        """ 
        def replace(match):
            return replace_dict[match.group(0)]
        rule_replaced = copy.copy(rule)
        rule_replaced.rule = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replace_dict), replace, rule.rule)
        replaced_agg_dict = {}
        for feature, symbol in rule_replaced.agg_dict:
            replaced_agg_dict[(replace_dict[feature], symbol)] = rule_replaced.agg_dict[(feature, symbol)]
        rule_replaced.agg_dict = replaced_agg_dict
        return rule_replaced

    def display_metrics_table(self):
        """ 
            Display a dataframe of the scores of all metric scores by unique alpha 
        """ 
        if(self.metric):
            alpha_avg  = self.alpha_df.groupby("alpha").mean()
            alpha_avg = alpha_avg.style.apply(self.apply_formatting)
            # display(alpha_avg)
        else:
            print("Metric tracking was not turned on when training the model. Please run again by setting metric = True")           
    
    def remove_duplicate_combos(self, max_rank = 3):   
        """ 
            Remove duplicated rules and creating a matrix of 
            combinations with ranks 1-3 for each iteration
            -----
            Parameters
                max_rank: Int
                    the maximum rank of the rules and extract the top X unique combinations for each iteration
            
            Return 
                unique_combo_table : DataFrame
                Dataframe holding the values that represent the rank that the unique combinations are in for each iteration
                example:
                             0.1 0.1 2.0   <-- alpha
                 combo 1      1   0   2    <-- ranks
                 combo 2      2   1   1
                 
        """       
        alpha_list = []       
        for al in self.alpha_df.alpha.unique():   ## alpha_list SAME AS self.alpha_arr_iterations (MAY BE REDUNDANT)
            count = list(self.alpha_df.alpha).count(al)
            alpha_list += [al] * count        
        combos = {}
        ## create combination dictionary with the unique combination as key and values of (iteration, rank)
        for idx, row in self.alpha_df.iterrows():
            if(row.combinations):  ## if iteration has rules 
                for rank in range(1,max_rank+1):    ## only rules that are rank 1-3 are included
                    if rank <= len(row.combinations):   
                        combo = row.combinations[rank-1]
                        if len(combo) > 1:
                            ## permut the different arrangments of the combinations
                            permut = list(itertools.permutations(tuple(combo)))                            
                            ## original arrangement of combination 
                            og_combo = tuple(permut[0])                          
                            for i in range(0,len(permut)):
                                r = permut[i]
                                if r in combos.keys():   ## if permut is in combination list, add iteration idx and rank
                                    indexes = combos[r][0] + [idx]
                                    ranks = combos[r][1] + [rank]                                    
                                    combos[r] = (indexes,ranks) ## update the combination
                                    break
                                if i == len(permut)-1:
                                    combos[og_combo] = ([idx],[rank]) ## new combination
            else:
                print("no combinations")        
        ## translate combination dataframe into 2d array and values of rank
        og_arr = np.empty((len(combos) , self.alpha_df.shape[0]))
        og_arr[:] = np.nan
        rowno = 0        
        for c in combos:
            for x in range(0,len(combos[c][0])):
                iteration = combos[c][0][x]
                rank = combos[c][1][x]
                if(og_arr[rowno,iteration] and ~np.isnan(og_arr[rowno,iteration])):
                    continue
                else:
                    og_arr[rowno,iteration] = rank
            rowno += 1
        ## the unique combination dataframe 
        unique_combo_table = pd.DataFrame(og_arr, index=list(combos.keys()) , columns = alpha_list)       
        return unique_combo_table
    
    def translate_to_rules(self):
        """ 
           Translate combinations into rule strings and corresponding to its alpha
      
            Returns
            -------
                alpha_rule_strings_dic : DataFrame
                Dataframe holding the keys of unique alphas and the values are a list of rules (feature, symbol, threshold value)                    
        """         
        alpha_rule_strings_dict ={}
        for a in self.alpha_df.alpha.unique():
            og_rules_alpha = self.alpha_df.query(f'alpha == {a}')
            og_rules_per_alpha  = og_rules_alpha.og_rules.tolist()
            iteration_list_query_rule_string = []
            for it in og_rules_per_alpha: #each iteration holding a list of Rule Objects of an alpha
                list_query_rule_string = []
                for j in it:  #rule object in each iteration 
                    strings = []
                    feature_string = []
                    for feature,sym in j.agg_dict:   #in one rule
                        rule_string= feature,sym, j.agg_dict[(feature,sym)]                        
                        ## only features 
                        feature_string.append(feature)                        
                        ## the whole rule 
                        rule_string = " ".join(rule_string)
                        strings.append(rule_string)
                    strings = sorted(strings)   ## sort alphabetically
                    query_rule_string = " & ".join(strings)
                    list_query_rule_string.append(query_rule_string)
                iteration_list_query_rule_string.extend(list_query_rule_string)
            ## remove any duplicate rules
            set_rule_string_per_alpha = set(iteration_list_query_rule_string)
            alpha_rule_strings_dict[a] = set_rule_string_per_alpha
        return alpha_rule_strings_dict
        
    def results_per_alpha(self, results, iterations_per_alpha):        
        """ 
           Create Alpha Objects based on their results
            
           Parameters
           ----------
               results: Dataframe
                   Dataframe holding the values that represent the rank 
                   that the unique combinations are in for each iteration

               iterations_per_alpha: Int 
                    Number of iterations running logistic regression per alpha
                    
            Returns
            -------
               variance_dataframe : DataFrame
                   showcases the information of every alpha such as
                   the variance of the combinations, number of unique rules, number of iterations                    
        """ 
        # print("Results Dataframe ", "\n")
        # display(results)        
        ## translate combinations into a list of rule strings 
        og_rules = self.translate_to_rules()        
        for a in results.columns.unique():  ## to split according to the alphas             
            try:
                sub_df= results[a]  ## returns a series if only one column
                sub_df = sub_df.dropna(axis = 0, how='all')   ## removing other rules that did not appear in the top 3 ranks for that particular alpha
                no_of_combos_per_alpha = len(list(sub_df.index))  ## number of unique combinations for the particular alpha
                if(no_of_combos_per_alpha == 0):  ## no rules in the alpha
                    break                
                self.linear_model_info['no_of_combos_per_alpha'].append(no_of_combos_per_alpha)                
                if (isinstance(sub_df, pd.Series)):  ## only one iteration of logreg for this alpha
                    list_of_rule_strings = og_rules[a]                    
                    ## create alpha object 
                    self.alpha_info_list.append(Alpha( alpha= a , sub_df=sub_df, rule_string= list_of_rule_strings))                    
                    self.linear_model_info['vari_arr'].append(0)  ## no variance to calculate bc only one iteration                    
                else:
                    ## have MORE than one iteration of logreg for this alpha                    
                    ## rename columns to the no of iterations
                    sub_df.columns = range(0, len(sub_df.columns))
                    total_variance = 0                    
                    ## the information table on variance of one alpha 
                    used_alpha_info = { a: list(sub_df.index) , 'frequent rank': [], 'frequency count': []
                                       , 'avg rank': [], 'variance' : []}                 
                    for idx, row in sub_df.iterrows():
                        r = list(row)
                        ## remove nan values in the row/ for each combo
                        new_r = [a for a in r if not math.isnan(a)]
                        try:
                            mode = statistics.mode(new_r)
                        except statistics.StatisticsError:
                            mode = Counter.most_common(new_r)
                        avg_rank = mean(new_r)
                        variance = statistics.pvariance(new_r)  ## variance for each combo
                        total_variance += variance  ## sum variance of all combos -- measure the variance of the alpha
                        freq_count = new_r.count(mode)
                        used_alpha_info['frequency count'].append(freq_count)
                        used_alpha_info['frequent rank'].append(mode)
                        used_alpha_info['avg rank'].append(avg_rank)
                        used_alpha_info['variance'].append(variance)
                    list_of_rule_strings = og_rules[a]
                    ## create Alpha object
                    self.alpha_info_list.append(Alpha(alpha = a ,sub_df= sub_df,rule_string= list_of_rule_strings,  used_alpha_info= used_alpha_info))
                    ## total variance of all the combinations for one alpha
                    self.linear_model_info['vari_arr'].append(total_variance)
                iterations_per_alpha = self.iterations_w_rules_alpha_dict[a]
                self.linear_model_info['no_of_iterations'].append(iterations_per_alpha) 
                self.linear_model_info['alpha_arr2'].append(a)
            except Exception as e:
                print("EXCEPTION OCCURRED " , e ,a)
                print(type(e) , type(a))
                # display(sub_df)
        try:
            ## variance dataframe 
            variance_df = pd.DataFrame(self.linear_model_info)
            variance_df.columns  = ['alpha', 'variance', 'unique rule' , 'no of iterations']
        except:
            print("not the same len arrays")
        return variance_df
    
    def find_best_alpha(self):
        """ 
           Find the best alpha according to 3 conditions :
           (1) Must have more/equal than 3 unique rules
           (2) Must have more/equal than 2 unique ranks
           (3) Variance of the alpha must be more than 0 and have the smallest variance
           
            Returns
            -------
               best_alpha : List
                   the float of the best alpha in the list of alphas 
                   -> using the combinations of the best alpha for the results
        """ 
        variance_df = self.variance_df
        ## Extract the unique ranks onto variance df
        unique_ranks = [a.no_unique_ranks for a in self.alpha_info_list]
        variance_df['unique ranks'] = unique_ranks
        ## check if there is more than one alpha generate more than 3 unique rule
        if(variance_df[(variance_df['unique rule'] >= 3)]['unique rule'].nunique() > 1):
            ## must fulfil all 3 conditions
            best_alpha = variance_df[( variance_df['unique ranks'] >= 2) & (variance_df['unique rule'] >= 3)
                                    & (variance_df['variance'] > 0)].nsmallest(1,'variance').alpha
            best_alpha = list(best_alpha.values)
        else: ## when variance can't be calculated and alphas only output one rule 
            print("when there is only one rule and no variance")
            best_alpha = list(variance_df.alpha)
        if best_alpha is None or best_alpha == []:
            best_alpha.append(0.1)
            print('No best alpha found. Defaulting to 0.1')
        print("The best alpha chosen " , best_alpha)
        return best_alpha

    def display_alpha_top3combinations(self, selected_alpha):
        """ 
        Showcase the combinations ranked top 3 of the selected alpha
        
        Parameters
        ----------
            selected_alpha: Float
                the alpha user inputs to get the top 3 combinations
        Returns
        -------
           combination_dataframe : Dataframe
               dataframe of the combinations and its frequent rank, 
               frequency count, avg rank and variance                    
        """ 
        list_of_alphas = [alpha.alpha for alpha in self.alpha_info_list]        
        ## when the same alpha appears more than once as a resort of running the same alpha within a range of iterations
        if(list_of_alphas.count(selected_alpha) > 1 and len(set(list_of_alphas)) == 1):
            combination_dataframe = [b.alpha_summary_table for b in self.alpha_info_list]
        else:
            ## when alpha only appears on in the list of alphas
            idx = list_of_alphas.index(selected_alpha)
            combination_dataframe = self.alpha_info_list[idx].alpha_summary_table
        return combination_dataframe
    
    def get_selected_alpha(self, selected_alpha):
        """ 
           gets the alpha object with the corresponding alpha value
            
            Parameters
            ----------
                selected_alpha: Float
                    the alpha user inputs to get the top 3 combinations
            Returns
            -------
                Alpha object                    
        """ 
        idx = [alpha.alpha for alpha in self.alpha_info_list].index(selected_alpha)
        return self.alpha_info_list[idx]    
     
    def apply_formatting(self, col):
        metrics_max = ['f1 scores','f1 macro','roc']
        metrics_min = ['brier loss', 'log loss']
        if col.name in metrics_max:
            return ['background-color: green' if c == max(col.values) and c!= 0 else '' for c in col.values]
        if col.name in metrics_min:
            return ['background-color: green' if c == min(col.values) else '' for c in col.values]
    
class Alpha:
    """An Object """
    """ 
       holds the information of one alpha 
        -----
       alpha: Float
           the alpha of the logistic regression
       rule_string: List
           list of rules 
       unique_combinations : List
           the list of combinations that resulted from the alpha
        sub_df : DataFrame
            Dataframe holding the values that represent the rank that the unique combinations are in for each iteration
            example:
                          0   1   2   <-- iteration no
             combo 1      1   0   2    <-- ranks
             combo 2      2   1   1
       alpha_summary_table  : Dataframe
           dataframe of the combinations and its frequent rank, 
           frequency count, avg rank and variance
       frequent_rank : List
           the ranking that appeared the most frequent in all iterations
       frequent_rank_count : List
           the count of the frequent rank in all iterations
       avg_rank : List
           the average ranking of all iterations
       no_unique_ranks : Int
            the number of unique ranks 
          
        """ 
    
    def __init__(self,alpha, sub_df, rule_string, used_alpha_info= None):
        
        self.sub_df = sub_df
        
        ## when there is more than one iteration of the alpha
        if used_alpha_info:
            self.alpha_summary_table = pd.DataFrame()
            self.alpha_summary = used_alpha_info
            self.frequent_rank = used_alpha_info['frequent rank']
            self.avg_rank = used_alpha_info['avg rank']
            self.frequent_rank_count = used_alpha_info['frequency count']
            self.unique_combinations = used_alpha_info[alpha]
            self.calculate_unique_ranks()
        else:
            self.no_unique_ranks = sub_df.nunique() ## no unique rank for one iteration
            self.alpha_summary_table = pd.DataFrame(sub_df)  ## shows the one iteration 

        
        self.alpha = alpha                      
        self.rule_string = rule_string
                                     
        
    def calculate_unique_ranks(self):
        """ 
        Calculate the unique ranks based on the frequent rank of each combinations
        
        """ 
        rank_df= pd.DataFrame(self.alpha_summary)
        rank_df = rank_df.sort_values(by= ['frequent rank', 'frequency count'], ascending = True)
        self.no_unique_ranks = rank_df['frequent rank'].nunique()
        
        self.alpha_summary_table = rank_df

    def display_rankgraph(self):
        """ 
           Plot how the ranks of each combination varies between each iteration
            --
        """ 
        sub_df_transponse = self.sub_df.transpose()

        ## Plot the ranking graphs 
        a4_dims = (20, 5)
        fig, ax = pyplot.subplots(figsize=a4_dims)
        ax.plot(sub_df_transponse)
        ax.invert_yaxis()
        pyplot.yticks(np.arange(1,4,1))
        pyplot.title(self.alpha)
        ax.legend(sub_df_transponse.columns, bbox_to_anchor=(1, 1))
