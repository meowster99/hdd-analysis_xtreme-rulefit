from collections import Counter, OrderedDict
import re
from time import thread_time_ns
from typing import Dict, Iterable, List, Union
import warnings
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from sklearn.utils import check_array
from xgboost import XGBClassifier
from imodels import RuleFitClassifier, RuleFitRegressor
from imodels.util.transforms import Winsorizer, FriedScale
import numpy as np
import pandas as pd
import itertools as it
from sklearn.preprocessing import RobustScaler

class XtremeRulefitClassifier():
    """
    Parameters
    ----------
    X               :       (dataframe) X values of dataset
    y               :       (dataframe) response values of dataset
    model_type      :       (str) 'classifier' or 'regressor' depending on the task
    gcbtuner        :       (object) initialise bayesian optimisation for GradientBoostingClassifier, else use sklearn default values
    xgbtuner        :       (object) initialise bayesian optimisation for XGBClassifier, else use the default xgb settings
    tree_size       :       (int) manually restrict the tree size
    boosted         :       (bool) allow user to use XGBClassifier or GBClassifier when model_type = 'classifier'
    """
    def __init__(self, 
                 X, y,
                 model_type='classifier', 
                 gbctuner=None, 
                 xgbtuner=None,
                 max_depth=7, 
                 tree_size=None, 
                 boosted=False,):
        self.X = X
        self.y = pd.DataFrame(y) if not isinstance(y, pd.core.series.Series) else y
        self.model_type = model_type
        self.GBCTUNER = gbctuner # intialise the tuners, tuning happens in main.py
        self.XGBTUNER = xgbtuner # intialise the tuners, tuning happens in main.py
        self.max_depth = max_depth # terminology for xgbc
        self.tree_size = tree_size # terminology for gbc
        self.boosted = boosted
        self.maxrules = 5000
        self.ext_scaler = None
        
    def fit_transform(self):
        """
        Initialises tree model. Creates rulefit. Outputs transformed rules.
        """
        tree_generator = self.init_set_classifier() # get the base model to train
        rulefit = self.create_rulefit(tree_generator)
        self.init_rulefit(tree_generator)
        self.extracted_rules = self.extract_rules(tree_generator) # extract rules from the model
        X_transformed = self.transform(self.extracted_rules)
        return X_transformed

    def init_set_classifier(self):
        """
        Initialises the various tree models.
        """
        sample_fract_ = min(0.5, (100 + 6 * np.sqrt(self.X.shape[0])) / self.X.shape[0])
        if self.model_type == 'classifier':
            if self.boosted == True:
                if hasattr(self.XGBTUNER, 'optimal_params'):
                    model = XGBClassifier(**self.XGBTUNER.optimal_params, use_label_encoder=False, eval_metric='logloss')
                    print('XGBClassifier with Bayesian Optimisation selected.')
                else:
                    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                    print('XGBClassifier without Bayesian Optimisation selected.')
            else:
                if hasattr(self.GBCTUNER, 'optimal_parameters'):
                    model = GradientBoostingClassifier(**self.GBCTUNER.optimal_parameters)
                    print('GradientBoostingClassifier with Bayesian Optimisation selected.')
                else:
                    model = GradientBoostingClassifier(warm_start=True, random_state=42)
                    print('GradientBoostingClassifier without Bayesian Optimisation selected.')
        elif self.model_type == 'regressor':
            warnings.warn("Regression model has not been tested before.")
            model = GradientBoostingRegressor(**self.GBCTUNER.optimal_parameters)
            print('GradientBoostingRegressor selected.')
        else:
            raise NotImplementedError("No suitable model chosen to train the rulefit classifier.")
        self.model = model # assigning for easier access throughout code
        return model # tree_generator

    def create_rulefit(self, tree_generator):
        """
        Function to initialise rulefit model from imodels. Will also set the max_depth/tree_size depending on the optimal hyperparameters set.

        Parameters
        ----------
        tree_generator      :       (object) Either GradientBoostingClassifier or GradientBoostingRegressor or XGBClassifier

        Returns
        --------
        rulefit             :       (object) rulefit model
        """
        if isinstance(tree_generator, XGBClassifier):
            if self.XGBTUNER != None and hasattr(self.XGBTUNER, 'optimal_params'): # using XGBC --> ergo take parameters from xgbtuner
                self.max_depth = self.XGBTUNER.optimal_params['max_depth']
            else:
                self.max_depth = np.random.randint(3,50) # randomise max_depth
        elif isinstance(tree_generator, GradientBoostingClassifier):
            if self.tree_size is None:
                if self.GBCTUNER != None and hasattr(self.GBCTUNER, 'optimal_parameters'):
                    self.tree_size = self.GBCTUNER.optimal_parameters['max_leaf_nodes']
                else:
                    self.tree_size = np.random.randint(3,50) # randomise tree_size
        if self.model_type == 'classifier':
            rulefit = RuleFitClassifier(exp_rand_tree_size=True, tree_size=self.tree_size, max_rules=self.maxrules, include_linear=False, tree_generator=tree_generator)
        elif self.model_type == 'regressor':
            rulefit = RuleFitRegressor(exp_rand_tree_size=True, tree_size=self.tree_size, max_rules=self.maxrules, include_linear=False, tree_generator=tree_generator)
        return rulefit

    def init_rulefit(self, tree_generator):
        features = list(self.X.columns)
        rulefit = self.create_rulefit(tree_generator)
        rulefit.n_features_ = self.X.shape[1]
        rulefit.feature_dict_ = self.get_feature_dict(self.X.shape[1], features)
        print("Feature dictonary of features sent to model: \n", rulefit.feature_dict_)
        rulefit.feature_placeholders = list(rulefit.feature_dict_.keys())
        rulefit.feature_names = list(rulefit.feature_dict_.values())
        lin_trim_quantile=0.025
        rulefit.winsorizer = Winsorizer(trim_quantile=lin_trim_quantile)
        rulefit.friedscale = FriedScale(rulefit.winsorizer)
        rulefit.stddev = None
        rulefit.mean = None
        self.rulefit_model = rulefit

    def extract_rules(self, tree_generator, exp_rand_tree_size=True, random_state=0):
        if not exp_rand_tree_size:
            tree_generator.fit(self.X, self.y)
        else: # randomise tree size as per Friedman 2005 Sec 3.3
            if isinstance(tree_generator, (GradientBoostingClassifier, GradientBoostingRegressor)):
                np.random.seed(random_state)
                tree_sizes = np.random.exponential(scale=self.tree_size-2,
                                                    size=int(np.ceil(self.maxrules*2/self.tree_size)))
                tree_sizes = np.asarray([2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))], dtype=int)
                tree_generator.set_params(warm_start=True)
                curr_est_ = 0
                for i_size in np.arange(len(tree_sizes)):
                    size = tree_sizes[i_size]
                    tree_generator.set_params(n_estimators=curr_est_ + 1)
                    tree_generator.set_params(max_leaf_nodes=size)
                    random_state_add = random_state if random_state else 0
                    tree_generator.set_params(
                        random_state=i_size + random_state_add)  # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
                    tree_generator.fit(np.copy(self.X, order='C'), np.copy(self.y, order='C'))
                    curr_est_ = curr_est_ + 1
                tree_generator.set_params(warm_start=False)
            elif isinstance(tree_generator, XGBClassifier):
                np.random.seed(random_state)
                tree_sizes = np.random.exponential(scale=self.max_depth-2,
                                                    size=int(np.ceil(self.maxrules*2/self.max_depth)))
                tree_sizes = np.asarray([2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))], dtype=int)
                curr_est_ = 0
                for i_size in np.arange(len(tree_sizes)):
                    size = tree_sizes[i_size]
                    tree_generator.set_params(n_estimators=curr_est_ + 1)
                    random_state_add = random_state if random_state else 0
                    tree_generator.set_params(
                        random_state=i_size + random_state_add)  # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
                    tree_generator.fit(np.copy(self.X, order='C'), np.copy(self.y, order='C'))
                    curr_est_ = curr_est_ + 1
        seen_antecedents = set()
        extracted_rules = [] 
        if isinstance(tree_generator, XGBClassifier): # i.e., self.boosted == True
            print('Getting rules from xgboosted trees >>>>>')
            self.features = self.X.columns.values
            self.X = check_array(self.X)
            self._rule_dump = tree_generator._Booster.get_dump()
            leaves_l = []
            for tree_i in self._rule_dump:
                leaves = [int(i) for i in re.findall(r'([0-9]+):leaf=', tree_i)]
                leaves_l.append(leaves)
            tree_df = self.model.get_booster().trees_to_dataframe()
            rules = list(it.chain(*[self.__extract_xgb_dt_rules__(dt) for dt in self._rule_dump]))
            if len(rules) > 1:
                formatted_rules = np.array(tuple([' and '.join(self.__convert_rule__(r, labels=None, scaler=self.ext_scaler)) for r in rules]))
                # print('EXTRACTED RULES: ', formatted_rules)
                for rule_value_pair in formatted_rules:
                    if rule_value_pair not in seen_antecedents:
                        extracted_rules.append(rule_value_pair)
                        seen_antecedents.add(rule_value_pair)
                print('extracted rules : ', extracted_rules)
            else:
                print('failed')
            # extracted_rules = sorted(extracted_rules, key=lambda x: x[1])
            # extracted_rules = list(map(lambda x: x[0], extracted_rules))
        else:
            print('Getting rules from gradientboosted trees >>>>>')
            estimators_ = tree_generator.estimators_
            for estimator in estimators_:
                for rule_value_pair in self.tree_to_rules(estimator[0], np.array(self.rulefit_model.feature_placeholders), prediction_values=True):
                    if rule_value_pair[0] not in seen_antecedents:
                        extracted_rules.append(rule_value_pair)
                        seen_antecedents.add(rule_value_pair[0])
            extracted_rules = sorted(extracted_rules, key=lambda x: x[1]) # sorts based on prediction values
            extracted_rules = list(map(lambda x: x[0], extracted_rules)) 
            print('extracted rules : ', extracted_rules)
        print("Number of extracted rules : " , len(extracted_rules))
        return extracted_rules

    def transform(self, extracted_rules):
        """"Function to transform extracted rules."""
        df = pd.DataFrame(self.X)
        df.columns = self.rulefit_model.feature_placeholders
        X_transformed = np.zeros([self.X.shape[0], 0])
        for r in extracted_rules:
            if r!= '': # handle case when empty rule is appended
                curr_rule_feature = np.zeros(self.X.shape[0])
                curr_rule_feature[list(df.query(r).index)] = 1
                curr_rule_feature = np.expand_dims(curr_rule_feature, axis=1)  # make into a column
                X_transformed = np.concatenate((X_transformed, curr_rule_feature), axis=1)
        return X_transformed

    def get_feature_dict(self, num_features: int, feature_names: Iterable[str] = None) -> Dict[str, str]:
        # format of rules extracted from the tree is as follows:
        #       
        feature_dict = OrderedDict()
        if feature_names is not None:
            for i in range(num_features):
                feature_dict[f'feature_{i}'] = feature_names[i]
        else:
            for i in range(num_features):
                feature_dict[f'feature_{i}'] = f'feature_{i}'
        return feature_dict
    
    def tree_to_rules(self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
                      feature_names: List[str],
                      prediction_values: bool = False, round_thresholds=True) -> List[str]:
        """
        Return a list of rules from a tree

        Parameters
        ----------
            tree : Decision Tree Classifier/Regressor
            feature_names: list of variable names

        Returns
        -------
        rules : list of rules.
        """
        tree_ = tree.tree_
        feature_name = [
            self.rulefit_model.feature_placeholders[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        rules = []
        def recurse(node, base_name):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                symbol = '<='
                symbol2 = '>'
                threshold = tree_.threshold[node]
                if round_thresholds:
                    threshold = np.round(threshold, decimals=5)
                text = base_name + ["{} {} {}".format(name, symbol, threshold)]
                recurse(tree_.children_left[node], text)
                text = base_name + ["{} {} {}".format(name, symbol2,
                                                      threshold)]
                recurse(tree_.children_right[node], text)
            else:
                rule = str.join(' and ', base_name)
                rule = (rule if rule != ''
                        else ' == '.join([self.rulefit_model.feature_placeholders[0]] * 2))
                # a rule selecting all is set to "c0==c0"
                if prediction_values:
                    rules.append((rule, tree_.value[node][0][0]))
                else:
                    rules.append(rule)
        recurse(0, [])
        return rules if len(rules) > 0 else 'True'
    
    def __extract_xgb_dt_rules__(self, dt):
        """ Extract rule set from single decision tree according
        to `XGBClassifier` format

        Parameters
        ----------
        dt: string
        
        Returns
        -------
        list of numpy.ndarray
            Each array is of length three. 
            First indicates feature number,
            Second indicates operator (1 if $>$ otherwise $\leq$),
            Third indicates threshold value
        """ 
        md = self.max_depth + 1  # upper limit of max_depth?
        rules = []
        levels = np.zeros((md, 3))  # Stores: (feature name, threshold, next node id)
        path = []
        # Extract feature numbers and thresholds for all nodes
        feat_thresh_l = re.findall(r'\[f([0-9]+)<([-]?[0-9]+\.?[0-9]*)\]', dt)
        _id = 0
        prune = -1
        for line in dt.split('\n')[:-1]:
            # Separate node id and rest of line
            _id, rest = line.split(':')
            # Count number of tabs at start of line to get level (and then remove)
            level = Counter(_id)['\t']
            _id = _id.lstrip()
            if prune > 0:
                # If we were last at a leaf, prune the path
                path = path[:-1+(level-prune)]
            # Add current node to path
            path.append(int(_id))
            if 'leaf' in rest:
                prune = level  # Store where we are so we can prune when we backtrack
                rules.append(levels[:level, (0, 2, 1)].copy())  # Add rules
                rules[-1][:, 1] = rules[-1][:, 1] == np.array(path[1:])  # Convert path to geq/leq operators
            else:
                # Extract (feature name, threshold, next node id)
                levels[level, :] = re.findall(r'\[f([0-9]+)<([-]?[0-9]+\.?[0-9]*)\].*yes=([0-9]+)', line)[0]
                # Don't prune
                prune = -1
        return rules

    def __convert_rule__(self, x, labels=None, scaler=None):
        """Convert rule represented by an array to readable format
        
        Parameters
        ----------
        x: numpy.ndarray
            Input array where each row represents a feature in a rule.
            3 columns:
            First indicates feature number,
            Second indicates operator (1 if $>$ otherwise $\leq$),
            Third indicates threshold value
        
        labels: list of str, optional
            Names of features to replace feature numbers with
        
        scaler:
            Scaler to reverse scaling done in fitting so interpretable
            feature values can be used.
        
        Returns
        -------
        list of str
            List containing each stage of input rule
        
        """
        strop = ['>', '<=']
        try:
            if scaler is None:
                # If no scaler, do not shift or scale
                nf = x[:, 0].astype(int).max()+1
                scale = np.ones(nf)
                center = np.zeros(nf)
            else:
                scale = scaler.scale_
                center = scaler.center_
        except ValueError as e:
            print('VALUE ERROR OCCURRED: ', e)
            pass

        if labels is None:
            return [('feature_{}'.format(str(int(f))) + ' ' + str(strop[int(op)]) + ' ' + str(thresh*scale[int(f)]+center[int(f)])) for f, op, thresh in x]
        else:
            return [(labels[int(f)] + str(strop[int(op)]) + str(thresh*scale[int(f)]+center[int(f)])) for f, op, thresh in x]
