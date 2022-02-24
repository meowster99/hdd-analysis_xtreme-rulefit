import time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

## for bayesion optimization
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from sklearn.model_selection import cross_val_score

class GBCTuner():
    """ Finding the optimal hyperparameters for the Gradient Boosting Classifier in Rulefit using bayesian optimisation
    
    Parameters
    ----------
    X : Series
    The whole dataset except for the Response column
    Y : Series
    The Response column
    scoring: String, optional
    The scoring represents the metric being used for hypertuning 
    negative_metric: bool, optional
    If True, it is a loss metric so the model is better if the metric is the lowest (ex. neg_log_loss , brier_loss)
    If False, it is a metric where the model is better if it is increased (ex. f1, roc_auc)

    Returns
    -------
    optimal parameters

    """
    def __init__(self, X, Y, scoring='neg_los_loss', negative=True, args=None):
        self.scoring = scoring
        self.negative_metric = negative
        self.X= X
        self.Y = Y

    def gb_hyperparameter_tuning(self,params):
        clf = GradientBoostingClassifier(**params , random_state = 0)
        acc = cross_val_score(clf, self.X, self.Y,scoring="neg_log_loss").mean()
        if self.negative_metric:
            return {"loss": acc, "status": STATUS_OK}
        else:
            return {"loss": -acc, "status": STATUS_OK}
    def start_tuning(self):
        trials = Trials()
        start_time = time.time()
        ## the search space for hyperparameters
        self.maxfeatures =  ['log2' ,'sqrt']
        RF_param_distributions = {
            "min_samples_split": hp.randint("min_samples_split", 5,30),
            "min_samples_leaf": hp.randint("min_samples_leaf", 5,30),
            'learning_rate': hp.loguniform('learning_rate',
                                             np.log(0.005),
                                             np.log(0.2)),
            'max_leaf_nodes': hp.randint("max_leaf_nodes", 4,16),  ## max depth will be 3,4,5
            # "max_features" : hp.choice("max_features", self.maxfeatures) # causing error in training
            }

        best = fmin( fn=self.gb_hyperparameter_tuning, space = RF_param_distributions, algo=tpe.suggest, max_evals=50, trials=trials)
        print("Optimizing took: ", time.time()-start_time)
        self.optimal_parameters =best
        return best

from bayes_opt import BayesianOptimization
import xgboost as xgb

class XGBTuner():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dtrain = xgb.DMatrix(self.X, label = self.Y)

    def bo_tune_xgb(self, max_depth, gamma, n_estimators, learning_rate, reg_lambda, max_delta_step, min_child_weight):
        params = {
                'max_depth': int(max_depth),
                'gamma': gamma,
                'learning_rate':learning_rate,
                # 'eta': 0.1,
                'eval_metric': 'auc',
                'tree_method': 'exact', # exact not bad
                'reg_lambda':reg_lambda,
                'min_child_weight':min_child_weight,
                'max_delta_step':max_delta_step,
                'subsample':0.8,
                # 'process_type': 'update',
                'grow_policy':'lossguide',
                # 'num_parallel_tree':10,
                }
        #Cross validating with the specified parameters in 5 folds and 10 iterations
        cv_result = xgb.cv(params, self.dtrain, num_boost_round=20, nfold=20)
        #Return the negative RMSE
        return cv_result['test-auc-mean'].iloc[-1]

    def start_tuning(self):
        start_time = time.time()
        xgb_bo = BayesianOptimization(self.bo_tune_xgb, {
                'max_depth': (3, 150),
                'gamma': (0, 5),
                'learning_rate':(0.001,1),
                'n_estimators':(100,150),
                'reg_lambda': (0.001,1),
                'min_child_weight':(0,5),
                'max_delta_step': (1,10)
                #'tree_method': ['exact', 'approx', 'hist', 'gpu_hist']
                })
        xgb_bo.maximize(n_iter= 20, init_points=10, acq='ucb')
        end_time = time.time()
        print("Optimizing took: ", end_time-start_time)
        params = xgb_bo.max['params']
        params['max_depth']= int(params['max_depth'])
        params['n_estimators']= int(params['n_estimators'])
        self.optimal_params = params
        print(params)
        return params