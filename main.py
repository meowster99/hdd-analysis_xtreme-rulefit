from http.client import NOT_FOUND
from itertools import filterfalse
import json
import ast
import re
from four_m.models import Dataset
from four_m.home.training.features.processing import preprocessing_pipeline
from four_m.home.training.models.tuning import GBCTuner, XGBTuner
from four_m.home.training.models.training_logreg import Rule_Filter
import pandas as pd
from four_m.home.training.models.xtremerulefit import XtremeRulefitClassifier
# for graph visualisation 12/01/22
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

BOOSTED = True # easy control to activate XtremeRuleFit

def start_training(rulefit, gbctuner, xgbtuner):
    if BOOSTED:         
        xgbtuner.start_tuning()
    else:     
        gbctuner.start_tuning()
    X_concat = rulefit.fit_transform()
    return X_concat

def __main__(id, training):
    no_of_features_dict = {}
    pearson_html = None
    chart_html = None
    faf_html = None
    hyperparameters = None

    print("Entering main processing function", id)
    dataset = dict(Dataset.objects.values().get(pk=id)) # create instance of dataset and store as a dictionary

    print(dataset.get('features'), type(dataset.get('features')))
    print(dataset['features'], type(dataset.get('features')))

    if(isinstance(dataset['features'], str) and dataset.get('features') != ''): # called after the dataset has already been processed and features have been saved
        try:
            dataset['features'] = json.loads(dataset['features']) # successfully converts to a list
        except json.decoder.JSONDecodeError:
            dataset['features'] = ast.literal_eval(dataset['features']) # covert to list using ast literal eval if json fails
        print(type(dataset['features']))
    # print("dataset ", dataset, type(dataset))

    obj = Dataset.objects.get(pk=id)
    data = pd.read_csv(obj.dataset)  # read the dataset as a pandas dataframe
    RESPONSE = dataset.get('response') # response is the class variable
    print('Response variable selected: ', RESPONSE, type(RESPONSE))

    if(training):
        print("TRAINING HAS BEGUN >>>>>>>>>>>>>")
        try:
            data = data.drop(columns=['Unnamed: 0'])
            # print(data[RESPONSE], type(data), data.shape)
            pipeline , clean_data = preprocessing_pipeline(data, dataset, training=True) # initialise the pipeline from home.training.features.processing
            X = clean_data.drop(columns=[RESPONSE]) # drop the class value to train the model 
            # print(X, type(X), X.shape)
            Y = clean_data[RESPONSE] # store the class value as Y to train the model
            # print(Y, type(Y), Y.shape)
            gbctuner, xgbtuner, rulefit, rule_filter,status = train_models(X,Y) # local function refer below
        except:
            status = "No results"

        if(status == "No results"):
            combo_tables = None
            combo_to_rule_dict = None
        else:
            select_alpha = rule_filter.best_alpha[0]
            alpha_obj = rule_filter.get_selected_alpha(select_alpha)
            combo_to_rule_dict = rule_pivot_table(alpha_obj.rule_string, clean_data, data, RESPONSE)
            combo_tables =  create_combo_tables(combo_to_rule_dict, data, RESPONSE)
        try:
            if BOOSTED and hasattr(xgbtuner, 'optimal_params'):
                hyperparameters = xgbtuner.optimal_params
            elif hasattr(gbctuner, 'optimal_parameters'):
                hyperparameters = gbctuner.optimal_parameters
            else:
                hyperparameters = {0:'Bayesian Optimisation was not initialised.'}
        except:
            status = "No results"
            hyperparameters = {0:'No results found. Pls re-analyse or use a larger dataset.'}
        training_context = { 
                'hyperparameters': hyperparameters,
                'best_alpha_combo': combo_tables,
                'status' :status,
            }
        return training_context, combo_to_rule_dict

    pipeline , clean_data = preprocessing_pipeline(data, dataset, training = False) # initialise the pipeline from home.training.features.processing
    X = clean_data.drop(columns= [RESPONSE])
    Y = clean_data[RESPONSE]
    no_of_features_dict['initial'] = len(data.columns)
    keep_cols = pipeline['data preparation'].keep_cols.to_html()
    no_of_features_dict['level'] = pipeline['data preparation'].keep_cols.shape[0]
    if dataset.get('pearson'):
        pearson_html = pipeline['pearson test'].p_values.to_html()
        no_of_features_dict['pearson'] = pipeline['pearson test'].p_values.shape[0]
        chart_html = plot_failure_rates(data, pipeline['pearson test'].p_values, Y)
    if dataset.get('remove_association'):
        # faf_html = pipeline['full association'].chart.to_html(index=False)
        faf_html = pipeline['full association'].pair_list
        print("Associated Feature dictionary " , faf_html)
        no_of_features_dict['full association'] = no_of_features_dict[list(no_of_features_dict.keys())[-1]] - pipeline['full association'].chart.shape[0]
    feature_level = 'Primary' if dataset.get('level') == 'P' else 'Secondary'
    feature_options= list(set( data.columns).difference(set(X.columns)))
    no_of_features_dict['output'] = len(X.columns)
    context = {
        'level_features': keep_cols,
        'no_of_features_dict' : no_of_features_dict,
        'pearson':pearson_html,
        'chart' : chart_html,
        'faf': faf_html,
        'level': feature_level,
        'relevant_features': list(X.columns),
        'data_columns': feature_options,
        # 'hyperparameters': hyperparameters,
        # 'log_reg_results':log_reg_results,
        # 'best_alpha_combo':alpha_combo
    }
    return context

def train_models(X,Y):
    xgbtuner = XGBTuner(X,Y)
    gbctuner = GBCTuner(X,Y) # from home.training.models.tuning
    # adjust model_type to either 'classifier' or 'regressor' WARNING: regressor has never been tested
    rulefit = XtremeRulefitClassifier(X,Y, gbctuner = gbctuner, xgbtuner = xgbtuner, boosted=BOOSTED) #, xgbtuner=xgbtuner, gbctuner=gbctuner)
    X_concat = start_training(rulefit, gbctuner, xgbtuner) # local function refer above
    rule_filter  = Rule_Filter(model=rulefit, maxrules=rulefit.maxrules, X=X_concat, Y=Y) # from home.training.models.training_logreg
    rule_filter.train_logreg_range_of_iterations([100])
    if hasattr(rule_filter, 'variance_df'):
        log_reg_results = rule_filter.variance_df.to_html()
        # print(log_reg_results)
        if(not rule_filter.variance_df.empty):
            log_reg_results = rule_filter.variance_df.to_html()
            select_alpha = rule_filter.best_alpha[0]
            print("BEST ALPHA ", select_alpha)
            status ="Success"
        else:
            status = "No results"
            print("NO RESULTS")
    else:
        status = "No results"
        print("NO RESULTS")  
    return gbctuner, xgbtuner, rulefit, rule_filter,status

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def create_combo_tables(combo_to_rule_dict,  data, RESPONSE):
    alpha_combos_html =[]
    for combo, rule in combo_to_rule_dict.items():
        combo = list(np.unique(np.array(combo)))
        if len(combo)>1:
            pt = make_pivot_table(combo, data, RESPONSE)
            # pt = pt.sort_values(by='yield', ascending=False) # to sort everything by yield, the entire pivot table
            best_rule = pt['yield'].idxmax() # selects rule based on the yield
            locs = list()
            locs.append(pt.index.get_loc(best_rule[0]).start)
            if len(combo) > 2:
                for i in range(pt.index.get_loc(best_rule[0]).start, pt.index.get_loc(best_rule[0]).stop):
                    if i <= pt.index.get_loc(best_rule):
                        values = pt.index.get_level_values(1)
                        if values[i] == best_rule[1] and len(locs)<2:
                            locs.append(i)
                        if len(combo) > 3:
                            values = pt.index.get_level_values(2)
                            if values[i] == best_rule[2]:
                                locs.append(i)
            locs.append(pt.index.get_loc(best_rule))
            css = [{'selector': f'.level{level}.row{idx}', 'props': [('background-color', f'rgba(144,190,74,0.8)')]} for level, idx in enumerate(locs)]
            css.append({'selector': f'.row{locs[-1]}', 'props': [('background-color', f'rgba(144,190,74,0.8)')]})
            pt = pt.style.set_table_styles(css)
            pt_rendered = pt.render()
            alpha_combos_html.append(pt_rendered)
    return alpha_combos_html

def rule_pivot_table( alpha_rule_strings_list, cleaned_data, data, RESPONSE):  ## pass in rule string list of one alpha
    combo_to_rule_dict = {}
    for set_rule_string in alpha_rule_strings_list:
        print(set_rule_string)
        features = re.findall("\w*\s\S*>|\w*\s\S*<", set_rule_string)
        final_features = list()
        for m in [' <', ' >']:
            features =[ feature.replace(m, "") for feature in features]
        for feature in features:
            if feature not in final_features:
                final_features.append(feature)
        query_df = cleaned_data.query(set_rule_string)
        query_df_features = data.iloc[query_df.index][query_df.columns]
        if len(final_features) > 1:
            pt = make_pivot_table(final_features, query_df_features, RESPONSE)
            final_features.sort(reverse=False)
            list_of_feature_values = list(pt.iloc[0].name)
            combo_to_rule_dict[tuple(final_features)] = list_of_feature_values
    return combo_to_rule_dict

def make_pivot_table(colname, orig, response):
    multiple_col = []
    res_list=[]
    for i in colname:
        multiple_col.append(orig[i])
    orig_crosstab = pd.crosstab(multiple_col, orig[response]) 
    try:
        ## calculating yield 
        fail_val = orig[response].unique()[1]
        pass_val = orig[response].unique()[0]
        for i in orig_crosstab.iterrows():
            res = i[1][fail_val] / (i[1][pass_val] + i[1][fail_val])
            res_list.append(res)
        orig_crosstab['yield'] = res_list
    except:
        print("pivot table can't be created ")
    # orig_crosstab.sort_values(by='yield')
    orig_crosstab.rename(columns={"F": "FAIL", "P": "PASS"} , inplace = True)
    return orig_crosstab

def generate_pivot_table(colname, orig, response):
    multiple_col = []
    res_list=[]
    if len(colname) >1:
        for i in colname:
            multiple_col.append(orig[i])
        orig_crosstab = pd.crosstab(multiple_col, orig[response]) 
    else:
        return "Choose at least 2 relevant features."
    try:
        ## calculating yield 
        fail_val = orig[response].unique()[1]
        pass_val = orig[response].unique()[0]
        for i in orig_crosstab.iterrows():
            res = i[1][fail_val] / (i[1][pass_val] + i[1][fail_val])
            res_list.append(res)
        orig_crosstab['yield'] = res_list
    except:
        print("pivot table can't be created ")
    orig_crosstab.rename(columns={"F": "FAIL", "P": "PASS"} , inplace = True)
    return orig_crosstab

# for graph visualisation 12/01/22
def plot_failure_rates(data, pearson, y_values):
    """ 
    plot_failure_rates Function
    
    This function will plot a bar chart of the pearson filtered features 
    against the failure rates for that feature.
    

    Parameters
    ----------
    data        :       cleaned data after being passed through the pipeline
    pearson     :       list of pearson features produced during feature selection        
    
    Returns
    ----------
    html figure      :       figure.to_html() -- plotly graph in html
    """
    pearson_features = [feature for feature in pearson.index] # list of features that have been filtered

    if len(pearson_features) > 0:
        pearson_df = data[pearson_features] # dataframe of the pearson filtered features
        pearson_df['class'] = y_values # reading the class col for easier access
        # constants, edit when changing matrix subplots
        if len(pearson_features)%2 == 0: 
            ROWS = len(pearson_features) // 2
            COLS = len(pearson_features) // ROWS # 2 columns
        else:
            ROWS = (len(pearson_features) // 2) + 1
            COLS = 2 # 2 columns
        
        # try:
        fig = make_subplots(
            cols = COLS, 
            rows = ROWS,
            # horizontal_spacing=0.02, # cannor be greater than 1/(cols-1)
            # vertical_spacing=0.01 # cannot be greater than 1/(rows -1)
            )
        for i, feature  in enumerate(pearson_features):
            pearson_df.loc[:,[feature, 'class']].value_counts().to_csv('results_{}.csv'.format(feature))
            df = pd.read_csv('results_{}.csv'.format(feature))    
            df = df.sort_values(by = 'class', ascending = False)
            df.loc[:, 'failure_rate'] = df.groupby([feature])['0'].apply(lambda x: x / x.sum())
            df = df[df['class'] == 1]
            # clear files
            if os.path.exists('results_{}.csv'.format(feature)):
                os.remove('results_{}.csv'.format(feature))
            # create trace
            trace = go.Bar(x = df[feature], y = df['failure_rate'], text = df['0'], marker = dict(color = '#90be4a'))
            # to format into a grid matrix
            i = i+1
            if i <= len(pearson_features) // 2:
                fig.add_trace(trace, row =i, col = 1,)
                fig.update_yaxes(title_text="Failure Rate", row =i, col = 1, tickfont_size = 10, titlefont_size = 13, tickformat = ',.0%')
                fig.update_xaxes(tickfont_size = 9, titlefont_size = 13, title_text = feature, row =i, col = 1, tickangle=-10)
            else:
                i = i - len(pearson_features) // 2
                fig.add_trace(trace, row =i, col = 2,)
                fig.update_yaxes(title_text="Failure Rate", row =i, col = 2, tickfont_size = 10, titlefont_size = 13, tickformat = ',.0%')
                fig.update_xaxes(tickfont_size = 9, titlefont_size = 13, title_text = feature, row =i, col = 2, tickangle=-10)
            fig.update_traces(hoverinfo = 'x+y+text') # editing the text in the hover label
        fig.update_traces(textfont_size=10, textangle=0, textposition="outside", cliponaxis=False)
        fig.update_layout( # adjusting the layout
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
            ),
            height=len(pearson_features)*108, #dynamically adjust the height of the graph depending onthe filtered features
            width=1000,
            autosize=False,
            template="plotly_white",
            showlegend = False,
            hovermode = "x"
            )
        # except:
        #     fig = go.Figure()
        #     fig.update_layout(yaxis={'title':'Failure Rate (%)'}, title = 'Unable to display figure')
        return fig.to_html()

    else:
        return 'No Pearson Features to display!'