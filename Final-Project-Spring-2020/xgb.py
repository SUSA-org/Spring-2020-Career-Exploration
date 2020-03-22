import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime

def xgb_model(trn_x, trn_y, val_x, val_y, test, verbose, xgb_train, params) :
    
    record = dict()
    model = xgb_train(params
                      , xgb.DMatrix(trn_x, trn_y)
                      , 100000
                      , [(xgb.DMatrix(trn_x, trn_y), 'train'), (xgb.DMatrix(val_x, val_y), 'valid')]
                      , verbose_eval=verbose
                      , early_stopping_rounds=500
                      , callbacks = [xgb.callback.record_evaluation(record)])
    best_idx = np.argmin(np.array(record['valid']['rmse']))

    val_pred = model.predict(xgb.DMatrix(val_x), ntree_limit=model.best_ntree_limit)
    test_pred = model.predict(xgb.DMatrix(X_test.values), ntree_limit=model.best_ntree_limit)

    return {'val':val_pred, 'test':test_pred, 'error':record['valid']['rmse'][best_idx], 'importance':[i for k, i in model.get_score().items()]}

def run_xgb(X_train, y_train, X_test, xgb_train, params):

    random_seed = 42
    k = 10
    fold = list(KFold(k, shuffle = True, random_state = random_seed).split(X_train.values))
    np.random.seed(random_seed)
    
    params.update({
        'objective': 'reg:linear', 
        'eval_metric': 'rmse', 
        'seed': random_seed, 
    })
    
    result_dict = dict()
    val_pred = np.zeros(X_train.values.shape[0])
    test_pred = np.zeros(X_test.shape[0])
    final_err = 0
    verbose = False

    for i, (trn, val) in enumerate(fold) :
        print(i+1, "fold.    RMSE")

        trn_x = X_train.values[trn, :]
        trn_y = y_train[trn]
        val_x = X_train.values[val, :]
        val_y = y_train[val]

        fold_val_pred = []
        fold_test_pred = []
        fold_err = []

        #""" xgboost
        start = datetime.now()
        result = xgb_model(trn_x, trn_y, val_x, val_y, X_test, verbose, xgb_train, params)
        fold_val_pred.append(result['val'])
        fold_test_pred.append(result['test'])
        fold_err.append(result['error'])
        print("xgb model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')
        #"""

        val_pred[val] += np.mean(np.array(fold_val_pred), axis = 0)
        test_pred += np.mean(np.array(fold_test_pred), axis = 0) / k
        final_err += (sum(fold_err) / len(fold_err)) / k
    
    return test_pred, final_err, np.sqrt(np.mean((val_pred - y_train)**2))