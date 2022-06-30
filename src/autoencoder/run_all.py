import pandas as pd
from tensorflow.keras.layers import Lambda
from src.autoencoder.training import results_to_file, run_experiment
from src.layers.diffpool import DiffPool
from src.models.autoencoders import Autoencoder
from src.modules.upsampling import upsampling_with_pinv

from src.autoencoder.training_nt import run_experiment as run_experiment_nt
from src.modules.graclus import GRACLUS, preprocess

from src.layers import LaPool
from src.layers.mincut import MinCutPool
from src.layers import SAGPool
from src.layers.topk import TopKPool

from src.autoencoder.run_topk import upsampling_top_k


from src.modules.ndp import NDP
from src.modules.ndp import preprocess as preprocess_ndp
from src.modules.nmf import NMF
from src.modules.nmf import preprocess as preprocess_nmf

from spektral.utils import convolution

import time
'''
make models.
'''
def make_model_diffpool(F, **kwargs):
    pool = DiffPool(kwargs.get("k"), return_sel=True)
    lift = Lambda(upsampling_with_pinv)
    model = Autoencoder(F, pool, lift, post_procesing=True)
    return model

def pooling_graclus(X, A):
    X, A = preprocess(X, A)
    X_out, A_out, S_out = GRACLUS([X], [A], [0, 1])
    return A_out[0][0], X_out[0], A_out[0][1], S_out[0][0]

def make_model_lapool(F, **kwargs):
    pool = LaPool(shortest_path_reg=False, return_sel=True)
    lift = Lambda(upsampling_with_pinv)
    model = Autoencoder(F, pool, lift)
    return model

def make_model_mincut(F, **kwargs):
    pool = MinCutPool(kwargs.get("k"), return_sel=True)
    lift = Lambda(upsampling_with_pinv)
    model = Autoencoder(F, pool, lift)
    return model

def pooling_ndp(X, A):
    _, L = preprocess_ndp(X, A)
    A_out, S_out = NDP([L], 1)
    return A, X, A_out[0], S_out[0]

def pooling_nmf(X, A):
    _, A_in = preprocess_nmf(X, A)
    A_in = convolution.gcn_filter(A)
    A_out, S_out = NMF([A_in], 0.5)
    return A, X, A_out[0], S_out[0]

def make_model_sagpool(F, **kwargs):
    pool = SAGPool(kwargs.get("ratio"), return_sel=True, return_score=True)
    lift = Lambda(upsampling_top_k)
    model = Autoencoder(F, pool, lift)
    return model

def make_model_topk(F, **kwargs):
    pool = TopKPool(kwargs.get("ratio"), return_sel=True, return_score=True)
    lift = Lambda(upsampling_top_k)
    model = Autoencoder(F, pool, lift)
    return model

'''
global
'''
data_names = ['Grid2d','Ring','Bunny','Airplane','Car','Person','Guitar']
method2param = {
    'diffpool': {
        'method': 'DiffPool',
        'create_model': make_model_diffpool,
        'learning_rate': 5e-4,
        'es_patience': 1000,
        'es_tol': 1e-6,
        'is_nt':False,
    },
    'graclus': {
        'method': 'Graclus',
        'pooling': pooling_graclus,
        'learning_rate': 5e-4,
        'es_patience': 1000,
        'es_tol': 1e-6,
        'is_nt':True,
    },
    'lapool': {
        'method': 'LaPool',
        'create_model': make_model_lapool,
        'learning_rate': 5e-4,
        'es_patience': 1000,
        'es_tol': 1e-6,
        'is_nt': False,
    },
    'mincut': {
        'method': 'MinCut',
        'create_model': make_model_mincut,
        'learning_rate': 5e-4,
        'es_patience': 1000,
        'es_tol': 1e-6,
        'is_nt': False,
    },
    'ndp': {
        'method': 'NDP',
        'pooling': pooling_ndp,
        'learning_rate': 5e-4,
        'es_patience': 1000,
        'es_tol': 1e-6,
        'is_nt': True,
    },
    'nmf': {
        'method': 'NMF',
        'pooling': pooling_nmf,
        'learning_rate': 5e-4,
        'es_patience': 1000,
        'es_tol': 1e-6,
        'is_nt': True,
    },
    'sagpool': {
        'method': 'SAGPool',
        'create_model': make_model_sagpool,
        'learning_rate': 5e-4,
        'es_patience': 1000,
        'es_tol': 1e-6,
        'is_nt': False,
    },
    'topk': {
        'method': 'TopK',
        'create_model': make_model_diffpool,
        'learning_rate': 5e-4,
        'es_patience': 1000,
        'es_tol': 1e-6,
        'is_nt': False,
    },
}

def routine_run_all():
    for data_name in data_names:
        row_name = []
        col_name = ['avg.','std.','time']
        results = []
        for method in method2param.keys():
            print(f'perform {method} on {data_name}...')
            cur_param = {}
            cur_param.update(method2param[method])
            is_nt = cur_param['is_nt']
            del cur_param['is_nt']
            st_time = time.time()
            if is_nt:
                res_row = run_experiment_nt(name=data_name, runs=1, **cur_param)
            else:
                res_row = run_experiment(name=data_name,runs=1, **cur_param)
            res_row = list(res_row)+[time.time() - st_time]
            results.append(res_row)
            row_name.append(method)
        df = pd.DataFrame(results,index=row_name,columns=col_name)
        print(df)
        df.to_csv(f'results/{data_name}_{time.time()}.csv')

if __name__ == '__main__':
    print('benchmark autoencoder.')
    routine_run_all()