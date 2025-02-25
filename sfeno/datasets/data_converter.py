import os
import numpy as np
import pandas as pd
import json

def save_prediction_n_target(pred, targ, data_path, exp_name=''):

    pred_targ_dir_nm = "results"

    pred_nm = 'pred.npy'
    targ_nm = 'targ.npy'

    save_path = os.path.join(data_path, pred_targ_dir_nm)

    check_n_mkdir(save_path)

    for i in range(1000):
        cur_path = os.path.join(save_path, exp_name+str(i))

        if os.path.exists(cur_path):
            # if exist continue
            continue

        else:
            # create new and break
            check_n_mkdir(cur_path)
            break

    pred_path = os.path.join(cur_path, pred_nm)
    targ_path = os.path.join(cur_path, targ_nm)

    print(f'Saving predictions to {cur_path}')
    np.save(pred_path, pred)
    np.save(targ_path, targ)

    return

def augment_n_save(data_path):
    pert, expr, node_idx = augment_for_sfeno(data_path)

    save_path = os.path.join(data_path, "sfeno_data")

    check_n_mkdir(save_path)

    pert_save_path = os.path.join(save_path, "pert.csv")
    expr_save_path = os.path.join(save_path, "expr.csv")
    node_idx_save_path = os.path.join(save_path, "node_Index.json")

    np.savetxt(pert_save_path, pert, delimiter=',')
    np.savetxt(expr_save_path, expr, delimiter=',')
    with open(node_idx_save_path, "w") as fp:  # Pickling
        json.dump(node_idx, fp)


def check_n_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def augment_for_sfeno(data_path):
    cond_path = os.path.join(data_path, 'conds.tsv')
    exp_path = os.path.join(data_path, 'exp.tsv')

    cond_df = pd.read_csv(cond_path, sep='\t', header=0)
    cond_df = cond_df.iloc[:, 1:]
    exp_df = pd.read_csv(exp_path, sep='\t', header=0)
    exp_df = exp_df.iloc[:, 1:]

    cond_ls = list(cond_df.columns)
    exp_ls = list(exp_df.columns)

    cond_set = set(cond_ls)
    exp_set = set(exp_ls)

    intersect_ls = list(cond_set.intersection(exp_set))

    cond_without_itst = cond_ls.copy()
    exp_without_itst = exp_ls.copy()

    for itst_node in intersect_ls:
        cond_without_itst.remove(itst_node)
        exp_without_itst.remove(itst_node)

    correct_order = cond_without_itst + intersect_ls + exp_without_itst

    n_node = len(correct_order)

    arranged_cond = cond_df[intersect_ls + cond_without_itst]
    arranged_exp = exp_df[exp_without_itst + intersect_ls]

    # add padding on the left
    result_cond = np.pad(arranged_cond.to_numpy(), ((0, 0), (len(exp_without_itst), 0)), constant_values=0)
    result_exp = np.pad(arranged_exp.to_numpy(), ((0, 0), (0, len(cond_without_itst))), constant_values=0)

    return result_cond, result_exp, correct_order


for dir_name in os.listdir():
    if os.path.isdir(dir_name):
        print(dir_name)

    try:
        augment_n_save(dir_name)
    except:
        pass