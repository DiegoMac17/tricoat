from all_data_dataset import all_data_dataset
# from dataset_old import all_data_dataset
from hyperparm_tunning_new_tune import tuner
from test import test_model
from sklearn.model_selection import StratifiedKFold, train_test_split

from hyperparm_tunning_new_tune import tuner, train
from train_test_no_tune import train_test_no_tune

import numpy as np

def full_exp(paths, args, config):
    results_dict_all_splits = {}

    # Load data
    all_inputs = all_data_dataset(paths, args)
   
    # Generate splits using sklearn
    skf_outer = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # for i, (train_index, test_index) in enumerate(skf_outer.split(all_inputs[:][0], all_inputs[:][3])):
    for i, (train_index, test_index) in enumerate(skf_outer.split(np.zeros(len(all_inputs.get_labels())), all_inputs.get_labels())):
        results_dict_per_test_set = {}
        skf_inner = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        # for j, (train_index_sub, val_index) in enumerate(skf_inner.split(train_index, all_inputs[train_index][3])):
        for j, (train_index_sub, val_index) in enumerate(skf_inner.split(train_index, all_inputs.get_labels(train_index))):
            train_index_sub = train_index[train_index_sub]
            val_index = train_index[val_index]
            assert np.intersect1d(train_index_sub, val_index).size == 0 and np.intersect1d(train_index_sub, test_index).size == 0 and np.intersect1d(val_index, test_index).size == 0 
            # Hyperparam tunning (Train - validate)
            all_inputs.set_trainset_mean_std(train_index)
            all_inputs.normalize_data(train_index_sub, val_index, test_index)
            if args['tune_flag']:
                results_dict_per_test_set[j] = tuner(all_inputs, train_index_sub, val_index, test_index, paths, args)
            else:
                results_dict_per_test_set[j] = train_test_no_tune(all_inputs, train_index_sub, val_index, test_index, paths, args, config)
        
        print('#'*40)
        print(f'Finished 5-fold cv on test set: {i}')

        results_dict_all_splits[i] = results_dict_per_test_set
        
    print('#'*40)
    print('Finished 10-fold cross testing on test set')
    return results_dict_all_splits

    # # Train - Test splits
    # X_train = 
    # Y_train 
    # X_test
    # Y_test
    # # Train - Validation splits
    # X_train_v
    # Y_train_v
    # X_val
    # Y_val


    # train_val_results_dict, trained_model = tuner(X_train_v, Y_train_v, X_val, Y_val)

    # Testing 
    # results_dict = test_model(test_loader, trained_model)

def single_run(paths, args, config):
    results_dict_all_splits = {}

    # Load data
    all_inputs = all_data_dataset(paths, args)
   
    # Generate splits using sklearn
    ptidxs_train, ptidxs_test, y_train, y_test, = train_test_split(all_inputs.get_ptidxs(), all_inputs.get_labels(), stratify=all_inputs.get_labels(), random_state=42)
    ptidxs_train_sub, ptidxs_val, y_train_sub, y_val = train_test_split(ptidxs_train, y_train, stratify=y_train, random_state=42)

    # TODO: get indices for x based on y. probably need to create a get_indices or par_ids() function as well
    train_index_sub = ptidxs_train_sub
    val_index = ptidxs_val
    test_index = ptidxs_test

    all_inputs.set_trainset_mean_std(train_index)
    all_inputs.normalize_data(train_index_sub, val_index, test_index)
    # Run model - no hyperparam tuning
    results_dict_per_test_set[0] = train_test_no_tune(all_inputs, train_index_sub, val_index, test_index, paths, args, config)
    results_dict_all_splits[0] = results_dict_per_test_set



    for i, (train_index, test_index) in enumerate(skf_outer.split(np.zeros(len(all_inputs.get_labels())), all_inputs.get_labels())):
        results_dict_per_test_set = {}
        skf_inner = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        # for j, (train_index_sub, val_index) in enumerate(skf_inner.split(train_index, all_inputs[train_index][3])):
        for j, (train_index_sub, val_index) in enumerate(skf_inner.split(train_index, all_inputs.get_labels(train_index))):
            train_index_sub = train_index[train_index_sub]
            val_index = train_index[val_index]
            assert np.intersect1d(train_index_sub, val_index).size == 0 and np.intersect1d(train_index_sub, test_index).size == 0 and np.intersect1d(val_index, test_index).size == 0 
            # Hyperparam tunning (Train - validate)
            all_inputs.set_trainset_mean_std(train_index)
            all_inputs.normalize_data(train_index_sub, val_index, test_index)
            if args['tune_flag']:
                results_dict_per_test_set[j] = tuner(all_inputs, train_index_sub, val_index, test_index, paths, args)
            else:
                results_dict_per_test_set[j] = train_test_no_tune(all_inputs, train_index_sub, val_index, test_index, paths, args, config)
        
        print('#'*40)
        print(f'Finished 5-fold cv on test set: {i}')

        results_dict_all_splits[i] = results_dict_per_test_set
        
    print('#'*40)
    print('Finished 10-fold cross testing on test set')
    return results_dict_all_splits