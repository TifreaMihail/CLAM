import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/camelyon16.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = 0

def get_custom_test_ids(loc):
    # read csv
    df = pd.read_csv(loc)
    # remove duplicates / get unique case_ids
    df_unique = df.drop_duplicates(subset='case_id', keep='first')
    # sort by case_id alphabetically
    df_sorted = df_unique.sort_values(by='case_id', ascending=True).reset_index(drop=True)
    # get indices of test slides
    test_indices = df_sorted[df_sorted['slide_id'].str.contains("test")].index

    test_indices_list = (test_indices).tolist()
    return test_indices_list

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        # split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100)) + "CV_sametest_6"
        split_dir = "task_camelyon16_CV_sametest"
        os.makedirs(split_dir, exist_ok=True)
        
        # define the costom test ids from the csv file
        custom_test_ids = get_custom_test_ids('dataset_csv/camelyon16.csv')

        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf, custom_test_ids=custom_test_ids)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



