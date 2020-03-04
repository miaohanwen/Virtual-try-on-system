# DP-VTON
Implementation of eecs545-19F final project, based on cp-VTON.

## GMM train
```
python train.py --name gmm_train_new --stage GMM --workers 4 --save_count 5000 --shuffle
```

## GMM evaluation
```
python test.py --name gmm_traintest_new --stage GMM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint <path_to_module>/gmm_final.pth
```

## TOM train
```
python train.py --name tom_train_new --stage TOM --workers 4 --save_count 5000 --shuffle 
```

## TOM evaluation
```
python test.py --name tom_test_new --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint <path_to_module>/tom_final.pth
```
