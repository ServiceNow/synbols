from haven import haven_utils as hu

# Define exp groups for parameter search
EXP_GROUPS = {'lcfcn':
                hu.cartesian_exp_group({
                    'batch_size':[1],
                    'model': {'name':"lcfcn"},
                    'max_epoch': 100,
                    'dataset': {'name': 'detection'}}),
                
                }