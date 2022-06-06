import scipy.stats

from BIQA_CL import parse_config, main
import os
import numpy as np
from utils import unified_metric

#some global setting
dataset = ['live', 'csiq', 'bid', 'clive', 'koniq10k', 'kadid10k']
default_id2dataset = {}
default_dataset2id = {}
task_id = [0, 1, 2, 3, 4, 5]
for id, task in zip(task_id, dataset):
    default_id2dataset[id] = task
    default_dataset2id[task] = id

task_id = [0, 1, 2, 3, 4, 5]
seq_len = len(dataset)
eval_dict = {"live": False,
             "csiq": False,
             "bid": False,
             "clive": False,
             "koniq10k": False,
             "kadid10k": False}

config = parse_config()
trainset_dict = {"live": config.live_set,
                 "csiq": config.csiq_set,
                 "bid": config.bid_set,
                 "clive": config.clive_set,
                 "koniq10k": config.koniq10k_set,
                 "kadid10k": config.kadid10k_set}

cluster_dict = {"live": 128,
                 "csiq": 128,
                 "bid": 128,
                 "clive": 128,
                 "koniq10k": 128,
                 "kadid10k": 128}


if config.reverse:
    dataset.reverse()

id2dataset = {}
dataset2id = {}
current2default = {}
for id, task in zip(task_id, dataset):
    id2dataset[id] = task
    dataset2id[task] = id
    current2default[id] = default_dataset2id[task]

base_ckpt_path = config.ckpt_path

def SH_CL():
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            print("-------Start training on " + task + "---------")
            config = parse_config()
            config.n_task = 1
            config.shared_head = True
            base_ckpt_path = config.ckpt_path
            config.dataset = dataset
            config.id2dataset = id2dataset
            split = i + 1
            config.split = split
            config.lwf = False
            config.task_id = idx
            config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.indicator = idx+1
            config.trainset_dict = trainset_dict
            if idx > 0:
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+dataset[idx-1]) #resume from the previous task
                config.resume = True
                config.resume_best = True
                config.resume_new = True
            else:
                config.ckpt_resume_path = config.ckpt_path

            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train.txt'
            eval_dict[task] = True
            config.eval_dict = eval_dict
            if idx == 0:
                main(config)
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_' + task)
                config.resume_new = False
            # stage2: fine-tuning the whole network
            #config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.resume_best = False
            config.fc = False
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.max_epochs = config.max_epochs2
            config.batch_size = config.batch_size2
            if idx > 0:
                config.max_epochs = 6
                config.lr = 2e-5
            main(config)

def SH_CL_replay():
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            print("-------Start training on " + task + "---------")
            config = parse_config()
            config.n_task = 1
            config.shared_head = True
            base_ckpt_path = config.ckpt_path
            config.dataset = dataset
            config.id2dataset = id2dataset
            split = i + 1
            config.split = split
            config.lwf = False
            config.new_replay = False
            config.GDumb = False
            config.task_id = idx
            config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.indicator = idx+1
            config.trainset_dict = trainset_dict
            if idx > 0:
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+dataset[idx-1]) #resume from the previous task
                config.resume = True
                config.resume_best = True
                config.resume_new = True
                config.replay = True
                config.save_replay = True
            else:
                config.ckpt_resume_path = config.ckpt_path
            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train.txt'
            eval_dict[task] = True
            config.eval_dict = eval_dict
            if idx == 0:
                config.replay = False  # do not replay during the warm-up phase
                config.save_replay = False
                main(config)
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_' + task)
                config.resume_new = False
            else:
                config.max_epochs2 = config.max_epochs2 - config.max_epochs #do not warm up in the SH-CL setting
            # stage2: fine-tuning the whole network
            config.resume_best = True
            config.save_replay = True
            config.fc = False
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.max_epochs = config.max_epochs2
            config.batch_size = config.batch_size2

            if idx > 0:
                config.lr = 2e-5
            main(config)

def SL():
    config = parse_config()
    config.lwf = False
    config.shared_head = True
    config.task_id = 0
    config.n_task = 1
    config.dataset = dataset
    config.id2dataset = id2dataset
    config.replay = False
    for i in range(0, 1):
        split = i + 1
        config.split = split

        print("-------Start training on LIVE---------")
        eval_dict['live'] = True
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_live')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 1
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        config.trainset = config.live_set
        config.train_txt = 'live_train.txt'
        config.eval_live = True
        config.eval_dict = eval_dict
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        #config.lr = 3e-5
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on LIVE---------")


        print("-------Start training on CSIQ---------")
        config = parse_config()
        config.id2dataset = id2dataset
        eval_dict['csiq'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_csiq')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 2
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        config.trainset = config.csiq_set
        config.train_txt = 'csiq_train.txt'
        # config.eval_live = True
        config.eval_csiq = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on CSIQ---------")

        print("-------Start training on BID---------")
        config = parse_config()
        config.id2dataset = id2dataset
        eval_dict['bid'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_bid')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 3
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        config.trainset = config.bid_set
        config.train_txt = 'bid_train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on BID---------")


        print("-------Start training on CLIVE---------")
        config = parse_config()
        config.id2dataset = id2dataset
        eval_dict['clive'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_clive')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 4
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        config.trainset = config.clive_set
        config.train_txt = 'clive_train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        config.eval_clive = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on CLIVE---------")


        print("-------Start training on KONIQ10K---------")
        config = parse_config()
        config.id2dataset = id2dataset
        eval_dict['koniq10k'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_koniq10k')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 5
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        config.trainset = config.koniq10k_set
        config.train_txt = 'koniq10k_train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        config.eval_clive = True
        config.eval_koniq10k = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on KONIQ01K---------")


        print("-------Start training on KADID10K---------")
        config = parse_config()
        config.id2dataset = id2dataset
        eval_dict['kadid10k'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_kadid10k')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 6
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        config.trainset = config.kadid10k_set
        config.train_txt = 'kadid10k_train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        config.eval_clive = True
        config.eval_koniq10k = True
        config.eval_kadid10k = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on KADID10K---------")


def LwF():
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            eval_dict[task] = True
            print("-------Start training on " + task + "---------")
            config = parse_config()
            split = i + 1
            config.split = split
            config.lwf = False
            config.task_id = idx
            config.dataset = dataset
            config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.indicator = idx+1
            config.replay = False
            config.icarl = False
            if idx > 0:
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+dataset[idx-1]) #resume from the previous task
                config.resume = True
                config.resume_best = True
                config.resume_new = True
            else:
                config.ckpt_resume_path = config.ckpt_path

            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train.txt'
            config.eval_dict = eval_dict
            config.id2dataset = id2dataset
            main(config)
            # stage2: fine-tuning the whole network
            if idx > 0:
                config.lwf = True
            config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.resume_best = False
            config.resume_new = False
            config.fc = False
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.max_epochs = config.max_epochs2
            config.batch_size = config.batch_size2
            main(config)


def LwF_Replay():
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            eval_dict[task] = True
            print("-------Start training on " + task + "---------")
            config = parse_config()
            split = i + 1
            config.split = split
            config.lwf = False
            config.task_id = idx
            config.dataset = dataset
            config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.indicator = idx+1
            config.trainset_dict = trainset_dict
            if idx > 0:
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+dataset[idx-1]) #resume from the previous task
                config.resume = True
                config.resume_best = True
                config.resume_new = True
            else:
                config.ckpt_resume_path = config.ckpt_path

            #hack
            # if idx == 1:
            #     config.resume_best = True

            config.replay = False # do not replay during the warm-up phase
            config.save_replay = False
            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train.txt'
            config.eval_dict = eval_dict
            config.id2dataset = id2dataset
            main(config)
            # stage2: fine-tuning the whole network
            if idx > 0:
                config.lwf = True
                config.replay = True
            config.save_replay = True
            config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.resume_best = False
            config.resume_new = False
            config.fc = False
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.max_epochs = config.max_epochs2
            config.batch_size = config.batch_size2
            main(config)



def MH_CL():
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            print("-------Start training on " + task + "---------")
            config = parse_config()
            config.dataset = dataset
            split = i + 1
            config.split = split
            config.lwf = False
            config.task_id = idx
            config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.indicator = idx+1
            config.dataset2id = dataset2id
            config.id2dataset = id2dataset
            config.trainset_dict = trainset_dict
            if idx > 0:
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+dataset[idx-1]) #resume from the previous task
                config.resume = True
                config.resume_best = True
                config.resume_new = True
            else:
                config.ckpt_resume_path = config.ckpt_path

            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train.txt'
            eval_dict[task] = True
            config.eval_dict = eval_dict
            main(config)
            # stage2: fine-tuning the whole network
            config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.resume_best = False
            config.resume_new = False
            config.fc = False
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.max_epochs = config.max_epochs2
            config.batch_size = config.batch_size2
            main(config)


def MH_CL_Replay():
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            eval_dict[task] = True
            print("-------Start training on " + task + "---------")
            config = parse_config()
            split = i + 1
            config.split = split
            config.lwf = False
            config.icarl = False
            config.new_replay = False
            config.GDumb = False
            config.task_id = idx
            config.dataset = dataset
            config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.indicator = idx+1
            config.trainset_dict = trainset_dict
            if idx > 0:
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+dataset[idx-1]) #resume from the previous task
                config.resume = True
                config.resume_best = True
                config.resume_new = True
            else:
                config.ckpt_resume_path = config.ckpt_path
            config.replay = False # do not replay during the warm-up phase
            config.save_replay = False
            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train.txt'
            config.eval_dict = eval_dict
            config.id2dataset = id2dataset
            config.reg_weight = 1
            main(config)
            # stage2: fine-tuning the whole network
            if idx > 0:
                config.replay = True
            config.save_replay = True
            config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.resume_best = False
            config.resume_new = False
            config.fc = False
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.max_epochs = config.max_epochs2
            config.batch_size = config.batch_size2
            main(config)


def eval_each_head():
    print('-----------evaluate with oracle--------------')
    eval_dict = {"live": False,
                 "csiq": False,
                 "bid": False,
                 "clive": False,
                 "koniq10k": False,
                 "kadid10k": False}

    config = parse_config()
    config.dataset = dataset
    config.dataset2id = dataset2id
    config.id2dataset = id2dataset
    config.current2default = current2default
    config.train_expert = False
    config.train_kmeans = False

    config.replay = False

    for i in range(0, 1):
        config.base_ckpt_path = os.path.join(base_ckpt_path, str(config.split))
        srcc_matrix = np.zeros((seq_len, seq_len))
        srcc_matrix2 = np.zeros((seq_len, seq_len))
        s_srcc_matrix = np.ones((seq_len, seq_len))
        s_srcc_matrix2 = np.ones((seq_len, seq_len))

        predictions_test = {}
        predictions_val = {}
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            print("-------Start testing on " + task + "---------")
            config.eval_each = True
            config.train = False
            config.resume = True
            config.resume_best = True

            config.indicator = idx + 1
            config.task_id = idx
            config.subfolder = 'train_on_'+task
            config.ckpt_path = os.path.join(config.base_ckpt_path, config.subfolder)
            config.ckpt_resume_path = config.ckpt_path
            eval_dict[task] = True
            config.eval_dict = eval_dict
            config.current_task_id = idx
            if idx > 0:
                config.verbose = False
            srcc, srcc_val, all_hat_test, all_hat_val = main(config)
            for j in range(0, idx + 1):
                srcc_matrix[idx, j] = srcc[id2dataset[j]]
                srcc_matrix2[idx, j] = srcc_val[id2dataset[j]]
                task_name = config.dataset[j]
                append_name = task_name + '_' + str(idx)
                predictions_test[append_name] = all_hat_test[task_name]
                predictions_val[append_name] = all_hat_val[task_name]
        #
        initial_test = {}
        initial_val = {}
        for idx, task_name in enumerate(config.dataset):
            each_name = task_name + '_' + str(idx)
            initial_test[each_name] = predictions_test[each_name]
            initial_val[each_name] = predictions_val[each_name]

        for idx, task_name in enumerate(config.dataset):
            if idx == 0:
                continue #skip the first task
            else:
                subset = config.dataset[0:idx]
                for j, sub_name in enumerate(subset):
                    #index = idx + j
                    each_name = sub_name + '_' + str(idx)
                    hat_test = predictions_test[each_name]
                    hat_val = predictions_val[each_name]

                    initial_name_test = sub_name + '_' + str(j)
                    initial_name_val = sub_name + '_' + str(j)
                    initial_hat_test = predictions_test[initial_name_test]
                    initial_hat_val = predictions_val[initial_name_val]

                    s_srcc_matrix[idx, j] = scipy.stats.mstats.spearmanr(x=hat_test, y=initial_hat_test)[0]
                    s_srcc_matrix2[idx, j] = scipy.stats.mstats.spearmanr(x=hat_val, y=initial_hat_val)[0]

        stability_test, s_t_test, _, _ = unified_metric(s_srcc_matrix, mode=0)
        stability_val, s_t_val, _, _ = unified_metric(s_srcc_matrix2, mode=0)

        plasticity_test, p_t_test, avg_test, avg_test_t = unified_metric(srcc_matrix, mode=1)
        plasticity_val, p_t_val, avg_val, avg_val_t = unified_metric(srcc_matrix2, mode=1)

        print('stability test {}'.format(stability_test))
        print(s_t_test)
        print('stability val {}'.format(stability_val))
        print(s_t_val)

        print('plasticity test {}'.format(plasticity_test))
        print(p_t_test)
        print('plasticity val {}'.format(plasticity_val))
        print(p_t_val)

        print('final average test {}'.format(avg_test))
        print(avg_test_t)
        print('final average val {}'.format(avg_val))
        print(avg_val_t)

        print('mSRCC: {}'.format(avg_test_t[-1]))
        print('mPI: {}'.format(plasticity_test))
        print('mSI: {}'.format(stability_test))
        print('mPSI: {}'.format((stability_test + plasticity_test)/2))

def eval_weight_head():
    print('-----------evaluate with AW--------------')
    eval_dict = {"live": False,
                 "csiq": False,
                 "bid": False,
                 "clive": False,
                 "koniq10k": False,
                 "kadid10k": False}

    config = parse_config()
    config.dataset = dataset
    config.task_id = task_id
    config.dataset2id = dataset2id
    config.train_kmeans = False
    config.train_expert = False
    config.id2dataset = id2dataset
    config.current2default = current2default

    config.replay = False

    for i in range(0, 1):
        config.base_ckpt_path = os.path.join(base_ckpt_path, str(config.split))
        srcc_matrix = np.zeros((seq_len, seq_len))
        srcc_matrix2 = np.zeros((seq_len, seq_len))
        s_srcc_matrix = np.ones((seq_len, seq_len))
        s_srcc_matrix2 = np.ones((seq_len, seq_len))

        predictions_test = {}
        predictions_val = {}
        config.id2dataset = id2dataset
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            print("-------Start testing on " + task + "---------")
            config.eval_each = False
            config.train = False
            config.resume = True
            config.resume_best = True

            config.indicator = idx + 1
            config.task_id = idx
            config.subfolder = 'train_on_'+task
            config.ckpt_path = os.path.join(config.base_ckpt_path, config.subfolder)
            config.ckpt_resume_path = config.ckpt_path
            eval_dict[task] = True
            config.eval_dict = eval_dict
            config.current_task_id = idx
            if idx > 0:
                config.verbose = False
                config.weighted_output = True
            srcc, srcc_val, all_hat_test, all_hat_val = main(config)
            for j in range(0, idx + 1):
                srcc_matrix[idx, j] = srcc[id2dataset[j]]
                srcc_matrix2[idx, j] = srcc_val[id2dataset[j]]
                task_name = config.dataset[j]
                append_name = task_name + '_' + str(idx)
                predictions_test[append_name] = all_hat_test[task_name]
                predictions_val[append_name] = all_hat_val[task_name]

        initial_test = {}
        initial_val = {}
        for idx, task_name in enumerate(config.dataset):
            each_name = task_name + '_' + str(idx)
            initial_test[each_name] = predictions_test[each_name]
            initial_val[each_name] = predictions_val[each_name]

        for idx, task_name in enumerate(config.dataset):
            if idx == 0:
                continue  # skip the first task
            else:
                subset = config.dataset[0:idx]
                for j, sub_name in enumerate(subset):
                        # index = idx + j
                    each_name = sub_name + '_' + str(idx)
                    hat_test = predictions_test[each_name]
                    hat_val = predictions_val[each_name]

                    initial_name_test = sub_name + '_' + str(j)
                    initial_name_val = sub_name + '_' + str(j)
                    initial_hat_test = predictions_test[initial_name_test]
                    initial_hat_val = predictions_val[initial_name_val]

                    s_srcc_matrix[idx, j] = scipy.stats.mstats.spearmanr(x=hat_test, y=initial_hat_test)[0]
                    s_srcc_matrix2[idx, j] = scipy.stats.mstats.spearmanr(x=hat_val, y=initial_hat_val)[0]

        stability_test, s_t_test, _, _ = unified_metric(s_srcc_matrix, mode=0)
        stability_val, s_t_val, _, _ = unified_metric(s_srcc_matrix2, mode=0)

        plasticity_test, p_t_test, avg_test, avg_test_t = unified_metric(srcc_matrix, mode=1)
        plasticity_val, p_t_val, avg_val, avg_val_t = unified_metric(srcc_matrix2, mode=1)

        print('stability test {}'.format(stability_test))
        print(s_t_test)
        print('stability val {}'.format(stability_val))
        print(s_t_val)

        print('plasticity test {}'.format(plasticity_test))
        print(p_t_test)
        print('plasticity val {}'.format(plasticity_val))
        print(p_t_val)

        print('final average test {}'.format(avg_test))
        print(avg_test_t)
        print('final average val {}'.format(avg_val))
        print(avg_val_t)

        print('mSRCC: {}'.format(avg_test_t[-1]))
        print('mPI: {}'.format(plasticity_test))
        print('mSI: {}'.format(stability_test))
        print('mPSI: {}'.format((stability_test + plasticity_test)/2))



def eval_last_head():
    print('-----------evaluate with the latest head--------------')
    eval_dict = {"live": False,
                 "csiq": False,
                 "bid": False,
                 "clive": False,
                 "koniq10k": False,
                 "kadid10k": False}


    config = parse_config()
    config.current2default = current2default
    config.dataset = dataset
    config.task_id = task_id
    config.dataset2id = dataset2id
    config.train_kmeans = False
    config.train_expert = False
    config.id2dataset = id2dataset

    config.replay = False

    for i in range(0, 1):
        config.base_ckpt_path = os.path.join(base_ckpt_path, str(config.split))
        srcc_matrix = np.zeros((seq_len, seq_len))
        srcc_matrix2 = np.zeros((seq_len, seq_len))

        s_srcc_matrix = np.ones((seq_len, seq_len))
        s_srcc_matrix2 = np.ones((seq_len, seq_len))

        predictions_test = {}
        predictions_val = {}

        config.id2dataset = id2dataset
        for idx, task in enumerate(dataset):
            print("-------Start testing on " + task + "---------")
            config.eval_each = False
            config.train = False
            config.resume = True
            config.resume_best = True

            config.indicator = idx + 1
            config.task_id = idx
            config.subfolder = 'train_on_'+task
            config.ckpt_path = os.path.join(base_ckpt_path, str(i+1), config.subfolder)
            config.ckpt_resume_path = config.ckpt_path
            eval_dict[task] = True
            config.eval_dict = eval_dict
            config.current_task_id = idx
            if idx > 0:
                config.verbose = False
            srcc, srcc_val, all_hat_test, all_hat_val = main(config)
            for j in range(0, idx+1):
                srcc_matrix[idx, j] = srcc[id2dataset[j]]
                srcc_matrix2[idx, j] = srcc_val[id2dataset[j]]
                task_name = config.dataset[j]
                append_name = task_name + '_' + str(idx)
                predictions_test[append_name] = all_hat_test[task_name]
                predictions_val[append_name] = all_hat_val[task_name]

        initial_test = {}
        initial_val = {}
        for idx, task_name in enumerate(config.dataset):
            each_name = task_name + '_' + str(idx)
            initial_test[each_name] = predictions_test[each_name]
            initial_val[each_name] = predictions_val[each_name]

        for idx, task_name in enumerate(config.dataset):
            if idx == 0:
                continue #skip the first task
            else:
                subset = config.dataset[0:idx]
                for j, sub_name in enumerate(subset):
                    #index = idx + j
                    each_name = sub_name + '_' + str(idx)
                    hat_test = predictions_test[each_name]
                    hat_val = predictions_val[each_name]

                    initial_name_test = sub_name + '_' + str(j)
                    initial_name_val = sub_name + '_' + str(j)
                    initial_hat_test = predictions_test[initial_name_test]
                    initial_hat_val = predictions_val[initial_name_val]

                    s_srcc_matrix[idx, j] = scipy.stats.mstats.spearmanr(x=hat_test, y=initial_hat_test)[0]
                    s_srcc_matrix2[idx, j] = scipy.stats.mstats.spearmanr(x=hat_val, y=initial_hat_val)[0]

        stability_test, s_t_test, _, _ = unified_metric(s_srcc_matrix, mode=0)
        stability_val, s_t_val, _, _ = unified_metric(s_srcc_matrix2, mode=0)

        plasticity_test, p_t_test, avg_test, avg_test_t = unified_metric(srcc_matrix, mode=1)
        plasticity_val, p_t_val, avg_val, avg_val_t = unified_metric(srcc_matrix2, mode=1)

        print('stability test {}'.format(stability_test))
        print(s_t_test)
        print('stability val {}'.format(stability_val))
        print(s_t_val)

        print('plasticity test {}'.format(plasticity_test))
        print(p_t_test)
        print('plasticity val {}'.format(plasticity_val))
        print(p_t_val)

        print('final average test {}'.format(avg_test))
        print(avg_test_t)
        print('final average val {}'.format(avg_val))
        print(avg_val_t)

        print('mSRCC: {}'.format(avg_test_t[-1]))
        print('mPI: {}'.format(plasticity_test))
        print('mSI: {}'.format(stability_test))
        print('mPSI: {}'.format((stability_test + plasticity_test)/2))


def eval_single_head():
    config = parse_config()
    config.dataset = dataset
    config.task_id = task_id
    config.dataset2id = dataset2id
    config.train_kmeans = False
    config.n_task = 1
    config.shared_head = True
    config.train_expert = False
    config.train_kmeans = False
    config.replay = False

    for i in range(0, 1):
        config.base_ckpt_path = os.path.join(base_ckpt_path, str(config.split))
        srcc_matrix = np.zeros((seq_len, seq_len))
        srcc_matrix2 = np.zeros((seq_len, seq_len))

        s_srcc_matrix = np.ones((seq_len, seq_len))
        s_srcc_matrix2 = np.ones((seq_len, seq_len))

        predictions_test = {}
        predictions_val = {}

        config.id2dataset = id2dataset
        for idx, task in enumerate(dataset):
            print("-------Start testing on " + task + "---------")
            config.eval_each = False
            config.train = False
            config.resume = True
            config.resume_best = True
            config.indicator = idx + 1
            config.task_id = idx
            config.subfolder = 'train_on_'+task
            config.ckpt_path = os.path.join(base_ckpt_path, str(i+1), config.subfolder)
            config.ckpt_resume_path = config.ckpt_path
            eval_dict[task] = True
            config.eval_dict = eval_dict
            config.current_task_id = idx
            if idx > 0:
                config.verbose = False
            srcc, srcc_val, all_hat_test, all_hat_val = main(config)
            for j in range(0, idx+1):
                srcc_matrix[idx, j] = srcc[id2dataset[j]]
                srcc_matrix2[idx, j] = srcc_val[id2dataset[j]]
                task_name = config.dataset[j]
                append_name = task_name + '_' + str(idx)
                predictions_test[append_name] = all_hat_test[task_name]
                predictions_val[append_name] = all_hat_val[task_name]

        initial_test = {}
        initial_val = {}
        for idx, task_name in enumerate(config.dataset):
            each_name = task_name + '_' + str(idx)
            initial_test[each_name] = predictions_test[each_name]
            initial_val[each_name] = predictions_val[each_name]

        for idx, task_name in enumerate(config.dataset):
            if idx == 0:
                continue #skip the first task
            else:
                subset = config.dataset[0:idx]
                for j, sub_name in enumerate(subset):
                    #index = idx + j
                    each_name = sub_name + '_' + str(idx)
                    hat_test = predictions_test[each_name]
                    hat_val = predictions_val[each_name]

                    initial_name_test = sub_name + '_' + str(j)
                    initial_name_val = sub_name + '_' + str(j)
                    initial_hat_test = predictions_test[initial_name_test]
                    initial_hat_val = predictions_val[initial_name_val]

                    s_srcc_matrix[idx, j] = scipy.stats.mstats.spearmanr(x=hat_test, y=initial_hat_test)[0]
                    s_srcc_matrix2[idx, j] = scipy.stats.mstats.spearmanr(x=hat_val, y=initial_hat_val)[0]

        stability_test, s_t_test, _, _ = unified_metric(s_srcc_matrix, mode=0)
        stability_val, s_t_val, _, _ = unified_metric(s_srcc_matrix2, mode=0)

        plasticity_test, p_t_test, avg_test, avg_test_t = unified_metric(srcc_matrix, mode=1)
        plasticity_val, p_t_val, avg_val, avg_val_t = unified_metric(srcc_matrix2, mode=1)

        print('stability test {}'.format(stability_test))
        print(s_t_test)
        print('stability val {}'.format(stability_val))
        print(s_t_val)

        print('plasticity test {}'.format(plasticity_test))
        print(p_t_test)
        print('plasticity val {}'.format(plasticity_val))
        print(p_t_val)

        print('final average test {}'.format(avg_test))
        print(avg_test_t)
        print('final average val {}'.format(avg_val))
        print(avg_val_t)

        print('mSRCC: {}'.format(avg_test_t[-1]))
        print('mPI: {}'.format(plasticity_test))
        print('mSI: {}'.format(stability_test))
        print('mPSI: {}'.format((stability_test + plasticity_test)/2))

def train_kmeans_after_training():
    config = parse_config()
    config.task_id = task_id
    config.dataset2id = dataset2id
    config.id2dataset = id2dataset
    config.train_kmeans = True
    config.replay = False
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            print("-------Start training kmeans on " + task + "---------")
            config.eval_each = True
            config.train = False
            config.resume = True
            config.resume_best = True
            config.indicator = idx + 1
            config.task_id = idx
            config.subfolder = 'train_on_'+task
            config.ckpt_path = os.path.join(base_ckpt_path, str(i+1), config.subfolder)
            config.ckpt_resume_path = config.ckpt_path
            eval_dict[task] = True
            config.eval_dict = eval_dict
            if idx > 0:
                config.verbose = False
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train_score.txt'
            config.num_cluster = cluster_dict[task]
            config.ranking=False
            main(config)


def train_experts_after_training():
    config = parse_config()
    config.task_id = task_id
    config.dataset2id = dataset2id
    config.id2dataset = id2dataset
    config.train_kmeans = False
    config.train_expert = True
    config.replay = False
    for i in range(0, 1):        #
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            print("-------Start training expert on " + task + "---------")
            config.eval_each = True
            config.train = False
            config.resume = True
            config.resume_best = True
            config.indicator = idx + 1
            config.task_id = idx
            config.subfolder = 'train_on_'+task
            config.ckpt_path = os.path.join(base_ckpt_path, str(i+1), config.subfolder)
            config.ckpt_resume_path = config.ckpt_path
            eval_dict[task] = True
            config.eval_dict = eval_dict
            if idx > 0:
                config.verbose = False
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train_score.txt'
            config.num_cluster = cluster_dict[task]
            config.ranking=False
            main(config)


def JL():
    config = parse_config()
    config.lwf = False
    config.shared_head = True
    config.task_id = 0
    config.n_task = 1
    config.dataset = dataset
    config.id2dataset = id2dataset
    config.replay = False
    config.save_replay = False
    config.JL = True
    for i in range(0, 1):
        split = i + 1
        config.split = split

        print("-------Start joint training")
        eval_dict['live'] = True
        eval_dict['csiq'] = True
        eval_dict['bid'] = True
        eval_dict['clive'] = True
        eval_dict['koniq10k'] = True
        eval_dict['kadid10k'] = True
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'JL')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 1
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        #config.trainset = config.live_set
        config.train_txt = 'train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        config.eval_clive = True
        config.eval_koniq10k = True
        config.eval_kadid10k = True
        config.eval_dict = eval_dict
        config.lr = 2e-4
        config.max_epochs2 = 12
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        #config.lr = 3e-5
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on LIVE---------")


def Reg_CL():
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            if idx < 0:
                continue
            eval_dict[task] = True
            print("-------Start training on " + task + "---------")
            config = parse_config()
            split = i + 1
            config.split = split
            config.lwf = False
            config.task_id = idx
            config.dataset = dataset
            config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.indicator = idx+1
            config.replay = False
            config.lwf = False
            if idx > 0:
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+dataset[idx-1]) #resume from the previous task
                config.resume = True
                config.resume_best = True
                config.resume_new = True
            else:
                config.ckpt_resume_path = config.ckpt_path

            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            config.trainset = trainset_dict[task]
            config.train_txt = task + '_train.txt'
            config.eval_dict = eval_dict
            config.id2dataset = id2dataset
            config.reg_trigger = True
            main(config)
            # stage2: fine-tuning the whole network
            config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.resume_best = False
            config.resume_new = False
            config.fc = False
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.max_epochs = config.max_epochs2
            config.batch_size = config.batch_size2
            main(config)


def JL_CL():
    for i in range(0, 1):
        for idx, task in enumerate(dataset):
            print("-------Start training on " + task + "---------")
            config = parse_config()
            config.JL_CL = True
            config.n_task = 1
            config.shared_head = True
            base_ckpt_path = config.ckpt_path
            config.dataset = dataset
            config.id2dataset = id2dataset
            split = i + 1
            config.split = split
            config.lwf = False
            config.task_id = idx
            config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.indicator = idx+1
            config.trainset_dict = trainset_dict
            if idx > 0:
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+dataset[idx-1]) #resume from the previous task
                config.resume = True
                config.resume_best = True
                config.resume_new = True
            else:
                config.ckpt_resume_path = config.ckpt_path

            if not os.path.exists(config.ckpt_path):
                os.makedirs(config.ckpt_path)
            #config.trainset = trainset_dict[task]
            #config.train_txt = task + '_train.txt'
            config.train_txt = 'train.txt'
            eval_dict[task] = True
            config.eval_dict = eval_dict
            if idx == 0:
                main(config)
                config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_' + task)
                config.resume_new = False
            # stage2: fine-tuning the whole network
            #config.ckpt_resume_path = os.path.join(base_ckpt_path, str(config.split), 'train_on_'+task)
            config.resume_best = False
            config.fc = False
            config.resume = True  # resuming from the latest checkpoint of stage 1
            config.max_epochs = config.max_epochs2
            config.batch_size = config.batch_size2
            if idx > 0:
                config.max_epochs = 6
                config.lr = 2e-5
            main(config)


def GDumb():
    config = parse_config()
    config.lwf = False
    config.shared_head = True
    config.task_id = 0
    config.n_task = 1
    config.dataset = dataset
    config.id2dataset = id2dataset
    config.replay = False
    config.GDumb = True
    for i in range(0, 1):
        split = i + 1
        config.split = split

        print("-------Start training on LIVE---------")
        eval_dict['live'] = True
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'gdumb/1')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 1
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        #config.trainset = os.path.join(config.trainset, 'gdumb/1')
        config.train_txt = 'gdumb_train.txt'
        config.eval_live = True
        config.eval_dict = eval_dict
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        #config.lr = 3e-5
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on LIVE---------")


        print("-------Start training on CSIQ---------")
        config = parse_config()
        config.GDumb = True
        config.id2dataset = id2dataset
        eval_dict['csiq'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'gdumb/2')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 2
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        #config.trainset = os.path.join(config.trainset, 'gdumb/2')
        config.train_txt = 'gdumb_train.txt'
        # config.eval_live = True
        config.eval_csiq = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on CSIQ---------")

        print("-------Start training on BID---------")
        config = parse_config()
        config.GDumb = True
        config.id2dataset = id2dataset
        eval_dict['bid'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'gdumb/3')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 3
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        #config.trainset = os.path.join(config.trainset, 'gdumb/3')
        config.train_txt = 'gdumb_train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on BID---------")


        print("-------Start training on CLIVE---------")
        config = parse_config()
        config.GDumb = True
        config.id2dataset = id2dataset
        eval_dict['clive'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'gdumb/4')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 4
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        #config.trainset = os.path.join(config.trainset, 'gdumb/4')
        config.train_txt = 'gdumb_train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        config.eval_clive = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on CLIVE---------")


        print("-------Start training on KONIQ10K---------")
        config = parse_config()
        config.GDumb = True
        config.id2dataset = id2dataset
        eval_dict['koniq10k'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'gdumb/5')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 5
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        #config.trainset = os.path.join(config.trainset, 'gdumb/5')
        config.train_txt = 'gdumb_train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        config.eval_clive = True
        config.eval_koniq10k = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on KONIQ01K---------")


        print("-------Start training on KADID10K---------")
        config = parse_config()
        config.GDumb = True
        config.id2dataset = id2dataset
        eval_dict['kadid10k'] = True
        config.eval_dict = eval_dict
        config.dataset = dataset
        split = i + 1
        config.split = split
        config.lwf = False
        config.shared_head = True
        config.task_id = 0
        config.n_task = 1
        config.replay = False
        base_ckpt_path = config.ckpt_path
        # # stage1: freezing previous layers, training fc
        config.ckpt_path = os.path.join(base_ckpt_path, str(config.split), 'gdumb/6')
        config.ckpt_resume_path = config.ckpt_path
        config.indicator = 6
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        #config.trainset = os.path.join(config.trainset, 'gdumb/6')
        config.train_txt = 'gdumb_train.txt'
        config.eval_live = True
        config.eval_csiq = True
        config.eval_bid = True
        config.eval_clive = True
        config.eval_koniq10k = True
        config.eval_kadid10k = True
        main(config)
        # stage2: fine-tuning the whole network
        config.fc = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = config.max_epochs2
        config.batch_size = config.batch_size2
        main(config)
        print("-------Finish training on KADID10K---------")


def eval_gdumb_head():
    config = parse_config()
    config.dataset = dataset
    config.task_id = task_id
    config.dataset2id = dataset2id
    config.train_kmeans = False
    config.n_task = 1
    config.shared_head = True
    config.train_expert = False
    config.train_kmeans = False
    config.replay = False
    config.GDumb = True

    for i in range(0, 1):
        config.base_ckpt_path = os.path.join(base_ckpt_path, str(config.split))
        srcc_matrix = np.zeros((seq_len, seq_len))
        srcc_matrix2 = np.zeros((seq_len, seq_len))

        s_srcc_matrix = np.ones((seq_len, seq_len))
        s_srcc_matrix2 = np.ones((seq_len, seq_len))

        predictions_test = {}
        predictions_val = {}

        config.id2dataset = id2dataset
        for idx, task in enumerate(dataset):
            print("-------Start testing on " + task + "---------")
            config.eval_each = False
            config.train = False
            config.resume = True
            config.resume_best = True
            config.indicator = idx + 1
            config.task_id = idx
            config.subfolder = 'gdumb'
            config.ckpt_path = os.path.join(base_ckpt_path, str(i+1), config.subfolder, str(config.indicator))
            config.ckpt_resume_path = config.ckpt_path
            eval_dict[task] = True
            config.eval_dict = eval_dict
            config.current_task_id = idx
            if idx > 0:
                config.verbose = False
            config.train_txt = 'gdumb_train.txt'
            srcc, srcc_val, all_hat_test, all_hat_val = main(config)
            for j in range(0, idx+1):
                srcc_matrix[idx, j] = srcc[id2dataset[j]]
                srcc_matrix2[idx, j] = srcc_val[id2dataset[j]]
                task_name = config.dataset[j]
                append_name = task_name + '_' + str(idx)
                predictions_test[append_name] = all_hat_test[task_name]
                predictions_val[append_name] = all_hat_val[task_name]

        initial_test = {}
        initial_val = {}
        for idx, task_name in enumerate(config.dataset):
            each_name = task_name + '_' + str(idx)
            initial_test[each_name] = predictions_test[each_name]
            initial_val[each_name] = predictions_val[each_name]

        for idx, task_name in enumerate(config.dataset):
            if idx == 0:
                continue #skip the first task
            else:
                subset = config.dataset[0:idx]
                for j, sub_name in enumerate(subset):
                    #index = idx + j
                    each_name = sub_name + '_' + str(idx)
                    hat_test = predictions_test[each_name]
                    hat_val = predictions_val[each_name]

                    initial_name_test = sub_name + '_' + str(j)
                    initial_name_val = sub_name + '_' + str(j)
                    initial_hat_test = predictions_test[initial_name_test]
                    initial_hat_val = predictions_val[initial_name_val]

                    s_srcc_matrix[idx, j] = scipy.stats.mstats.spearmanr(x=hat_test, y=initial_hat_test)[0]
                    s_srcc_matrix2[idx, j] = scipy.stats.mstats.spearmanr(x=hat_val, y=initial_hat_val)[0]

        stability_test, s_t_test, _, _ = unified_metric(s_srcc_matrix, mode=0)
        stability_val, s_t_val, _, _ = unified_metric(s_srcc_matrix2, mode=0)

        plasticity_test, p_t_test, avg_test, avg_test_t = unified_metric(srcc_matrix, mode=1)
        plasticity_val, p_t_val, avg_val, avg_val_t = unified_metric(srcc_matrix2, mode=1)

        print('stability test {}'.format(stability_test))
        print(s_t_test)
        print('stability val {}'.format(stability_val))
        print(s_t_val)

        print('plasticity test {}'.format(plasticity_test))
        print(p_t_test)
        print('plasticity val {}'.format(plasticity_val))
        print(p_t_val)

        print('final average test {}'.format(avg_test))
        print(avg_test_t)
        print('final average val {}'.format(avg_val))
        print(avg_val_t)