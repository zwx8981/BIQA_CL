from ImageDataset import ImageDataset, ImageDataset_pandas
from torch.utils.data import DataLoader
import numpy as np
import argparse

eps = 1e-16

def set_dataset(config, csv_file, data_set, transfrom, num_workers, shuffle, test, verbose):

    if config.JL_CL:
        dataset_data = ImageDataset(
            csv_file=csv_file,
            img_dir=data_set,
            verbose=verbose,
            transform=transfrom,
            test=test,
            task_id=config.task_id)
    else:
        dataset_data = ImageDataset(
            csv_file=csv_file,
            img_dir=data_set,
            verbose=verbose,
            transform=transfrom,
            test=test)



    if test:
        bs=1
    else:
        bs=config.batch_size
    dataset_loader = DataLoader(dataset_data,
                                   batch_size=bs,
                                   shuffle=shuffle,
                                   pin_memory=True,
                                   num_workers=num_workers)

    return dataset_data, dataset_loader

def set_dataset2(batch_size, num_sample, pandas_object, data_set, transfrom, num_workers):
    dataset_data = ImageDataset_pandas(
        pandas_object=pandas_object,
        num_sample=num_sample,
        img_dir=data_set,
        transform=transfrom)

    bs=batch_size
    dataset_loader = DataLoader(dataset_data,
                                   batch_size=bs,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=num_workers)

    return dataset_data, dataset_loader

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def minmax_norm(x, min=0, max=1):
    assert max > min
    x = (x - min) / (max - min)
    return x


def unified_metric(srcc_matrix, mode=0):
    T = np.shape(srcc_matrix)[0]
    m_metric = 0
    metric_t = []
    avg_srcc_t = []

    for t in range(0, T):
        if mode == 0: #stability
            if t == 0:
                m = 1
            else:
                m = 0
                for s in range(0, t):
                    m += srcc_matrix[t, s]
                m = m / t
            metric_t.append(m)
            m_metric += m
        else:  #plasticity
            m = srcc_matrix[t, t]
            metric_t.append(m)
            m_metric += m

    #final_results = srcc_matrix[T-1, :]

    #final_results = 0
    for i in range(T):
        srccs = srcc_matrix[i, :i+1]
        msrcc = np.mean(srccs)
        avg_srcc_t.append(msrcc)

    final_results = np.array(avg_srcc_t)
    avg_srcc = np.mean(final_results)
    mean_metric = m_metric / T
    return mean_metric, metric_t, avg_srcc, avg_srcc_t


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_new", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)  # KD:19901116 #19890801 #19901014 #20160801 #20190525 #20180620

    parser.add_argument("--backbone", type=str, default='resnet18')
    parser.add_argument("--fc", type=bool, default=True)
    parser.add_argument('--scnn_root', type=str, default='saved_weights/scnn.pkl')

    parser.add_argument("--network", type=str, default="dbcnn")  # basecnn or dbcnn or dbcnn2 or metaiqa or ma19 or koncept
    parser.add_argument("--representation", type=str, default="gap") #bpv: bilinear pooling variant

    parser.add_argument("--ranking", type=bool, default=True)  # True for learning-to-rank False for regular regression
    parser.add_argument("--fidelity", type=bool,
                        default=True)  # True for fidelity loss False for regular ranknet with CE loss

    parser.add_argument("--resume_best", type=bool, default=False)
    parser.add_argument("--best_indicator", type=int, default=1) #1-6: LIVE, CSIQ, KADID10K, BID, CLIVE, KONIQ10K

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--trainset", type=str, default="./IQA_Database")

    parser.add_argument("--live_set", type=str, default="./IQA_Database/databaserelease2/")
    parser.add_argument("--csiq_set", type=str, default="./IQA_Database/CSIQ/")
    parser.add_argument("--bid_set", type=str, default="./IQA_Database/BID/")
    parser.add_argument("--clive_set", type=str, default="./IQA_Database/ChallengeDB_release/")
    parser.add_argument("--koniq10k_set", type=str, default="./IQA_Database/koniq-10k/")
    parser.add_argument("--kadid10k_set", type=str, default="./IQA_Database/kadid10k/")

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt_resume_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument("--train_txt", type=str,
                        default='train.txt')  # train.txt | train_synthetic.txt | train_authentic.txt | train_sub2.txt | train_score.txt

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size2", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_epochs2", type=int, default=9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decay_interval", type=int, default=3)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)

    parser.add_argument("--lwf", type=bool, default=False)
    parser.add_argument("--n_task", type=int, default=6)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--shared_head", type=bool, default=False)
    parser.add_argument("--get_scores", type=bool, default=False)

    parser.add_argument("--eval_each", type=bool, default=False)
    parser.add_argument("--weighted_output", type=bool, default=False)
    parser.add_argument("--subfolder", type=str, default='train_on_live')

    parser.add_argument("--amp", type=bool, default=True)

    # kmeans related args
    parser.add_argument("--current_task_id", type=int, default=0)
    parser.add_argument('--base_ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--kmeans", type=bool, default=False) # training with model
    parser.add_argument("--train_kmeans", type=int, default=0) # training after the model is trained

    parser.add_argument("--num_cluster", type=int, default=2)
    parser.add_argument("--reverse", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--reg_weight", type=float, default=10) #10 for lwf, 1000 for si, 10 for mas, 10000 for ewc
    parser.add_argument("--prev_weight", type=bool, default=False)
    parser.add_argument("--manual_lr", type=bool, default=False)

    parser.add_argument("--b_fidelity", type=bool, default=True)
    parser.add_argument("--train_expert", type=bool, default=True)
    parser.add_argument("--experts_eval", type=bool, default=False)  #expert gate
    parser.add_argument("--meanfeat_val", type=bool, default=False)  #mean feature baseline

    parser.add_argument("--replay", type=bool, default=False)
    parser.add_argument("--replay_memory", type=int, default=1000)
    parser.add_argument("--save_replay", type=bool, default=False)
    parser.add_argument("--sample_strategy", type=str, default='all') #random / all
    parser.add_argument("--icarl", type=bool, default=True)
    parser.add_argument("--new_replay", type=bool, default=True) # icarl-v2

    parser.add_argument("--JL", type=bool, default=False)
    parser.add_argument("--JL_CL", type=bool, default=False)
    parser.add_argument("--GDumb", type=bool, default=False)


    #other_methods
    parser.add_argument("--online_reg", type=bool, default=True)
    parser.add_argument("--reg_trigger", type=bool, default=False)
    #ewc
    parser.add_argument("--ewc", type=bool, default=False)
    #SI
    parser.add_argument("--SI", type=bool, default=False)
    #MAS
    parser.add_argument("--MAS", type=bool, default=False)


    return parser.parse_args()