import torch
import numpy as np
import random
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
import argparse
import time


fix_seed = 1024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='PathFormer',
                        help='model name, options: [PathFormer]')
    parser.add_argument('--model_id', type=str, default="test")

    # data loader
    parser.add_argument('--data', type=str, default='ALFA_ad', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ALFA_dataset1/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--loss_save_name', type=str, default='test.csv', help='save path')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S]; M:multivariate predict multivariate, S:univariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='s',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # model
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=16)
    parser.add_argument('--num_nodes', type=int, default=18)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--residual_connection', type=int, default=1)
    parser.add_argument('--metric', type=str, default='mse')
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--temp', type=int, default=2)
    parser.add_argument('--lambda_contrastive', type=float, default=0.1)
    parser.add_argument('--no_inter_atten', type=int, default=0, help='0: select inter_atten, 1: drop inter_atten')
    parser.add_argument('--no_intra_atten', type=int, default=0, help='0: select intra_atten, 1: drop intra_atten')
    parser.add_argument('--no_contrastive', type=int, default=0, help='0: add contrastive loss, 1: drop contrastive loss')
    
    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=3, help='prior anomaly ratio (%)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    print('Args in experiment:')
    print(args)

    Exp = Exp_Anomaly_Detection

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_bs{}_sl{}_dm{}_df{}_ln{}_k{}_emb{}_ar{}_lr{}_pt{}_te{}_bn{}_{}'.format(
            args.model_id,
            args.batch_size,
            args.seq_len,
            args.d_model,
            args.d_ff,
            args.layer_nums,
            args.k,
            args.embed,
            args.anomaly_ratio,
            args.learning_rate,
            args.patience,
            args.train_epochs,
            args.batch_norm, ii)

            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            time_now = time.time()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            print('Inference time: ', time.time() - time_now)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_bs{}_sl{}_dm{}_df{}_ln{}_k{}_emb{}_ar{}_lr{}_pt{}_te{}_bn{}_{}'.format(
        args.model_id,
        args.batch_size,
        args.seq_len,
        args.d_model,
        args.d_ff,
        args.layer_nums,
        args.k,
        args.embed,
        args.anomaly_ratio,
        args.learning_rate,
        args.patience,
        args.train_epochs,
        args.batch_norm, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
