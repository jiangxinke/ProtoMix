import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run ProtoMix.")
    parser.add_argument('--seed', type=int, default=2222, help='Random seed.')
    parser.add_argument('--cuda_choice', nargs='?', default='cuda:0',
                        help='GPU choice.')
    parser.add_argument('--dataset', nargs='?', default='MIMIC3', choices=["MIMIC3","MIMIC4"],
                        help='Dataset.')
    parser.add_argument('--train_val_test_split', nargs='?', default='[0.8,0.1,0.1]',
                        help='Train/Val/Test Split.')
    parser.add_argument('--use_last_checkpoint', type=int, default=-1,
                        help='Use last checkpoint')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay.')
    parser.add_argument('--train_lr', type=float, default=2e-4,
                        help='Train Learning rate.')
    parser.add_argument('--monitor_criterion', default="max",
                        choices=['max','min'], nargs='?', help='Monitor_criterion.')
    parser.add_argument('--clip', type=int, default=5,
                        help='Clip Value for gradient.')
    parser.add_argument('--metrics', default=["pr_auc","roc_auc","f1"],
                         nargs='*', help='Metrics.')
    parser.add_argument('--num_prototypes', type=int, default=6,
                        help='Prototype Num.')
    parser.add_argument('--prototype_alpha', type=float, default=0.1,
                        help='Prototype  alpha.')
    parser.add_argument('--monitor', nargs='?', default="pr_auc",
                        help='Monitor.')
    parser.add_argument('--sample_ratio', type=float, default=0.6,
                        help='Sample ratio.')
    parser.add_argument('--train_dropout_rate', type=float, default=0.5,
                        help='Train Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')

    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dim.')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embed dim.')
    parser.add_argument('--encoder_layer', type=int, default=1,
                        help='Encoder layer.')
    
    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                        help='Mixup alpha.')

    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Re-weighting Coefficient.')

    parser.add_argument('--epochs_train', type=int, default=300,
                        help='Number of training epoch.')
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--max_epochs_before_stop', default=60, type=int,
                        help='stop training if dev does not increase for N epochs.')

    args = parser.parse_args()

    save_dir = 'log/{}/{}/seed{}_{}/'.format(
        args.dataset,
        str(datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")),
        args.seed,args.cuda_choice)
    args.save_dir = save_dir

    return args
