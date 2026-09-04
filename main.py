import model
from utils import train_and_predict, Seed_everything
import optuna
import argparse
import os

# 1. 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./Dataset/')
parser.add_argument("--feature_path", type=str, default='./Feature/')
parser.add_argument("--dataset", type=str, default='AAI')
parser.add_argument("--output_path", type=str, default='./output/')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--schedule", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--layer", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--vocab_size", type=int, default=251)
parser.add_argument("--v_dim", type=int, default=8)
parser.add_argument("--j_dim", type=int, default=8)
parser.add_argument("--vj_dropout", type=float, default=0.2)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--tune", action='store_true', default=False, help="Run Bayesian optimization for hyperparameter tuning")

args = parser.parse_args()

Seed_everything(seed=args.seed)
model_class = model.GraphTrans

# 2. 定义基础超参数配置
nn_config = {
    'node_input_dim_0': 1024 + 6 + 9 + 20, # ESM2 + 二面角 + dssp + pssm
    'node_input_dim_1': 768 + 6 + 9 + 20, # Ablang + 二面角 + dssp + pssm
    'edge_input_dim': 32 + 7,
    'hidden_dim': args.hidden_dim,
    'layer': args.layer,
    'augment_eps': 0.05,
    'dropout': args.dropout,
    'lr': args.lr,  # top lr
    'warmup': 5,
    'obj_max': 1,   # optimization object: max is better
    'epochs': args.epoch,
    'patience': 50,   # no early stop for demo
    'batch_size': args.batch_size,
    'num_samples_multiplier': 3,  # 1个epoch等于3个
    'folds': 5,
    'seed': args.seed,
    'schedule': args.schedule,
    'vocab_size': 155,
    'v_dim': 8,
    'j_dim': 8,
    'vj_dropout': 0.2,
    'num_heads': 8,
    'output_path': args.output_path
}


def run_training(config):
    """封装训练过程，接收超参数配置并返回 Test AUC"""
    run_id = f"lr_{config['lr']}_epoch_{config['epochs']}_batchsize_{config['batch_size']}_hiddendim_{config['hidden_dim']}_layer_{config['layer']}_dropout_{config['dropout']}"
    output_dir = os.path.join(config['output_path'], run_id)
    os.makedirs(output_dir, exist_ok=True)
    config['output_path'] = output_dir
    args.run_id = run_id
    train_and_predict(model_class, config, args)  # 训练和保存模型

    # 打开 test 日志文件，提取 Test AUC
    log_file_path = os.path.join(output_dir, 'train.log')
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if 'test_auc' in line:
                test_auc = float(line.split(':')[1].strip().split(',')[0])
                test_aupr = float(line.split(':')[2].strip().split(',')[0])
                break
    return test_auc


if args.tune:
    # 3. 定义贝叶斯优化目标函数
    def objective(trial):
        # 定义超参数搜索空间
        lr = trial.suggest_float('lr', 1e-5, 5e-4, step=0.00001)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
        layer = trial.suggest_int('layer', 4, 12, step=2)
        dropout = trial.suggest_float('dropout', 0.3, 0.5, step=0.05)

        # 创建超参数配置字典
        config = nn_config.copy()
        config.update({
            'lr': lr,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim,
            'layer': layer,
            'dropout': dropout
        })

        # 返回 Test AUC 作为优化目标
        return run_training(config)

    # 创建 Optuna study 并运行贝叶斯优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # 输出最佳超参数和对应的 Test AUC
    print("Best hyperparameters:", study.best_params)
    print("Best Test AUC:", study.best_value)

else:
    # 运行普通训练模式
    run_training(nn_config)

