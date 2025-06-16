import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from archs.unet import UNet
from archs.ukan import UKan
from archs.edsn import EDSN
from train.datasets import MitoMTDataset
from utils import str2bool
import time


def seed_torch(seed=1029):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(config, device):
    """创建模型实例"""
    print("=> creating model %s" % config['model_name'])
    if config['model_name'] == 'UNet':
        model = UNet(n_channels=config['input_channels'], n_classes=config['num_classes'])
    elif config['model_name'] == 'UKan':
        model = UKan(n_channels=config['input_channels'], n_classes=config['num_classes'], device=device)
    elif config['model_name'] == 'EDSN':
        model = EDSN(n_channels=config['input_channels'], n_classes=config['num_classes'])
    else:
        raise NotImplementedError('Model not implemented')

    return model.to(device)


def load_checkpoint(model, model_name):
    """加载模型检查点"""
    checkpoint_path = f'/root/autodl-tmp/output_recons/models/{model_name}/model_final/last_model.pth'
    ckpt = torch.load(checkpoint_path)
    try:
        model.load_state_dict(ckpt)
    except:
        model.load_state_dict(ckpt, strict=False)
    return model


def test_inference_speed(model, dataloader, num_warmup=50, num_tests=3, save_dir=None):
    """
    测试模型在实际数据集上的推理速度
    Args:
        model: 待测试的模型
        dataloader: 数据加载器
        num_warmup: 预热迭代次数
        num_tests: 每张图像测试次数
        save_dir: 结果保存路径
    Returns:
        tuple: (统计信息字典, 详细结果DataFrame)
    """
    device = next(model.parameters()).device
    model.eval()

    # GPU预热
    print("Warming up GPU...")
    warmup_batch = next(iter(dataloader))[0].to(device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(warmup_batch)
    torch.cuda.synchronize()

    # 测试每张图像的推理速度
    print("Testing inference speed on actual images...")
    inference_stats = []

    with torch.no_grad():
        for i, (input, _, _, img_names) in enumerate(tqdm(dataloader)):
            input = input.to(device)

            # 每张图像测试多次取平均
            times = []
            for _ in range(num_tests):
                torch.cuda.synchronize()
                start_time = time.time()
                _ = model(input)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = np.mean(times) * 1000  # 转换为毫秒
            std_time = np.std(times) * 1000  # 时间标准差

            # 记录每张图像的信息
            base_name = os.path.splitext(os.path.basename(img_names[0]))[0]
            inference_stats.append({
                'image_name': base_name,
                'inference_time_ms': avg_time,
                'std_time_ms': std_time,
                'fps': 1000 / avg_time,
                'min_time_ms': np.min(times) * 1000,
                'max_time_ms': np.max(times) * 1000
            })

    # 转换为DataFrame进行统计分析
    df = pd.DataFrame(inference_stats)

    # 计算统计指标
    stats = {
        'avg_time_ms': df['inference_time_ms'].mean(),
        'std_time_ms': df['inference_time_ms'].std(),
        'min_time_ms': df['inference_time_ms'].min(),
        'max_time_ms': df['inference_time_ms'].max(),
        'avg_fps': df['fps'].mean(),
        'total_images': len(df),
        'gpu_name': torch.cuda.get_device_name(0)
    }

    # 打印结果
    print("\nInference Speed Statistics:")
    print(f"GPU: {stats['gpu_name']}")
    print(f"Total images tested: {stats['total_images']}")
    print(f"Average time per frame: {stats['avg_time_ms']:.2f} ms")
    print(f"Standard deviation: {stats['std_time_ms']:.2f} ms")
    print(f"Min time per frame: {stats['min_time_ms']:.2f} ms")
    print(f"Max time per frame: {stats['max_time_ms']:.2f} ms")
    print(f"Average FPS: {stats['avg_fps']:.2f}")

    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 保存每张图像的详细信息
        df.to_csv(os.path.join(save_dir, 'per_image_inference_times.csv'), index=False)

        # 保存总体统计信息
        pd.DataFrame([stats]).to_csv(os.path.join(save_dir, 'inference_speed_summary.csv'), index=False)

    return stats, df


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/output-final/models', help='output dir')
    parser.add_argument('--dataset', default='MitoMts', help='dataset name')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/without_normalization/same', help='dataset dir')
    parser.add_argument('--model_name', default='EDSN')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=1, type=int, help='input channels')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--num_warmup', default=50, type=int, help='number of warmup iterations')
    parser.add_argument('--num_tests', default=3, type=int, help='number of tests per image')

    return parser.parse_args()


def main():
    # 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch()
    args = vars(parse_args())

    # 设置模型名称
    if args['name'] is None:
        args['name'] = '%s_%s' % (args['dataset'], args['model_name'])

    # 加载配置
    config_path = f'/root/autodl-tmp/output_recons/models/{args["name"]}/config.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 打印配置信息
    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    # 创建模型
    model = create_model(config, device)

    # 加载数据
    test_datasets = MitoMTDataset(root_dir=args['data_dir'], phase='testing')
    test_data_loader = DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 加载模型权重
    model = load_checkpoint(model, config['name'])
    model.eval()

    # 设置保存路径
    save_dir = os.path.join(args['output_dir'], config['name'], 'inference_speed_test')

    # 执行推理速度测试
    stats, df = test_inference_speed(
        model,
        test_data_loader,
        num_warmup=args['num_warmup'],
        num_tests=args['num_tests'],
        save_dir=save_dir
    )


if __name__ == '__main__':
    main()
