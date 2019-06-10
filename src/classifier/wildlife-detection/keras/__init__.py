import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-1)
parser.add_argument("--l2", help="weight decay", type=float, default=2e-4)
parser.add_argument("--resnet_size", type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument(
        '--lr_steps',
        nargs='+',
        type=int)
args = parser.parse_args()
