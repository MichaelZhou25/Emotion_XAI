import argparse

from data.seed_v import prepare_seedv_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare official SEED-V DE features for EAGLE-Net'
    )
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sessions', default='123')
    parser.add_argument('--time_steps', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    prepare_seedv_features(
        args.root,
        args.out,
        time_steps=args.time_steps,
        stride=args.stride,
        sessions=args.sessions,
    )
