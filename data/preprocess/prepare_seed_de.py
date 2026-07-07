import argparse
from data.seed import prepare_seed_features

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--time_steps', type=int, default=10)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--input_type', choices=['de', 'lds_de'], default='lds_de')
    args = p.parse_args()
    prepare_seed_features(args.root, args.out, args.time_steps, args.stride, args.session, args.input_type)
