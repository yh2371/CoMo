import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--in-dir', type=str, default='./dataset', help='output directory') 
    parser.add_argument('--out-dir', type=str, default='./dataset', help='output directory') 
    
    return parser.parse_args()