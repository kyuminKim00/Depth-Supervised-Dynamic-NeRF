from dataLoader.load_llff import load_colmap_depth
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--match_type', type=str, 
					default='exhaustive_matcher', help='type of matcher used.  Valid options: \
					exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
parser.add_argument('scenedir', type=str,
                    help='input scene directory')
parser.add_argument('--downsample', type=int,
                    default= 4, help= 'downscaling factor')
args = parser.parse_args()

if __name__=='__main__':
    load_colmap_depth(args.scenedir, args.downsample, bd_factor=.75)