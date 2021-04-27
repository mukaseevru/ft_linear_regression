import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', help='number of km to predict price', default=-1, type=int)
    parser.add_argument('--path', help='path to thetas.csv', default='data/thetas.csv')
    args = parser.parse_args()
    return args.__dict__


def read_thetas(path: str):
    if os.path.exists(path):
        with open(path, 'r') as f:
            for row in csv.DictReader(f):
                theta0 = float(row['theta0'])
                theta1 = float(row['theta1'])
                model_mean = float(row['model_mean'])
                model_std = float(row['model_std'])
        return theta0, theta1, model_mean, model_std
    else:
        exit('False path to thetas.csv. Try: \'python3 predict.py --path data/thetas.csv\'')


def norm_data(x, model_mean, model_std):
    return (x - model_mean) / model_std


def model_predict(theta0, theta1, x, model_mean, model_std):
    return theta0 + (theta1 * norm_data(x, model_mean, model_std))


def main():
    args = parse_args()
    theta0, theta1, model_mean, model_std = read_thetas(args['path'])
    if (theta0 == 0) & (theta1 == 0):
        print('Model is not trained. Train it first: \'python3 train.py\'')
    if args['x'] < 0:
        print('Distance must be bigger than 0. For example: \'python3 predict.py --x 1\'')
    else:
        print(model_predict(theta0, theta1, args['x'], model_mean, model_std))


if __name__ == '__main__':
    main()
