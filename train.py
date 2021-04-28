import argparse
import csv
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to data.csv', default='data/data.csv')
    parser.add_argument('--lr', help='number of learning_rate', default=0.1, type=float)
    parser.add_argument('--show', help='show train steps', default=0, type=int)
    parser.add_argument('--plot', help='make a plot', default=0, type=int)
    parser.add_argument('--r2', help='show R2 metric', default=0, type=int)
    parser.add_argument('--max_iter', help='max iteration for train', default=1000, type=int)
    args = parser.parse_args()
    return args.__dict__


def read_data(path: str):
    if os.path.exists(path):
        x = []
        y = []
        with open(path, 'r') as f:
            for row in csv.DictReader(f):
                x.append(int(row['km']))
                y.append(int(row['price']))
        return x, y
    else:
        exit('False path to data.csv. Try: \'python3 train.py --path data/data.csv\'')


def mean_lst(lst):
    return float(sum(lst)) / len(lst)


def std_lst(lst):
    variance = sum([((x - mean_lst(lst)) ** 2) for x in lst]) / len(lst)
    stddev = variance ** 0.5
    return stddev


def norm_data(lst):
    new_lst = []
    for elem in lst:
        new_lst.append((elem - mean_lst(lst)) / std_lst(lst))
    return new_lst


def mse(y, y_pred):
    loss_lst = []
    for i in range(len(y)):
        loss_lst.append((y_pred[i] - y[i]) ** 2)
    return mean_lst(loss_lst)


def predict_lst(x, theta0, theta1):
    y_pred = []
    for i in range(len(x)):
        y_pred.append(theta0 + (theta1 * x[i]))
    return y_pred


def update_thetas(x, y, lr, theta0, theta1):
    a = []
    b = []
    for i in range(len(x)):
        a.append(theta0 + (theta1 * x[i]) - y[i])
        b.append((theta0 + (theta1 * x[i]) - y[i]) * x[i])
    new_theta0 = theta0 - lr * mean_lst(a)
    new_theta1 = theta1 - lr * mean_lst(b)
    return new_theta0, new_theta1


def save_thetas(x, theta0, theta1):
    with open('data/thetas.csv', 'w') as f:
        f.write('theta0,theta1,model_mean,model_std\n')
        f.write(str(theta0) + ',' +
                str(theta1) + ',' +
                str(mean_lst(x)) + ',' +
                str(std_lst(x)))


def model_train(x, y, lr, show, max_iter):
    theta0 = random.random()
    theta1 = random.random()
    x = norm_data(x)
    loss = 0
    new_loss = mse(y, predict_lst(x, theta0, theta1))
    iteration = 0
    while abs(loss - new_loss) > 0.0000001:
        iteration += 1
        theta0, theta1 = update_thetas(x, y, lr, theta0, theta1)
        loss = new_loss
        new_loss = mse(y, predict_lst(x, theta0, theta1))
        if show == 1:
            print('Iteration - {}, loss - {}'.format(iteration, new_loss))
        if iteration == max_iter:
            break
    if show == 1:
        print('Model trained with {} iteration, theta0 - {}, theta1 - {}'.format(iteration, theta0, theta1))
    return theta0, theta1


def r_squared(y, y_pred):
    ss_tot = []
    ss_res = []
    mean_y = mean_lst(y)
    for i in range(len(y)):
        ss_tot.append((y[i] - mean_y)**2)
        ss_res.append((y[i] - y_pred[i])**2)
    return 1 - (sum(ss_res)/sum(ss_tot))


def plot(x, y, y_pred):
    sns.set_style('white')
    sns.scatterplot(x=x, y=y, label='Data')
    plt.plot(x, y_pred, color='red', label='Linear Regression')
    plt.legend(loc='best')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.savefig('plot.png')


def main():
    args = parse_args()
    x, y = read_data(args['path'])
    if (args['lr'] > 1) | (args['lr'] < 0):
        args['lr'] = 0.1
        print('--lr must be bigger than 0 and not bigger than 1. It was set to 0.1')
    if args['max_iter'] <= 0:
        args['max_iter'] = 1000
        print('--max_iter must be bigger than 0. It was set to 1000')
    theta0, theta1 = model_train(x, y, args['lr'], args['show'], args['max_iter'])
    save_thetas(x, theta0, theta1)
    if args['plot'] == 1:
        plot(x, y, predict_lst(norm_data(x), theta0, theta1))
        print('Graph save to plot.png')
    if args['r2'] == 1:
        print('R2 metric -', r_squared(y, predict_lst(norm_data(x), theta0, theta1)))


if __name__ == '__main__':
    main()
