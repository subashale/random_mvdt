# import d.csv and make scatter plot
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

#np.arange(-10, 10, 0.1) min/max matplot

def viz_data(df):
    #df.y = df.label.astype(str)

    # Use the 'hue' argument to provide a factor variable
    sns.lmplot(x="x1", y="x2", data=df, fit_reg=False, hue='label', legend=False)

    # Move the legend to an empty part of the plot
    plt.legend(loc='lower right')

    plt.show()

def DDScatterDFSns(column1, column2, df):
    # assume last column is dependable variable
    sns.lmplot(x=[key for key in df.columns[[column1]]][0], y=[key for key in df.columns[[column2]]][0], 
               data=df, fit_reg=False, hue=[key for key in df.columns[[-1]]][0], legend=False)

    # Move the legend to an empty part of the plot
    plt.legend(loc='lower right')

    plt.show()

# plotting using numpy data
def viz_data_np(split_list):
    #df.y = df.label.astype(str)

    # making dataframe
    df = pd.DataFrame(split_list[0].tolist(), columns=['x1', 'x2'])
    df['label'] = split_list[1].tolist()

    # Use the 'hue' argument to provide a factor variable
    sns.lmplot(x="x1", y="x2", data=df, fit_reg=False, hue="label", legend=False)

    # Move the legend to an empty part of the plot
    plt.legend(loc='lower right')

    plt.show()
#
#
# def viz_data_with_line(X, Y, theta):
#     # this will show line separates data points
#     # Plotting Values and Regression Line
#
#     # Calculating line values x and y
#     y = np.arange(-10, 10, 0.1)
#     x = (theta_f[2] - theta_f[1] * yl) / theta_f[0]
#
#     # Ploting Line
#     plt.plot(x, y, color='#58b970', label='Regression Line')
#     # Ploting Scatter Points
#     plt.scatter(df.x1, df.x2, c='#ef5423', label='Scatter Plot')
#
#     plt.xlabel('Head Size in cm3')
#     plt.ylabel('Brain Weight in grams')
#     plt.legend()
#     plt.show()
#
# def line():
#     theta_f = list(theta.flat)
#
#     y = np.arange(-10, 10, 0.1)
#     x = (theta_f[2] - theta_f[1] * yl) / theta_f[0]
#
#     plt.figure(figsize=(12, 8))
#     sns.lmplot(x='x1', y="x2", data=df, fit_reg=False, hue='label', legend=False)
#     plt.plot(x, y, 'b-', label='Linear Regression: h(x) = %0.2f + %0.2fx' % (theta[0], theta[1]))
#     plt.legend(loc='lower right')
#
#
#

#Data Visualization Scatter plot with legend and add coloring per class:

def scatter_plot(split_list):
    df = pd.DataFrame(split_list[0].tolist(), columns=['x1', 'x2'])
    df['label'] = split_list[1].tolist()
    label = np.unique(df['label'])
    plt.title('x1 vs x2')
    plt.xlabel("x1")
    plt.ylabel("x2")
    for i in range(len(label)):
        bucket = df[df['label'] == i]
        bucket = bucket.iloc[:, [0, 1]].values
        plt.scatter(bucket[:, 0], bucket[:, 1], label=label[i])
    plt.legend()
    plt.show()


def scatter_plot_line(theta, split_list):
    theta_f = list(theta.flat)
    # Calculating line values x and y
    y = np.arange(-10, 10, 0.1)
    x = (theta_f[2] - theta_f[1] * y) / theta_f[0]

    # plt.figure(figsize=(10,10))

    # plt.plot(x, y, color='#58b970', label='LR: h(x) = %0.2f + %0.2fx'%(theta[0], theta[1]))
    plt.plot(x, y, color='#008000')

    df = pd.DataFrame(split_list[0].tolist(), columns=['x1', 'x2'])
    df['label'] = split_list[1].tolist()
    label = np.unique(df['label'])
    plt.title('x1 vs x2')
    plt.xlabel("x1")
    plt.ylabel("x2")
    for i in range(len(label)):
        bucket = df[df['label'] == i]
        bucket = bucket.iloc[:, [0, 1]].values
        plt.scatter(bucket[:, 0], bucket[:, 1], label=label[i])
        plt.autoscale()
    plt.autoscale()
    plt.legend()
    #plt.show()

# for combine split 
def viz_data_with_line_np(theta, split_list):
    # this will show decision boundary as line with scatter plot

    # Ploting Line, decision boundary
    theta_f = list(theta.flat)
    # Calculating line values x and y
    #y = np.arange(-10, 10, 0.1)
    #x = (-theta_f[2] - theta_f[1] * y) / theta_f[0]

    #ref #https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
    #https://stackoverflow.com/questions/42704698/logistic-regression-plotting-decision-boundary-from-theta
    
    x1_line = np.arange(int(round(split_list[0].min())), int(round(split_list[0].max())) , 0.1)
    x2_line = (-theta_f[2] - theta_f[0] * x1_line) / theta_f[1]
    # x2 = - (theta_f[2] + np.dot(theta_f[0], x)) / theta_f[1]
    plt.plot(x1_line, x2_line, label='Decision Boundary')


    # zooming plots
    x1 = split_list[0][:, 0]
    x2 = split_list[0][:, 1]

    X_min_max = [int(round(x1.min()-10)), int(round(x1.max()+10))]
    X2_min_max = [int(round(x2.min() - 10)), int(round(x2.max() + 10))]
    plt.xlim(X_min_max[0], X_min_max[1])
    plt.ylim(X2_min_max[0], X2_min_max[1])

    # scatter plot
    categories = split_list[1]
    colormap = np.array(['#277CB6', '#FF983E'])

    plt.scatter(x1, x2, c=colormap[categories])
    
    #plt.show()

# for individual splits
def viz_data_with_line(theta, split_list):
    # this will show line separates data points
    # Plotting scatter and a Line

    # Ploting Line

    # flatting theta
    theta_f = list(theta.flat)
    # Calculating line values x and y
    y = np.arange(-10, 10, 0.1)
    x = (theta_f[2] - theta_f[1] * y) / theta_f[0]

    plt.figure(figsize=(7,8))

    # plt.plot(x, y, color='#58b970', label='LR: h(x) = %0.2f + %0.2fx'%(theta[0], theta[1]))
    plt.plot(x, y, color='#008000')

    # scatter plot
    # for showing all line together
    categories = split_list[1]
    colormap = np.array(['#277CB6', '#FF983E'])

    x1 = split_list[0][:, 0]
    x2 = split_list[0][:, 1]

    plt.scatter(x1, x2, c=colormap[categories])

    plt.autoscale()


#def fill(theta1, theta2, split_list):  # flatting theta

# theta_1 = list(theta1.flat)
#     theta_2 = list(theta2.flat)
#     # Calculating line values x and y
#     x = np.arange(-10, 10, 0.1)
#     y1 = (theta_1[2] - theta_1[1] * x) / theta_1[0]
#     y2 = (theta_2[2] - theta_2[1] * x) / theta_2[0]


#     # plt.plot(x, y, color='#58b970', label='LR: h(x) = %0.2f + %0.2fx'%(theta[0], theta[1]))
#     plt.plot(x, y1, color='#008000')
#     plt.plot(x, y2, color='#008000')

#     x1 = split_list[:, 0]
#     x2 = split_list[:, 1]

#     plt.scatter(x1,x2, c='#ef5423', label='Scatter Plot')

#     plt.fill_between(x, y1, y2, color='grey', alpha='0.5')

# #fill(tree[2][:3], tree[1][:3], X)

##new plot with np
# plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
# plot_y = -1/theta_optimized[2]*(theta_optimized[0]
#           + np.dot(theta_optimized[1],plot_x))
# mask = y.flatten() == 1
# adm = plt.scatter(X[mask][:,1], X[mask][:,2])
# not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2])
# decision_boun = plt.plot(plot_x, plot_y)
# plt.xlabel('Exam 1 score')
# plt.ylabel('Exam 2 score')
# plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
# plt.show()