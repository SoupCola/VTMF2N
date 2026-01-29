import matplotlib.pyplot as plt

# Plot training/validation curves  
def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, save_path, colors,
                       loss_legend_loc='upper center', acc_legend_loc='upper left',
                       fig_size=(20, 10), sub_plot1=(1, 2, 1), sub_plot2=(1, 2, 2)):
    """
    Plot training and validation loss & accuracy curves  
    """
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()

    # --- Loss subplot ---  
    plt.subplot(*sub_plot1)
    t_loss = np.array(train_loss)
    v_loss = np.array(val_loss)
    min_train_loss = t_loss.min()
    min_val_loss = v_loss.min()

    plt.plot(range(len(train_loss)), train_loss, linestyle='-', color='tab:{}'.format(colors[0]),
             label="TRAIN LOSS ({:.4f})".format(min_train_loss))
    plt.plot(range(len(val_loss)), val_loss, linestyle='--', color='tab:{}'.format(colors[0]),
             label="VALID LOSS ({:.4f})".format(min_val_loss))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=loss_legend_loc)
    plt.title('Training and Validation Loss')

    # --- Accuracy subplot ---  
    plt.subplot(*sub_plot2)
    t_acc = np.array(train_acc)
    v_acc = np.array(val_acc)
    max_train_acc = t_acc.max()
    max_val_acc = v_acc.max()

    plt.plot(range(len(train_acc)), train_acc, linestyle='-', color='tab:{}'.format(colors[0]),
             label="TRAIN ACC ({:.4f})".format(max_train_acc))
    plt.plot(range(len(val_acc)), val_acc, linestyle='--', color='tab:{}'.format(colors[0]),
             label="VALID ACC ({:.4f})".format(max_val_acc))

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc=acc_legend_loc)
    plt.title('Training and Validation Accuracy')

    # Save figure  
    file_path = os.path.join(save_path, 'loss_acc_plot.png')
    fig.savefig(file_path)
