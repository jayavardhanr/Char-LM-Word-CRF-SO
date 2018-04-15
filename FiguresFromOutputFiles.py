import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


precisionRegex= re.compile(r'(?<=precision:\s\s)(\d+\.\d+)')
recallRegex= re.compile(r'(?<=recall:\s\s)(\d+\.\d+)')
f1ScoreRegex= re.compile(r'(?<=f1\sscore:\s\s)(\d+\.\d+)')
totalLossRegex= re.compile(r'(?<=total\sloss:\s)(\d+\.\d+)')

def plotting_storing(loss_train_record, loss_val_record, loss_test_record,
                                        f1_train_record, f1_val_record, f1_test_record):
    num_points = len(loss_train_record)
    x_list = range(1, len(loss_train_record) + 1)
    colors_train = []
    colors_val = []
    colors_test = []
    interval_idx = 0
    for idx in x_list:
        interval_idx += 1
        if interval_idx - 1 == 10:
            colors_train.append('b')
            colors_val.append('m')
            colors_test.append('g')
            interval_idx = 0
        else:
            colors_train.append('b')
            colors_val.append('m')
            colors_test.append('g')

    plt.figure(figsize=(12, 5))
    #figure_name = Model_Parameters['fig_path'] + 'loss_' + Model_Parameters['fig_name'] + '.png'
    plt.scatter(x=x_list, y=loss_train_record, c=colors_train, marker='o', label='Training loss')
    plt.scatter(x=x_list, y=loss_val_record, c=colors_val, marker='s', label='Validaton loss')
    plt.scatter(x=x_list, y=loss_test_record, c=colors_test, marker='v', label='Testing loss')
    plt.legend(loc=1)
    plt.title('Loss covergence curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_test.png")
    plt.show()

    plt.figure(figsize=(12, 5))
    #figure_name = Model_Parameters['fig_path'] + 'f1_' + Model_Parameters['fig_name'] + '.png'
    plt.scatter(x=x_list, y=f1_train_record, c=colors_train, marker='o', label='Training F1')
    plt.scatter(x=x_list, y=f1_val_record, c=colors_val, marker='s', label='Validation F1')
    plt.scatter(x=x_list, y=f1_test_record, c=colors_test, marker='v', label='Testing F1')
    x_best = f1_val_record.index(max(f1_val_record)) + 1
    val_best = max(f1_val_record)
    train_best = f1_train_record[x_best - 1]
    test_best = f1_test_record[x_best - 1]
    plt.annotate(str(max(f1_val_record)), xy=(x_best, val_best), xytext=(x_best-(len(x_list) / 5)-1, val_best-5), arrowprops=dict(arrowstyle='->', color='m'))
    plt.legend(loc=2)
    plt.title('F1 score convergence curve')
    plt.xlabel('Iteration')
    plt.ylabel('F1')
    plt.savefig("f1_test.png")
    plt.show()

    print("Done plotting")
    return


def getF1andLoss(filepath="./char-w-lstm-word-crf-conll2003.o3128445"):
    train_losses, train_precision, train_recall, train_f1, test_precision, test_recall, test_f1, val_precision, val_recall, val_f1,test_losses,val_losses = [],[],[],[],[],[],[],[],[],[],[],[]

    with open(filepath, 'r') as myfile:
        data = myfile.read()

    text_splits=data.split("Iteration")
    for Iteration in text_splits[1:]:
        precisions=precisionRegex.findall(Iteration)
        recalls = recallRegex.findall(Iteration)
        f1Scores = f1ScoreRegex.findall(Iteration)
        losses = totalLossRegex.findall(Iteration)

        train_precision.append(float(precisions[0]))
        train_recall.append(float(recalls[0]))
        train_f1.append(float(f1Scores[0]))
        train_losses.append(float(losses[0]))

        val_precision.append(float(precisions[1]))
        val_recall.append(float(recalls[1]))
        val_f1.append(float(f1Scores[1]))
        val_losses.append(float(losses[1]))

        test_precision.append(float(precisions[2]))
        test_recall.append(float(recalls[2]))
        test_f1.append(float(f1Scores[2]))
        test_losses.append(float(losses[2]))

    return train_losses, train_precision, train_recall, train_f1, test_precision, test_recall, test_f1,test_losses, val_precision, val_recall, val_f1,val_losses


train_losses, train_precision, train_recall, train_f1, test_precision, test_recall, test_f1, test_losses, val_precision, val_recall, val_f1, val_losses=getF1andLoss()
plotting_storing(train_losses,val_losses,test_losses,train_f1,val_f1,test_f1)

