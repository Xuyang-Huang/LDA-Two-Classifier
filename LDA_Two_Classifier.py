#-- coding: utf-8 --
#@Time : 2021/3/21 15:33
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : LDA_Two_Classifier.py
#@Software: PyCharm



import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sk_dataset


class LDATwoClassifier:
    """A LDA two classifier

    Only be used in two class dataset.
    """
    def __init__(self):
        self.w = None
        self.thr = None

    def train(self, data, label):
        """Training.

        Training and saving project matrix.

        :param data: A Numpy array float, [n, dim].
        :param label:  A Numpy array int, [n].
        :return: No return.
        """

        assert len(data.shape) > 1, f'data shape should be [n, dim].'
        assert len(data) == len(label), f'label number does not match data number.'

        dim = len(data[0])
        # Get sample means and scatters of each class.
        sample_mean = []
        scatters = np.zeros([dim, dim])
        for label_index in range(2):

            # Means
            current_label_data = data[np.where(label == label_index)]
            mean_tmp = (np.sum(current_label_data, axis=0) / len(current_label_data))[:, np.newaxis]
            sample_mean.append(mean_tmp)

            # Scatters
            for item in current_label_data:
                item = item[:, np.newaxis]
                scatters += np.dot((item - mean_tmp), (item - mean_tmp).T)
        self.w = np.dot(np.mat(scatters).I, sample_mean[0] - sample_mean[1])
        result = self.w.T * data.transpose([1, 0])
        result = np.array(result)[0]
        result_index = np.argsort(result)
        result = result[result_index]
        label = label[result_index]

        # Find the best threshold to classify.
        tp_fn = np.sum(label)
        np_tn = len(label) - tp_fn
        max_score = 0
        for i in range(len(result)):
            tp = np.sum(label[:i])
            tn = len(label[i:]) - np.sum(label[i:])
            sensitivity = tp/tp_fn
            specificity = tn/np_tn
            score = (sensitivity + specificity)/2
            if score > max_score:
                self.thr = result[i]
                max_score = score


    def predict(self, data):
        """

        :param data: A Numpy array float, [n, dim].
        :return: Prediction.
        """
        assert len(data.shape) > 1, f'data shape should be [n, dim].'

        result = self.w.T * data.transpose([1, 0])
        result = np.array(result)[0] <= self.thr
        return result

    def eval(self, data, label):
        """Evaluate val data and plot result.

        :param data: A Numpy array float, [n, dim].
        :param label: A Numpy array int, [n].
        :return: No return.
        """
        assert len(data.shape) > 1, f'data shape should be [n, dim].'
        assert len(data) == len(label), f'label number does not match data number.'

        data_0 = data[np.where(label == 0)]
        data_1 = data[np.where(label == 1)]

        result_0 = self.w.T * data_0.transpose([1, 0])
        result_0 = np.array(result_0)[0]
        result_1 = self.w.T * data_1.transpose([1, 0])
        result_1 = np.array(result_1)[0]

        plt.figure()
        plt.scatter(np.arange(len(result_0)), result_0, cmap='y')
        plt.scatter(np.arange(len(result_0), len(label)), result_1, cmap='g')
        plt.show()

        result = self.w.T * data.transpose([1, 0])
        result = np.array(result)[0] <= self.thr
        acc = np.sum(np.equal(result, label)) / len(label)
        return acc


def prepare_data(proportion):
    dataset = sk_dataset.load_breast_cancer()
    label = dataset['target']
    data = dataset['data']
    n_class = len(dataset['target_names'])

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


if __name__ == '__main__':
    train, val, num_class = prepare_data(0.9)

    lda = LDATwoClassifier()
    lda.train(train[0], train[1])
    accuracy = lda.eval(val[0], val[1])

    pred = lda.predict(val[0])
    print(pred)
