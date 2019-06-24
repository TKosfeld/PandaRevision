#!/usr/bin/env python
# coding: utf-8

# In[1]:


# consider replacing hardcoded path with input selection, Can Jupyter Notebook utilize argparse?
import os
PATH = os.path.join("HumanData")

# populates list with all .csv files in hardcoded folder, does not differentiate between m, m_2, and m_5 data.
filelist = []
for file in os.listdir(PATH):
    if file.endswith(".csv"):
        filelist.append(file)
# housekeeping performance checks
print(len(filelist))
print(filelist)

# creation and population of m, m_2, and m_5 filelists
mfilelist = []
m2filelist = []
m5filelist = []
for file in filelist:
    if "_2" not in file and "_5" not in file:
        mfilelist.append(file)
    if "_2" in file:
        m2filelist.append(file)
    if "_5" in file:
        m5filelist.append(file)
#housekeeping performance checks
print(len(mfilelist))
print(len(m2filelist))
print(len(m5filelist))
print(mfilelist)
print(m2filelist)
print(m5filelist)

# creation and population of corresponding namelists, through snipping of various filelists
mnamelist = []
m2namelist = []
m5namelist = []
for file in mfilelist:
    mnamelist.append(file[:-4])
for file in m2filelist:
    m2namelist.append(file[:-4])
for file in m5filelist:
    m5namelist.append(file[:-4])
# housekeeping performance checks
#print(mnamelist)
#print(m2namelist)
#print(m5namelist)
print(len(mnamelist))
print(len(m2namelist))
print(len(m5namelist))


# In[2]:


import pandas as pd

# oft implemented read csv function
def load_data(filename, path=PATH):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


# In[3]:


# dictionary of matching data variables and csv assignments
mdata = {}

# populates dictionary for every m format data file in mfilelist
for x in range(len(mfilelist)):
    mdata["data{0}".format(x)] = load_data(mfilelist[x])
# housekeeping performance check
# print(mdata["data0"])


# In[4]:


for y in range(len(mdata)):
    mdata["data" + str(y)].columns = ["aminoAcid", mnamelist[y]]
    mdata["data" + str(y)] = mdata["data" + str(y)].groupby(["aminoAcid"], as_index = False).agg({mnamelist[y]: "sum"})


# In[5]:


# dictionary representation of all _2 data
m2data = {}
for x in range(len(m2filelist)):
    m2data["m2data{0}".format(x)] = load_data(m2filelist[x])
#print(m2data)


# In[6]:


for y in range(len(m2data)):
    m2data["m2data" + str(y)].columns = ["aminoAcid", m2namelist[y]]
    m2data["m2data" + str(y)] = m2data["m2data" + str(y)].groupby(["aminoAcid"], as_index = False).agg({m2namelist[y]: "sum"})
    


# In[7]:


# dictionary representation of all _5 data
m5data = {}
for x in range(len(m5filelist)):
    m5data["m5data{0}".format(x)] = load_data(m5filelist[x])
#print(m5data)


# In[8]:


for y in range(len(m5data)):
    m5data["m5data" + str(y)].columns = ["aminoAcid", m5namelist[y]]
    m5data["m5data" + str(y)] = m5data["m5data" + str(y)].groupby(["aminoAcid"], as_index = False).agg({m5namelist[y]: "sum"})


# In[9]:


# pd.merge requires a two variable merge in order to declare datam, is it possible to declare an empty datam?
datam = pd.merge(mdata["data0"], mdata["data1"], on = "aminoAcid", how = "outer")
for data in mdata:
    if data == "data0" or data == "data1":
       continue
    datam = pd.merge(datam, mdata[data], on = "aminoAcid", how = "outer")
    print(data)
for data in m2data:
    datam = pd.merge(datam, m2data[data], on = "aminoAcid", how = "outer")
    print(data)
#current memory error happens here
for data in m5data:
    datam = pd.merge(datam, m5data[data], on = "aminoAcid", how = "outer")
    print(data)
#original format involved many gzip compressions to same named file, why more than once?
datam.to_csv("datam.gz", compression ="gzip")


# In[ ]:


datam


# In[ ]:


#the precense of hardcoded chopped trainer files are not immediately visible in home directory
trainers = ["KJW100_HLA-A2_6_PRE.tsv.chopped","KJW100_HLA-A2_7_PRE.tsv.chopped","KJW100_HLA-A2_8_PRE.tsv.chopped","KJW100_HLA-A2_9_PRE.tsv.chopped","KJW102_HLA-A2_11_PRE.tsv.chopped","KJW102_HLA-A2_16_PRE.tsv.chopped","KJW102_HLA-A2_17_PRE.tsv.chopped","KJW102_HLA-A2_18_PRE.tsv.chopped","KJW102_HLA-A2_33_PRE.tsv.chopped","KJW102_HLA-A2_35_PRE.tsv.chopped","KJW102_HLA-A2_37_PRE.tsv.chopped","KJW102_HLA-A2_40_PRE.tsv.chopped","KJW103_HLA-A2_41_PRE.tsv.chopped","KJW103_HLA-A2_43_PRE.tsv.chopped","KJW103_HLA-A2_45_PRE.tsv.chopped","KJW103_HLA-A2_46_PRE.tsv.chopped","KJW103_HLA-A2_49_PRE.tsv.chopped","KJW103_HLA-A2_50_PRE.tsv.chopped","KJW103_HLA-A2_52_PRE.tsv.chopped","KJW103_HLA-A2_53_PRE.tsv.chopped","KJW103_HLA-A2_54_PRE.tsv.chopped","KJW103_HLA-A2_57_PRE.tsv.chopped","KJW103_HLA-A2_59_PRE.tsv.chopped","KJW100_HLA-A2_1_14_day.tsv.chopped","KJW100_HLA-A2_10_14_day.tsv.chopped","KJW100_HLA-A2_2_14_day.tsv.chopped","KJW100_HLA-A2_20_14_day.tsv.chopped","KJW100_HLA-A2_22_14_day.tsv.chopped","KJW100_HLA-A2_23_14_day.tsv.chopped","KJW100_HLA-A2_24_14_day.tsv.chopped","KJW100_HLA-A2_25_14_day.tsv.chopped","KJW100_HLA-A2_26_14_day.tsv.chopped","KJW100_HLA-A2_27_14_day.tsv.chopped","KJW100_HLA-A2_28_14_day.tsv.chopped","KJW100_HLA-A2_29_14_day.tsv.chopped","KJW100_HLA-A2_3_14_day.tsv.chopped","KJW100_HLA-A2_6_14_day.tsv.chopped","KJW100_HLA-A2_7_14_day.tsv.chopped","KJW100_HLA-A2_8_14_day.tsv.chopped","KJW100_HLA-A2_9_14_day.tsv.chopped","KJW103_HLA-A2_41_14_day.tsv.chopped","KJW103_HLA-A2_43_14_day.tsv.chopped","KJW103_HLA-A2_44_14_day.tsv.chopped","KJW103_HLA-A2_50_14_day.tsv.chopped","KJW103_HLA-A2_52_14_day.tsv.chopped","KJW103_HLA-A2_53_14_day.tsv.chopped","KJW103_HLA-A2_54_14_day.tsv.chopped","KJW100_1_8wk.tsv.chopped","KJW100_10_8wk.tsv.chopped","KJW100_20_8wk.tsv.chopped","KJW100_21_8wk.tsv.chopped","KJW100_23_8wk.tsv.chopped","KJW100_25_8wk.tsv.chopped","KJW100_26_8wk.tsv.chopped","KJW100_27_8wk.tsv.chopped","KJW100_28_8wk.tsv.chopped","KJW100_29_8wk.tsv.chopped","KJW100_3_8wk.tsv.chopped","KJW100_4_8wk.tsv.chopped","KJW100_6_8wk.tsv.chopped","KJW100_7_8wk.tsv.chopped","KJW100_8_8wk.tsv.chopped","KJW100_9_8wk.tsv.chopped","KJW103_HLA-A2_42_8wk.tsv.chopped","KJW103_HLA-A2_44_8wk.tsv.chopped","KJW103_HLA-A2_45_8wk.tsv.chopped","KJW103_HLA-A2_50_8wk.tsv.chopped","KJW103_HLA-A2_52_8wk.tsv.chopped","KJW103_HLA-A2_53_8wk.tsv.chopped","KJW103_HLA-A2_54_8wk.tsv.chopped"]

predictor = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
idx =0
better_data.insert(loc=idx, column='Predictor', value=predictor)


better_data.head(90)attributes = better_data.columns[1:]
attributes

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('std_scaler',StandardScaler())
])

X = better_data[attributes]
xfit = num_pipeline.fit_transform(X)

import numpy as np
from sklearn.model_selection import train_test_split

testerz = ["KJW102_HLA-A2_31_PRE.tsv.chopped","KJW103_HLA-A2_51_8wk.tsv.chopped","KJW102_HLA-A2_39_PRE.tsv.chopped", "KJW103_HLA-A2_44_PRE.tsv.chopped", "KJW103_HLA-A2_42_14_day.tsv.chopped","KJW103_HLA-A2_43_8wk.tsv.chopped","KJW100_HLA-A2_29_PRE.tsv.chopped","KJW103_HLA-A2_45_14_day.tsv.chopped", "KJW100_2_8wk.tsv.chopped","KJW100_HLA-A2_4_14_day.tsv.chopped","KJW100_22_8wk.tsv.chopped","KJW103_HLA-A2_51_PRE.tsv.chopped", "KJW100_HLA-A2_21_14_day.tsv.chopped","KJW103_HLA-A2_42_PRE.tsv.chopped", "KJW100_HLA-A2_10_PRE.tsv.chopped", "KJW102_HLA-A2_38_PRE.tsv.chopped", "KJW103_HLA-A2_41_8wk.tsv.chopped","KJW103_HLA-A2_58_PRE.tsv.chopped","KJW100_24_8wk.tsv.chopped", "KJW103_HLA-A2_51_14_day.tsv.chopped"]
test_data = better_data.T[testerz]
test_data = test_data.T
test_data

flip = better_data.T
train_data= flip[trainers]
train_data= train_data.T
train_data


# In[ ]:


X_train, y_train = train_data[attributes],train_data["Predictor"]
X_test, y_test = test_data[attributes], test_data["Predictor"]


# In[ ]:


X_train = num_pipeline.fit_transform(X_train)
X_test = num_pipeline.fit_transform(X_test)


# In[ ]:


import tensorflow as tf
import numpy as np


# In[ ]:


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()


# In[ ]:


y_test = y_test.astype(int)
y_train = y_train.astype(int)


# In[ ]:


estimator.get_params().keys()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"C": uniform(0, 100)}
rnd_search_cv = RandomizedSearchCV(SVC, param_distributions, n_iter=10, verbose=2)
rnd_search_cv.fit(X_train, y_train)
Y_PREDSV = rnd_search_cv.predict(X_test)
accuracy_score(y_test, Y_PREDSV)


# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

random_forest_clf = RandomForestClassifier(random_state=42, max_depth=4, max_features= 100, n_estimators = 10)
extra_trees_clf = ExtraTreesClassifier(random_state=42, max_features=100, max_depth= 6, n_estimators = 10)
SVM = LinearSVC(random_state=42, loss="hinge")
SVC = SVC(random_state=42, kernel = "poly", degree = 3, C=67)

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("SVM", SVM)
]


poly= Pipeline([
    ("polyfeat", PolynomialFeatures(degree=3)),
    ("svm_clf", LinearSVC(C=67, loss="hinge"))
])

extra_trees_clf.fit(X_train, y_train)
y_pred = extra_trees_clf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


estimators = [random_forest_clf, extra_trees_clf, SVM]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)


# In[ ]:


[estimator.score(X_test, y_test) for estimator in estimators]


# In[ ]:


voting_clf = VotingClassifier(named_estimators)


# In[ ]:


voting_clf.fit(X_train, y_train)


# In[ ]:


voting_clf.score(X_test, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


reset_graph()


# In[ ]:


he_init = tf.variance_scaling_initializer()


# In[ ]:


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, n_neurons=100, optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01, batch_size=20, activation=tf.nn.elu, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None):
        """Initialize the DNNClassifier by simply storing all the hyperparameters."""
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    def _dnn(self, inputs):
        """Build the hidden layers, with support for batch normalization and dropout."""
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(inputs, self.n_neurons,
                                     kernel_initializer=self.initializer,
                                     name="hidden%d" % (layer + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,
                                                       training=self._training)
            inputs = self.activation(inputs, name="hidden%d_out" % (layer + 1))
        return inputs

    def _build_graph(self, n_inputs, n_outputs):
        """Build the same model as earlier"""
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, 211698), name="X")
        y = tf.placeholder(tf.int64, shape=(None), name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None

        dnn_outputs = self._dnn(X)

        logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        """Fit the model to the training set. If X_valid and y_valid are provided, use early stopping."""
        self.close_session()

        # infer n_inputs and n_outputs from the training set.
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)
        
        # Translate the labels vector to a vector of sorted class indices, containing
        # integers from 0 to n_outputs - 1.
        # For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6], then the sorted class
        # labels (self.classes_) will be equal to [5, 6, 7, 8, 9], and the labels vector
        # will be translated to [3, 3, 4, 0, 2, 1, 1, 1]
        self.class_to_index_ = {label: index
                                for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label]
                      for label in y], dtype=np.int32)
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # extra ops for batch normalization
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # needed in case of early stopping
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None
        
        # Now train the model!
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run([self._loss, self._accuracy],
                                                 feed_dict={self._X: X_valid,
                                                            self._y: y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_val, best_loss, acc_val * 100))
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
                else:
                    loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                     feed_dict={self._X: X_batch,
                                                                self._y: y_batch})
                    print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_train, acc_train * 100))
            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)
            return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)


# In[ ]:


dnn_clf = DNNClassifier(random_state=42)
dnn_clf.fit(X_train, y_train, n_epochs=10)


# In[ ]:


from sklearn.metrics import accuracy_score

y_pred = dnn_clf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


reset_graph()
from functools import partial
import tensorflow

from sklearn.model_selection import RandomizedSearchCV
from tensorflow import tensorboard

def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)
    return parametrized_leaky_relu



param_distribs = {
    "n_neurons": [5, 10, 30, 50, 70, 90, 100],
    "batch_size": [5, 10, 15,25],
    "learning_rate": [0.01, 0.02],
    "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
    # you could also try exploring different numbers of hidden layers, different optimizers, etc.
    "n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
    "dropout_rate": [0,.1, .2],
    "batch_norm_momentum": [None, 0.9, 0.95, 0.98, 0.99, 0.999],
}

rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
                                random_state=42, verbose=2)
rnd_search.fit(X_train, y_train, n_epochs = 10)


# In[ ]:


rnd_search.best_params_


# In[ ]:


y_pred = rnd_search.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:



reset_graph()

def leaky_relu(alpha=0.01):
  def parametrized_leaky_relu(z, name=None):
      return tf.maximum(alpha * z, z, name=name)
  return parametrized_leaky_relu

dnn = DNNClassifier(random_state=42, n_neurons=5, n_hidden_layers=2, learning_rate=0.01, batch_size=10, activation=leaky_relu(alpha=0.01), dropout_rate=.2)
dnn.fit(X_train, y_train, n_epochs = 10)


# In[ ]:


from sklearn.metrics import accuracy_score

y_pred = dnn.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


reset_graph()

X = tf.placeholder(tf.float32, shape=(None, 211698), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.2
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, 5, activation=tf.nn.relu,
                              name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, 5, activation=tf.nn.relu,
                              name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, 2, name="outputs")


# In[ ]:





# In[ ]:


import os

PATH = os.path.join("datasets")


# In[ ]:


import pandas as pd

def load_data(filename, path=PATH):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


# In[ ]:


data16 = load_data("ML16final.csv")

data16.head()


# In[ ]:


from numpy import genfromtxt, savetxt
data16 = data16.T
data16.columns = data16.iloc[0]
data16 = data16.iloc[1:]


data16.info()
data16.head()


# In[ ]:


predictor16 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
idx =0


data16.insert(loc=idx, column='Predictor', value=predictor16)

from pandas import concat

frames = [test_data, data16[90:]]
testers = concat(frames)
testers["Predictor"]
testers = testers[attributes]
len(testers)


# In[ ]:


from numpy import concatenate

X_test16 = num_pipeline.fit_transform(testers)
X_test16 = X_test16[20:]



y_test16 = data16["Predictor"]
y_test16 = y_test16[90:]
len(X_test16)


# In[ ]:


y_test16


# In[ ]:


y_pred16 = dnn.predict(X_test16)
accuracy_score(y_test16, y_pred16)


# In[ ]:


y_pred16


# 

# In[ ]:


general page count: 5


# In[ ]:


scale with test data


# In[ ]:


copy work into googledrive


# In[ ]:


test with parameters


# In[ ]:





# In[ ]:




