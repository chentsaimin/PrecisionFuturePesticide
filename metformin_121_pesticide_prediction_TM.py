print('載入AI深度學習模型...')
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os

import scipy.io as sio
import pickle
# import matplotlib.pyplot as plt

from scipy import stats
from os import listdir
from tensorflow.python.client import device_lib
from keras.models import Sequential, load_model
from keras.layers import CuDNNGRU, Bidirectional, LeakyReLU, Dense, Dropout, Input, Convolution1D, Layer,Flatten, Reshape
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import regularizers, initializers, constraints
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from keras.utils import plot_model
# -*- coding: UTF-8 -*-
print('''
*********************精準未來農藥*********************

######                                                
#     # #####  ######  ####  #  ####  #  ####  #    # 
#     # #    # #      #    # # #      # #    # ##   # 
######  #    # #####  #      #  ####  # #    # # #  # 
#       #####  #      #      #      # # #    # #  # # 
#       #   #  #      #    # # #    # # #    # #   ## 
#       #    # ######  ####  #  ####  #  ####  #    # 
                                                      
    #######                                   
    #       #    # ##### #    # #####  ###### 
    #       #    #   #   #    # #    # #      
    #####   #    #   #   #    # #    # #####  
    #       #    #   #   #    # #####  #      
    #       #    #   #   #    # #   #  #      
    #        ####    #    ####  #    # ###### 
                                          
######                                               
#     # ######  ####  ##### #  ####  # #####  ###### 
#     # #      #        #   # #    # # #    # #      
######  #####   ####    #   # #      # #    # #####  
#       #           #   #   # #      # #    # #      
#       #      #    #   #   # #    # # #    # #      
#       ######  ####    #   #  ####  # #####  ###### 

*********************精準未來農藥*********************                                                     
''')
random_seed = 34
batch_size = 16
epochs = 100
from rdkit import Chem
from rdkit.Chem import AllChem
def smi_to_morganfingerprint(smi, radius, MORGAN_SIZE):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        tempReturn = np.zeros(MORGAN_SIZE, dtype=np.int8)
        vec = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=MORGAN_SIZE)
        for i in range(tempReturn.shape[0]):
            tempReturn[i] = vec[i]   
        return tempReturn
    else:
        return np.zeros(MORGAN_SIZE)
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)
class PharmacophoreException(Exception):
    pass

class PharmacophoreFileEndException(PharmacophoreException):
    pass

class PharmacophorePoint(object):
    def __init__(self, code, cx, cy, cz, alpha, norm, nx, ny, nz):
        self.code = code
        self.cx = float(cx)
        self.cy = float(cy)
        self.cz = float(cz)
        self.alpha = float(alpha)
        self.norm = int(norm)
        self.nx = float(nx)
        self.ny = float(ny)
        self.nz = float(nz)
    
    @classmethod
    def from_line(cls, line):
        return cls(*line.split())
    
    def to_line(self):
        return "{} {} {} {} {} {} {} {} {}".format(self.code, self.cx, self.cy, self.cz, self.alpha, self.norm,\
                                                self.nx, self.ny, self.nz)
    
    def __str__(self):
        return self.to_line()
        
    
    
class Pharmacophore(object):
    def __init__(self, name, points):
        self.name = name
        self.points = points
        
    @classmethod
    def from_stream(cls, stream):
        name = stream.readline().strip()
        points = []
        line = stream.readline().strip()
        if not line:
            raise PharmacophoreFileEndException("End of file")
            
        while line != "$$$$" or not line:
            points.append(PharmacophorePoint.from_line(line))
            line = stream.readline().strip()
            
        if not line:
            raise PharmacophoreException("Wrong format, no end line")
        return cls(name, points)
    
    @classmethod
    def from_file(cls, file_path):
        with open(file_path) as fd:
            return cls.from_stream(fd)
            
    def write_to_stream(self, stream):
        stream.write("{}\n".format(self.name))
        for point in self.points:
            stream.write("{}\n".format(point.to_line()))
        stream.write("$$$$\n".format(self.name))
            
    def write_to_file(self, file_path):
        with open(file_path, "w") as fd:
            self.write_to_stream(fd)
            
    def __str__(self):
        return  "{}\n{}\n$$$$".format(self.name,
                                      "\n".join(str(x) for x in self.points))
    
    def __len__(self):
        return len(self.points)
    
    def sample(self, name, n):
        points = sample(self.points, min(n, len(self)))
        return Pharmacophore(name, points)

class PharmDatabaseException(Exception):
    pass


def calc_pharmacophore(lig_path, ph_path):
    proc = Popen(
        "align-it --dbase {} --pharmacophore {}".format(lig_path, ph_path),
        shell=True,
        stdout=PIPE, stderr=PIPE)
    _ = proc.communicate()
    

class PharmDatabase(object):
    def __init__(self, path_to_ligands, path_to_ph_db, is_calculated=False):
        self.path_to_ligands = path_to_ligands
        self.path_to_ph_db = path_to_ph_db
        self.is_calculated = is_calculated
    
    def repair_database(self):
        pass
    
    def calc_database(self):
        if not self.path_to_ph_db:
            self.calc_pharmacophore(self.path_to_ligands, self.path_to_ph_db)

    
    def sample_database(self):
        pass
    
    def iter_database(self):
        if not self.is_calculated:
            raise PharmDatabaseException("Not calculated")
        with open(self.path_to_ph_db, 'r') as fd:
            while True:
                try:
                    pharmacophore = Pharmacophore.from_stream(fd)
                    yield pharmacophore
                except PharmacophoreFileEndException:
                    break
def get_fasta(fasta_name, training_data):
    training_data['sequence'] = None
    file = open(fasta_name)
    index = 0
    seq = ''
    for line in file: 
        if line.startswith(">"):
            if index >= 1:
                training_data['sequence'][training_data['target_id'] == name] = seq                
                print(index,name,seq[:10])
            seq = ''
            name = line[4:10]
            index = index + 1
            
        else:
            seq = seq + line[:-1]
    return training_data
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def root_mean_squared_error_loss(y_true, y_pred):
    X = 10**(-y_pred)
    Y = 10**(-y_true)
    return K.sqrt(K.mean(K.square(X - Y)))
def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs): 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform') 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer) 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint) 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint) 
            self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint) 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W) 
        if self.bias:
            uit += self.b 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u) 
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx()) 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
#model structure
model_name = 'ACTHON_model_2048_6'
auxiliary_input1 = Input(shape=(3098,), dtype='float32', name='main_input')
x = Dense(1524)(auxiliary_input1)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
r = Dropout(0.2)(x)
x = Dense(768)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
r = Dropout(0.2)(x)
x = Dense(384)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
r = Dropout(0.2)(x)
x = Dense(192)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
r = Dropout(0.2)(x)
x = Dense(96)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
r = Dropout(0.2)(x)
x = Dense(48)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
r = Dropout(0.2)(x)
x = Dense(24)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
r = Dropout(0.2)(x)
x = Dense(12)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
r = Dropout(0.2)(x)
x = Dense(6)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Dropout(0.2)(x)
main_output = Dense(3,activation='relu')(x)
model = Model(inputs=auxiliary_input1, outputs=main_output)


opt = keras.optimizers.Adam()
model.compile(loss=root_mean_squared_error,
              optimizer=opt,
              metrics=[pearson_r])
checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)
print('請輸入農藥待測物SMILES(例 \'OC(=O)COc1ccc(Cl)cc1Cl\')')
smiles= input()
feature=pd.read_csv('feature_zero.txt',sep='\t')

print('請輸入待測植物俗名(例 Cabbage)')
plant_name= input()
plant_name='Plant_'+str(plant_name)
feature[plant_name].iloc[0]=1

print('請輸入待測植物Scientific name(例 Brassica oleracea capitata)')
Scientific_name= input()
Scientific_name='Scientific name_'+str(Scientific_name)
feature[Scientific_name].iloc[0]=1

print('請輸入農藥待測物Study location(例 Taiwan)')
Study_location= input()
Study_location='Study location_'+str(Study_location)
feature[Study_location].iloc[0]=1

print('請輸入欲觀察植物的部位(例 Leaves)')
Matrix= input()
Matrix='Matrix_'+str(Matrix)
feature[Matrix].iloc[0]=1

print('請輸入欲觀察植物的部位表面(O)或是內部(I)')
IN_ON= input()
IN_ON='IN or ON matrix_'+str(IN_ON)
feature[IN_ON].iloc[0]=1

print('請輸入欲地點 野外(F)或是室內(U)')
Field_Undercover= input()
Field_Undercover='Field or Undercover_'+str(Field_Undercover)
feature[Field_Undercover].iloc[0]=1

print('人工智慧分析中請稍後...')

radius = 6 
MORGAN_SIZE = 2048
X_list =np.zeros((1,3098))

X_list[0,:2048] =smi_to_morganfingerprint(smiles, radius, MORGAN_SIZE)
X_list[0,2048:] =feature
# SMILES_MORGAN[SMILES_MORGAN == 0] = -1





model.load_weights(model_name)
predict_test = model.predict(X_list)[:,0]
print('分析完成...\n\n')
print('您的農藥殘留RL50為 ',predict_test[0], ' 天\n\n')

print('Metformin-121團隊 : 陳在民、黃之瀚、施宣誠、吳致勳、方偉泉，感謝您使用"精準未來農藥"。')
