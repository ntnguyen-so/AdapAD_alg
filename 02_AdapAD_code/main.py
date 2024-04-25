import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import math
import json
from collections import defaultdict
from functools import reduce
import pickle
import sys

from utils import *
from learning_models import *
from supporting_components import *
import config 

torch.manual_seed(0)
        
class AdapAD:
    def __init__(self, predictor_config, value_range_config, minimal_threshold):
        # operation range of the framework
        self.value_range = value_range_config
        self.operation_range = value_range_config
        
        self.sensor_range = NormalValueRangeDb()
        
        # configuration for predictor
        self.predictor_config = predictor_config
        
        # init learning components
        self.data_predictor = NormalDataPredictor(config.LSTM_size_layer, 
                                                       config.LSTM_size, 
                                                       self.predictor_config['lookback_len'],
                                                       self.predictor_config['prediction_len'])
        self.generator = LSTMPredictor(self.predictor_config['prediction_len'], 
                    self.predictor_config['lookback_len'], 
                    config.LSTM_size, 
                    config.LSTM_size_layer, 
                    self.predictor_config['lookback_len']) 
        
        # immediate databases
        self.predicted_vals = PredictedNormalDataDb()
        self.minimal_threshold = minimal_threshold
        print('Minimal threshold:', self.minimal_threshold)
        self.thresholds = AnomalousThresholdDb()
        self.thresholds.append(self.minimal_threshold)
        
        # detected anomalies
        self.anomalies = list()
        
        # for logging purpose
        self.f_name = 'results/' + config.data_source + '/' + 'progress_' + str(minimal_threshold) + '.csv'
        print(self.f_name)
        self.f_log = open(self.f_name, 'w')
        self.f_log.write('observed,predicted,low,high,anomalous,err,threshold\n')
        self.f_log.close()
        
    def set_training_data(self, data):
        data = [self.__normalize_data(x) for x in data]
        self.observed_vals = DataSubject(data)
        
    def __batch_learning(self, learner, data2learn):
        num_epochs = config.epoch_train
        learning_rate = config.lr_train

        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)
        loss_l = list()
        
        # slide data 
        x, y = sliding_windows(data2learn, self.predictor_config['lookback_len'], self.predictor_config['prediction_len'])
        x, y = x.astype(float), y.astype(float)
        y = np.reshape(y, (y.shape[0], y.shape[1]))
        
        # prepare data for training
        train_tensorX = torch.Tensor(np.array(x))
        train_tensorY = torch.Tensor(np.array(y))
        train_dataset = torch.utils.data.TensorDataset(train_tensorX, train_tensorY)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
        
        # Train the model
        learner.train()
        for epoch in range(num_epochs):
            for i in range(len(x)):            
                optimizer.zero_grad()
                
                _x, _y = x[i], y[i]
                outputs = learner(torch.Tensor(_x).reshape((1,-1)))
                loss = criterion(outputs, torch.Tensor(_y).reshape((1,-1)))
                loss.backward(retain_graph=True)
                optimizer.step()

        return learner, train_tensorX, train_tensorY
    
    def __learn_error_pattern(self, trainX, trainY):
        for i in range(len(trainX)):
            train_predicted_val = self.data_predictor.predict(torch.reshape(trainX[i], (1,-1)))
            self.predicted_vals.append(train_predicted_val)

        trainY = trainY.data.numpy()
        observed_vals_ = list(trainY[:,0])
        
        
        predictive_errors = NormalDataPredictionErrorCalculator.calc_error(np.array(observed_vals_), 
                                                                           np.array(self.predicted_vals.get_tail(len(observed_vals_))))
        predictive_errors = predictive_errors.tolist()
        self.predictive_errors = PredictionErrorDb(predictive_errors)
        
        # self.threshold_generator.train(config.epoch_train,
                                         # config.epoch_update,
                                         # predictive_errors)
        self.generator, _, _ = self.__batch_learning(self.generator, self.predictive_errors.get_tail(self.predictive_errors.get_length()))
        
        predicted_vals = self.predicted_vals.get_tail(len(observed_vals_))
        for i in range(len(observed_vals_)):
            self.f_log = open(self.f_name, 'a')
            text2write = '{},{},,,,,\n'.format(self.__reverse_normalized_data(observed_vals_[i]),
                                    self.__reverse_normalized_data(predicted_vals[i]))
            self.f_log.write(text2write)
            self.f_log.close()
        
    def train(self, data):
        trainX, trainY = self.data_predictor.train(config.epoch_train, 
                                                                 config.lr_train,
                                                                 self.observed_vals.get_training_data())
        print('Trained NormalDataPredictor')
        
        # self.__learn_error_pattern(train_tensorX, train_tensorY)
        for i in range(len(trainX)):
            train_predicted_val = self.data_predictor.predict(torch.reshape(trainX[i], (1,-1)))
            self.predicted_vals.append(train_predicted_val)

        trainY = trainY.data.numpy()
        observed_vals_ = list(trainY[:,0])
        
        predictive_errors = NormalDataPredictionErrorCalculator.calc_error(np.array(observed_vals_), 
                                                                           np.array(self.predicted_vals.get_tail(len(observed_vals_))))
        predictive_errors = predictive_errors.tolist()
        self.predictive_errors = PredictionErrorDb(predictive_errors)
        
        # self.threshold_generator.train(config.epoch_train,
                                         # config.epoch_update,
                                         # predictive_errors)
        self.generator, _, _ = self.__batch_learning(self.generator, self.predictive_errors.get_tail(self.predictive_errors.get_length()))
        
        predicted_vals = self.predicted_vals.get_tail(len(observed_vals_))
        for i in range(len(observed_vals_)):
            self.f_log = open(self.f_name, 'a')
            text2write = '{},{},,,,,\n'.format(self.__reverse_normalized_data(observed_vals_[i]),
                                    self.__reverse_normalized_data(predicted_vals[i]))
            self.f_log.write(text2write)
            self.f_log.close()
        print('Trained AnomalousThresholdGenerator')
        
    def __logging(self, is_anomalous_ret):
        self.f_log = open(self.f_name, 'a')
            
        text2write = '{},{},{},{},{},{},{}\n'.format(self.__reverse_normalized_data(self.observed_vals.get_tail()),
                                self.__reverse_normalized_data(self.predicted_vals.get_tail()),
                                self.__reverse_normalized_data(self.predicted_vals.get_tail()-self.thresholds.get_tail()),
                                self.__reverse_normalized_data(self.predicted_vals.get_tail()+self.thresholds.get_tail()),
                                is_anomalous_ret,
                                self.predictive_errors.get_tail(),
                                self.thresholds.get_tail())
        self.f_log.write(text2write)
        self.f_log.close()                       
        
    def __normalize_data(self, val):
        #return float(val)
        return (float(val) - self.sensor_range.lower()) / (self.sensor_range.upper() - self.sensor_range.lower())
        
    def __reverse_normalized_data(self, val):
        return val*(self.sensor_range.upper() - self.sensor_range.lower())+self.sensor_range.lower()
        
    def is_inside_range(self, val):
        observed_val = self.__reverse_normalized_data(val)
        if observed_val >= self.sensor_range.lower() and observed_val <= self.sensor_range.upper():
            return True
        else:
            return False
            
    def __prepare_data_for_prediction(self, supposed_anomalous_pos):
        cnt = 0
        _x_temp = self.observed_vals.get_tail(1+self.predictor_config['lookback_len'])[:-1]
        predicted_vals = self.predicted_vals.get_tail(self.predictor_config['lookback_len'])
        for predicted_i in range(self.predictor_config['lookback_len']):
            checked_pos = predicted_i + 1
            if not self.is_inside_range(_x_temp[-checked_pos]):
                _x_temp[-checked_pos] = predicted_vals[-checked_pos]
            
                            
        _x = torch.from_numpy(np.array(_x_temp).reshape(1, -1)).float()
        
        return _x
        
    def __is_default_normal(self):
        observed_vals = self.observed_vals.get_tail(self.predictor_config['train_size'])
        cnt = 0
        for observed_val in observed_vals:
            if not self.is_inside_range(observed_val):
                cnt += 1
                
        if cnt > self.predictor_config['train_size']//2:
            return True
        else:
            return False
            
    def __update_generator(self, past_observations, observed_val):
        num_epochs = config.update_G_epoch
        learning_rate = config.update_G_lr
        
        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        
        self.generator.train()
        loss_l = list()
        for epoch in range(num_epochs):
            predicted_val = self.generator(past_observations.float())
            optimizer.zero_grad()           
            loss = criterion(predicted_val, torch.from_numpy(np.array(observed_val).reshape(1, -1)).float())        
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # for early stopping
            if len(loss_l) > 1 and loss.item() > loss_l[-1]:
              break     
            loss_l.append(loss.item())
            
      
    def simplify_error(self,errors, N_sigma=0):
        return np.mean(errors) + N_sigma*np.std(errors)
        
    def is_anomalous(self, observed_val):
        is_anomalous_ret = False
        is_update = False
        
        # save observed vals to the object
        observed_val = float(observed_val)
        observed_val = self.__normalize_data(observed_val)
        self.observed_vals.append(observed_val)
        # print(len(self.observed_vals))
        supposed_anomalous_pos = self.observed_vals.get_length()
        
        # predict normal value
        past_observations = self.__prepare_data_for_prediction(supposed_anomalous_pos)
        predicted_val = self.data_predictor.predict(torch.reshape(past_observations.float(), (1, -1)))
        self.predicted_vals.append(predicted_val)
        
        # perform range check
        if not self.is_inside_range(observed_val):
            self.anomalies.append(supposed_anomalous_pos)
            is_anomalous_ret = True
        else:
            self.generator.eval()
            past_predictive_errors = self.predictive_errors.get_tail(self.predictor_config['lookback_len'])
            past_predictive_errors = torch.from_numpy(np.array(past_predictive_errors).reshape(1, -1)).float()
            # threshold = self.threshold_generator.generate(torch.reshape(past_predictive_errors, (1,-1)), self.minimal_threshold)
            with torch.no_grad():       
                threshold = self.generator(past_predictive_errors)
                threshold = threshold.data.numpy()
                threshold = max(threshold[0,0] , self.minimal_threshold)
            self.thresholds.append(threshold)
        
            prediction_error = prediction_error = NormalDataPredictionErrorCalculator.calc_error(predicted_val, observed_val)
            self.predictive_errors.append(prediction_error)
            
        
            if prediction_error > threshold:
                if not self.__is_default_normal():
                    is_anomalous_ret = True
                    self.anomalies.append(supposed_anomalous_pos)
            
            self.data_predictor.update(config.epoch_update, 
                                       config.lr_update, 
                                       torch.reshape(past_observations, (1,-1)), 
                                       observed_val)
                
            if is_anomalous_ret or threshold > self.minimal_threshold:
                # self.threshold_generator.update(config.update_G_epoch, 
                                                # config.update_G_lr,
                                                # torch.reshape(past_predictive_errors, (1,-1)), 
                                                # prediction_error)
                self.__update_generator(past_predictive_errors, prediction_error)
            
        self.__logging(is_anomalous_ret)
            
        return is_anomalous_ret
        
    def clean(self):
        self.predicted_vals.clean(self.predictor_config['lookback_len'])
        self.predictive_errors.clean(self.predictor_config['lookback_len'])
        self.thresholds.clean(self.predictor_config['lookback_len'])
              

if __name__ == "__main__":
    predictor_config, value_range_config, minimal_threshold = config.init_config()
    if not minimal_threshold:
        raise("It is mandatory to set a minimal threshold")
    
    data_source = pd.read_csv(config.data_source_path)        
    data_production = data_source['value'].tolist()
    len_data_subject = len(data_production)
    
    # AdapAD
    AdapAD_obj = AdapAD(predictor_config, value_range_config, minimal_threshold)
    print('GATHERING DATA FOR TRAINING...', predictor_config['train_size'])
    
    observed_data = list()
    
    
    for data_idx in range(len_data_subject):
        measured_value = data_production[data_idx]
        timestamp = str(data_idx)
        data_idx += 1
        
        observed_data.append(float(measured_value))
        observed_data_sz = len(observed_data)
        
        # perform warmup training or make a decision
        if observed_data_sz == predictor_config['train_size']:
            AdapAD_obj.set_training_data(observed_data)
            AdapAD_obj.train(measured_value)
            print('------------STARTING TO MAKE DECISION------------')
        elif observed_data_sz > predictor_config['train_size']:
            is_anomalous_ret = AdapAD_obj.is_anomalous(measured_value)
            AdapAD_obj.clean()
        else:
            print('{}/{} to warmup training'.format(len(observed_data), predictor_config['train_size']))
        
            
    print('Done! Check result at {}'.format(AdapAD_obj.f_name))