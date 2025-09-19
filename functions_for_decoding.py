import numpy as np
from scipy import stats
from sklearn import linear_model,preprocessing
from sklearn.neighbors import KNeighborsClassifier
import random
# All functions to predict signal given velocity


class trial_direction_decoding(object):
    def __init__(self,Btrial_direction,Bspikes,is_shuffled,fraction_cells_to_use,fraction_time_train,n_stat,
                 C,trial_duration,bin_time_in_trial,idx_CW,idx_CCW):
        
        bin_time_prior_to_movement=40
        self.is_shuffled=is_shuffled
        self.frac=fraction_cells_to_use        
        self.fraction_time_train=fraction_time_train        
        self.n_stat=n_stat        
        self.C=C
        self.n_cells=np.shape(Bspikes)[0]
        n_trials=int(np.shape(Bspikes)[1]/trial_duration)
        if (idx_CW==1)&(idx_CCW==1):
            # CW/CCW
            indices=np.arange(n_trials)*trial_duration+bin_time_in_trial
            self.Bspikes=Bspikes[:,indices].T
            self.Btrial_direction=Btrial_direction
        if (idx_CW==0)&(idx_CCW==1):
            # 0/CCW
            indices=np.arange(int(n_trials/2))*2*trial_duration+bin_time_in_trial
            # Peak random bins prior to movement onset
            for idx_trial in np.random.choice(range(n_trials),int(n_trials/2),replace=False):
                indices=np.concatenate((indices,idx_trial*trial_duration
                                   +np.random.choice(range(bin_time_prior_to_movement),1,replace=False)));
            self.Bspikes=Bspikes[:,indices].T
            self.Btrial_direction=np.zeros(len(indices))
            self.Btrial_direction[0:int(n_trials/2)]=-1
        if (idx_CW==1)&(idx_CCW==0):
            # 0/CCW
            indices=np.arange(int(n_trials/2))*2*trial_duration+bin_time_in_trial+trial_duration
            # Peak random bins prior to movement onset
            for idx_trial in np.random.choice(range(n_trials),int(n_trials/2),replace=False):
                indices=np.concatenate((indices,idx_trial*trial_duration
                                   +np.random.choice(range(bin_time_prior_to_movement),1,replace=False)));
            self.Bspikes=Bspikes[:,indices].T
            self.Btrial_direction=np.zeros(len(indices))
            self.Btrial_direction[0:int(n_trials/2)]=+1
        self.indices=indices
        self.n_trials=len(self.Btrial_direction)

        
    def build_one_realization_of_test_and_train_set(self,):
        # Divides data into two a test and training part
        # Use leave two out model
        idx_cells=np.arange(0,self.n_cells,1);
        if self.frac<1:
            idx_cells=np.random.choice(range(self.n_cells),self.n_cells, replace=False);
            idx_cells=idx_cells[0:np.int32(self.frac*self.n_cells)]
            
        idx_all_trial=np.random.choice(range(self.n_trials),self.n_trials, replace=False);
        idx_train=idx_all_trial[0:np.int32(self.fraction_time_train*self.n_trials)]
        idx_test=idx_all_trial[np.int32(self.fraction_time_train*self.n_trials)::]
        
        TRy=np.zeros(np.shape(self.Btrial_direction[idx_train]))
        TRx=np.zeros(np.shape(self.Bspikes[idx_train,:][:,idx_cells]))
        TSy=np.zeros(np.shape(self.Btrial_direction[idx_test]))
        TSx=np.zeros(np.shape(self.Bspikes[idx_test,:][:,idx_cells]))

        TRy[:]=self.Btrial_direction[idx_train]
        TRx[:,:]=self.Bspikes[idx_train,:][:,idx_cells]
        TSy[:]=self.Btrial_direction[idx_test]
        TSx[:,:]=self.Bspikes[idx_test,:][:,idx_cells]
        if self.is_shuffled==1:
            idx_shuffle=np.random.choice(range(len(TRy)),len(TRy), replace=False);
            TRy=TRy[idx_shuffle]
                

        return TRx,TRy,TSx,TSy

    def train_decoder_on_bin(self,TRx,TRy):
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(TRx)

        regr = linear_model.LogisticRegression(solver='liblinear', C=self.C,class_weight='balanced', multi_class='ovr',random_state=0)

        regr.fit(x_train,TRy)
        def decoder(X):
            x_test = scaler.transform(X)
            return regr.predict(x_test)
        return decoder
    def evaluate_performance(self):
        n_test=len(np.arange(np.int32(self.fraction_time_train*self.n_trials),
                         self.n_trials,1))
                         
        Correct_predictions=np.zeros((self.n_stat,n_test));
        for idx_stat in range(self.n_stat):
            TRx,TRy,TSx,TSy=self.build_one_realization_of_test_and_train_set()
            decoder=self.train_decoder_on_bin(TRx,TRy)
            Predicted_y=decoder(TSx)
            Correct_predictions[idx_stat,Predicted_y==TSy]=1

            #print()
            #print(TSy)
            #print(Predicted_y)
            #print()

        return Correct_predictions      
  


class instanteneous_velocity_decoding(object):
    def __init__(self,Bvelocity,Bspikes,is_shuffled,fraction_cells_to_use,
                 fraction_time_train,n_stat,C,alpha,trial_duration,n_vel,Bvel_th,
                          idx_CW,idx_CCW):
        self.is_shuffled=is_shuffled
        self.frac=fraction_cells_to_use        
        self.fraction_time_train=fraction_time_train        
        self.n_stat=n_stat        
        self.C=C
        self.alpha=alpha
        self.n_cells=np.shape(Bspikes)[0]
        self.trial_duration=trial_duration
        self.Bvel_th=Bvel_th
        
        n_trials=int(np.shape(Bspikes)[1]/trial_duration)
        if (idx_CW==1)&(idx_CCW==1):
            # CW/CCW
            indices=np.arange(trial_duration)+0*trial_duration
            for idx_trial in np.arange(1,n_trials,1):
                indices=np.concatenate((indices,np.arange(trial_duration)+idx_trial*trial_duration));
            self.Bspikes=Bspikes[:,indices].T
            Bvelocity_to_use=Bvelocity[indices]
            
            
        if (idx_CW==0)&(idx_CCW==1):
            indices=np.arange(trial_duration)+0*trial_duration
            for idx_trial in np.arange(2,n_trials,2):
                indices=np.concatenate((indices,np.arange(trial_duration)+idx_trial*trial_duration));
            self.Bspikes=Bspikes[:,indices].T
            Bvelocity_to_use=Bvelocity[indices]
            
             
        if (idx_CW==1)&(idx_CCW==0):
            indices=np.arange(trial_duration)+1*trial_duration
            for idx_trial in np.arange(3,n_trials,2):
                indices=np.concatenate((indices,np.arange(trial_duration)+idx_trial*trial_duration));
            self.Bspikes=Bspikes[:,indices].T
            Bvelocity_to_use=Bvelocity[indices]
            
        self.indices=indices        
        
        digBvelocity=np.zeros(len(Bvelocity_to_use))
        directionBvelocity=np.zeros(len(Bvelocity_to_use))
        Edges_digBvelocity=np.linspace(np.min(Bvelocity_to_use)-1,np.max(Bvelocity_to_use)+1,n_vel+1)

        self.Edges_digBvelocity=Edges_digBvelocity
        Weight=np.zeros(len(digBvelocity))
        for idx in range(n_vel):
            mask=(Bvelocity_to_use>=Edges_digBvelocity[idx])&(Bvelocity_to_use<Edges_digBvelocity[idx+1])
            if len(Bvelocity_to_use[mask])>0:
                Weight[mask]=len(Bvelocity_to_use)/(len(Bvelocity_to_use[mask]))
                digBvelocity[mask]=np.mean(Bvelocity_to_use[mask])

        self.Bvelocity=digBvelocity
        self.Weight=Weight
        self.n_trials=int(len(self.Bvelocity)/trial_duration)
        
        

        
        
        
    def build_one_realization_of_test_and_train_set(self,):
        # Divides data into two a test and training part
        # Use leave two out model
        idx_cells=np.arange(0,self.n_cells,1);
        if self.frac<1:
            idx_cells=np.random.choice(range(self.n_cells),self.n_cells, replace=False);
            idx_cells=idx_cells[0:np.int32(self.frac*self.n_cells)]
        #idx_all_trial=np.arange(self.n_trials*self.trial_duration)
           
        idx_all_trial=np.random.choice(range(self.n_trials*self.trial_duration),self.n_trials*self.trial_duration, replace=False);
        idx_train=idx_all_trial[0:np.int32(self.fraction_time_train*self.n_trials*self.trial_duration)]
        idx_test=idx_all_trial[np.int32(self.fraction_time_train*self.n_trials*self.trial_duration)::]
        
        TRy=np.zeros(np.shape(self.Bvelocity[idx_train]))
        TRx=np.zeros(np.shape(self.Bspikes[idx_train,:][:,idx_cells]))
        TR_weight=self.Weight[idx_train]

        TSy=np.zeros(np.shape(self.Bvelocity[idx_test]))
        TSx=np.zeros(np.shape(self.Bspikes[idx_test,:][:,idx_cells]))
        TS_weight=self.Weight[idx_test]

        TRy[:]=self.Bvelocity[idx_train]
        TRx[:,:]=self.Bspikes[idx_train,:][:,idx_cells]
        TSy[:]=self.Bvelocity[idx_test]
        TSx[:,:]=self.Bspikes[idx_test,:][:,idx_cells]
        
        if self.is_shuffled==1:
            idx_shuffle=np.random.choice(range(len(TRy)),len(TRy), replace=False);
            TRy=TRy[idx_shuffle]
            TR_weight=TR_weight[idx_shuffle]
        

        return TRx,TRy,TSx,TSy,TR_weight,TS_weight

    def train_decoder_on_bin(self,TRx,TRy):
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(TRx)

        regr = linear_model.LogisticRegression(solver='liblinear', C=self.C,class_weight='balanced', multi_class='ovr',random_state=0)

        regr.fit(x_train,TRy)
        def decoder(X):
            x_test = scaler.transform(X)
            return regr.predict(x_test)
        return decoder
    def train_Ridge_regression(self,TRx,TRy,TR_weight):
        scaler = preprocessing.StandardScaler()
        

        x_train = scaler.fit_transform(TRx)
        regr = linear_model.Ridge(self.alpha)
        regr.fit(x_train,TRy,sample_weight=TR_weight)
        def decoder(X):
            x_test = scaler.transform(X)
            Prediction=regr.predict(x_test)
            return Prediction
        return decoder
    
    def train_decoders_for_Nonlinear_prediction(self,TRx,TRy,TR_weight):
        TRy_direction=np.zeros(len(TRy))
        TRy_direction[TRy>self.Bvel_th]=1
        TRy_direction[TRy<-self.Bvel_th]=-1

        Direction_decoder=self.train_decoder_on_bin(TRx,TRy_direction)
        Linear_Positive_decoder,Linear_Negative_decoder=0,0
        mask=TRy>self.Bvel_th
        if np.size(TRx[mask])>0:
            Linear_Positive_decoder=self.train_Ridge_regression(TRx[mask,:],TRy[mask],TR_weight[mask])
        mask=TRy<-self.Bvel_th
        if np.size(TRx[mask])>0:
            Linear_Negative_decoder=self.train_Ridge_regression(TRx[mask,:],TRy[mask],TR_weight[mask])
        
        return Direction_decoder,Linear_Negative_decoder,Linear_Positive_decoder

  
    def Nonlinear_decoder(self,TSx,Direction_decoder,Linear_Negative_decoder,Linear_Positive_decoder):
        Prediction_Dir=Direction_decoder(TSx)
        Prediction=np.zeros(len(Prediction_Dir))
        mask=Prediction_Dir<0
        if len(Prediction_Dir[mask])>0:
            Prediction[mask]=Linear_Negative_decoder(TSx[mask,:])
        mask=Prediction_Dir>0
        if len(Prediction_Dir[mask])>0:
            Prediction[mask]=Linear_Positive_decoder(TSx[mask,:])
        return Prediction
    
    def Evaluate_error(self,Prediction,TSy):
        Vel=np.zeros(len(TSy))
        Error_v_Vel=np.zeros(len(TSy))
        for idx_vel in range(len(self.Edges_digBvelocity)-1):
            mask=(TSy>=self.Edges_digBvelocity[idx_vel])&(TSy<self.Edges_digBvelocity[idx_vel+1])
            if len(TSy[mask])>1:
                Vel[idx_vel]=np.mean(TSy[mask])
                Error_v_Vel[idx_vel]=np.mean((Prediction[mask]-TSy[mask])**2)

        return Vel,Error_v_Vel
    
    def evaluate_performance(self):
        for idx_stat in range(self.n_stat):
            print(idx_stat)
            TRx,TRy,TSx,TSy,TR_weight,TS_weight=self.build_one_realization_of_test_and_train_set()
            Direction_decoder,Linear_Negative_decoder,Linear_Positive_decoder=self.train_decoders_for_Nonlinear_prediction(TRx,TRy,TR_weight)

            Prediction=self.Nonlinear_decoder(TSx,Direction_decoder,Linear_Negative_decoder,Linear_Positive_decoder)

            if idx_stat==0:
                Vel=np.zeros((self.n_stat,len(TSy)))
                Error2=np.zeros((self.n_stat,len(TSy)))

            Vel[idx_stat,:],Error2[idx_stat,:]=self.Evaluate_error(Prediction,TSy)

        Error_v_Vel=np.zeros((len(self.Edges_digBvelocity)-1,4))
        for idx_vel in range(len(self.Edges_digBvelocity)-1):
            mask=(Vel.ravel()>self.Edges_digBvelocity[idx_vel])&(Vel.ravel()<=self.Edges_digBvelocity[idx_vel+1])
            Error_v_Vel[idx_vel,:]=np.mean(Vel.ravel()[mask]),np.sqrt(np.mean(Error2.ravel()[mask])),np.sqrt(stats.sem(Error2.ravel()[mask])),np.sqrt(np.std(Error2.ravel()[mask]))

        
        return Vel,Error2,Error_v_Vel   


    def evaluate_performance_once(self):
        TRx,TRy,TSx,TSy,TR_weight,TS_weight=self.build_one_realization_of_test_and_train_set()
        Direction_decoder,Linear_Negative_decoder,Linear_Positive_decoder=self.train_decoders_for_Nonlinear_prediction(TRx,TRy,TR_weight)

        Prediction=self.Nonlinear_decoder(TSx,Direction_decoder,Linear_Negative_decoder,Linear_Positive_decoder)


        Vel,Error2=self.Evaluate_error(Prediction,TSy)

        
        return Vel,Error2
        