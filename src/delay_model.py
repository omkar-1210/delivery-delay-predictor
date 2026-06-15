from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from catboost import CatBoostClassifier


class DelayModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

    def train_logistic_regression(self):
        model = LogisticRegression(class_weight='balanced',max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.lr_model = model 
        y_pred = self.lr_model.predict(self.X_test)         
        return y_pred
    
    def evaluate_model(self, y_pred):
        accuracy=accuracy_score(self.y_test,y_pred)
        precision=precision_score(self.y_test,y_pred)
        recall=recall_score(self.y_test,y_pred)
        f1=f1_score(self.y_test,y_pred)
        cm=confusion_matrix(self.y_test, y_pred)
        return {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'confusion_matrix':cm}

    def train_random_forest(self):
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        model.fit(self.X_train, self.y_train)
        self.rf_model = model 
        y_pred = self.rf_model.predict(self.X_test)         
        return y_pred


    def train_xgboost(self):
        model = XGBClassifier(random_state=42,eval_metric='logloss', scale_pos_weight=20)
        model.fit(self.X_train, self.y_train)
        self.xgb_model = model 
        y_pred = self.xgb_model.predict(self.X_test)         
        return y_pred
    
    def train_catboost(self):
        model = CatBoostClassifier(random_state=42,auto_class_weights='Balanced',verbose=0,)
        model.fit(self.X_train, self.y_train)
        self.cbc_model = model 
        y_pred = self.cbc_model.predict(self.X_test)         
        return y_pred

