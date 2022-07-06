import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
import tkinter as tk
from tkinter import *

df = pd.read_csv('diabetes_data.csv')

# Changing Postive to 1 and Negative to 0
df['class'] = df['class'].map({'Negative': 0, 'Positive': 1})

# Separating Target feature
X = df.drop(['class'], axis=1)
y = df['class']

# Storing Features
#objectList = X.select_dtypes(include = "object").columns
#print(objectList)


#Label Encoding for object to numeric conversion
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in list(X.columns[1:]):
    X[feature] = le.fit_transform(X[feature])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Logistic Regression
logi = LogisticRegression(random_state = 42, max_iter = 1000)
logi.fit(X_train, y_train)

# Model Evaluation
yp_test = logi.predict(X_test)
acc = accuracy_score(y_test, yp_test)
f1 = f1_score(y_test, yp_test)

print("Accuracy: {}, F1-Score: {}".format(acc, f1))


"""**Building a Predictive System With GUI**"""

r = tk.Tk()
r.title('Diabetes Prediction')
r.geometry("1000x700")


def submit():

    gen=0
    polyuria=0
    polydipsia=0
    wight_loss=0
    weakness=0
    polyphagis=0
    thrush=0
    visual_blurring=0
    itching=0
    irritability=0
    delayed_healing=0
    partial_paresis=0
    muscle_stiffness=0
    alopecia=0
    obesity=0
    user_age=0

    user_age=int(t1.get())
        

    if t2.get()=="Male":
        gen=1
    
    if t3.get()=="Yes": 
        polyuria=1
    
    if t4.get()=="Yes":
        polydipsia=1
    
    if t5.get()=="Yes": 
        wight_loss=1
    
    if t6.get()=="Yes":
        weakness=1 
        
    if t7.get()=="Yes": 
        polyphagis=1
    
    if t8.get()=="Yes":
        thrush=1
        
    if t9.get()=="Yes":
        visual_blurring=1
    
    if t10.get()=="Yes": 
        itching=1
    
    if t11.get()=="Yes":
        irritability=1
    
    if t12.get()=="Yes": 
        delayed_healing=1
    
    if t13.get()=="Yes":
        partial_paresis=1 
        
    if t14.get()=="Yes": 
        muscle_stiffness=1
    
    if t15.get()=="Yes":
        alopecia=1
        
    if t16.get()=="Yes":
        obesity=1

        
    input_data = (user_age,gen,polyuria,polydipsia,wight_loss,weakness,polyphagis,thrush,visual_blurring,
                  itching,irritability,delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity)
    print(input_data)
    
    df1 = pd.DataFrame({
        "Age": [user_age],
        "Gender": [gen],
        "Polyuria": [polyuria],
        "Polydipsia": [polydipsia],
        "sudden weight loss": [wight_loss],
        "weakness":	[weakness],
        "Polyphagia": [polyphagis],
        "Genital thrush": [thrush],	
        "visual blurring":	[visual_blurring],
        "Itching": [itching],
        "Irritability":	[irritability],
        "delayed healing":	[delayed_healing],
        "partial paresis":	[partial_paresis],
        "muscle stiffness":	[muscle_stiffness],
        "Alopecia": [alopecia],
        "Obesity": [obesity],
    })
    
    prediction = logi.predict(df1)
    print(prediction)

    if (prediction[0] == 0):
        lb2 = Label(r, text="The Person has no Diabetes", fg="#34deeb", font=('monospace',15,'bold'))
        lb2.place(x=400, y=600)
        print('The Person has no Diabetes')
    else:
        lb2 = Label(r, text="The Person has Diabetes", fg="#e01010", font=('monospace',15,'bold'))
        lb2.place(x=400, y=600)
        print('The Person has Diabetes')

l1 = Label(r, text="Age :").place(x=350, y=80) 
t1 = Entry(r, width=30)
t1.place(x=420, y=80) 
    
l2 = Label(r, text="Gender :").place(x=350, y=110)
t2 = Entry(r, width=30)
t2.place(x=420, y=110)

l3 = Label(r,text="Polyuria :").place(x=180, y=200)        
t3 = Entry(r, width=30)
t3.place(x=300, y=200)        

l4 = Label(r,text="Polydipsia :").place(x=180, y=230)         
t4 = Entry(r, width=30)
t4.place(x=300, y=230)        

l5 = Label(r, text="Sudden Weight Loss :").place(x=180, y=260)             
t5 = Entry(r, width=30)
t5.place(x=300, y=260)               

l6 = Label(r, text="Weakness :").place(x=180, y=290)        
t6 = Entry(r, width=30)
t6.place(x=300, y=290)        

l7 = Label(r,text="Polyphagia:").place(x=180, y=320)        
t7 = Entry(r, width=30)
t7.place(x=300, y=320) 

l8 = Label(r, text="Gental Thrush :").place(x=180, y=350) 
t8 = Entry(r, width=30)
t8.place(x=300, y=350) 
    
l9 = Label(r, text="Visual Blurring :").place(x=180, y=380)
t9 = Entry(r, width=30)
t9.place(x=300, y=380)

l10 = Label(r,text="Itching :").place(x=500, y=200)        
t10 = Entry(r, width=30)
t10.place(x=620, y=200)        

l11 = Label(r,text="Irritability :").place(x=500, y=230)         
t11 = Entry(r, width=30)
t11.place(x=620, y=230)        

l12 = Label(r, text="Delayed Healing :").place(x=500, y=260)             
t12 = Entry(r, width=30)
t12.place(x=620, y=260)               

l13 = Label(r, text="Partial Peresis :").place(x=500, y=290)        
t13 = Entry(r, width=30)
t13.place(x=620, y=290)        

l14 = Label(r,text="Muscle Stiffness :").place(x=500, y=320)        
t14 = Entry(r, width=30)
t14.place(x=620, y=320) 

l15 = Label(r, text="Alopecia :").place(x=500, y=350)        
t15 = Entry(r, width=30)
t15.place(x=620, y=350)        

l16 = Label(r,text="Obesity :").place(x=500, y=380)        
t16 = Entry(r, width=30)
t16.place(x=620, y=380) 
       

lbl = Label(r, text="Diabetes Prediction", fg="#56eb34", font=("Times New Roman Bold", 25))
lbl.place(x=350, y=20)


btns = Button(r, text="Submit", command=submit, bg='black',fg='white',font=('monospace',10,'bold')).place(x=500, y=500)

r.mainloop()

