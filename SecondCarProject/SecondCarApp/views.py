from django.shortcuts import render, redirect
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
# Create your views here.
from django.http import HttpResponse

fuel = { 'Diesel':1, 'Petrol':4, 'CNG':0, 'LPG':3, 'Electric':2 }
transmission = { 'Manual':1, 'Automatic':0 }
company_name = { 
    'Tata':25, 'Maruti':18, 'Chevrolet':3, 'Hyundai':10, 'Ford':8, 'Volkswagen':27,
       'Mahindra':17, 'Fiat':6, 'Nissan':21, 'Renault':23, 'Toyota':26, 'Datsun':5,
       'Honda':9, 'Skoda':24, 'Ambassador':0, 'OpelCorsa':22, 'Daewoo':4, 'Force':7,
       'Mercedes':19, 'BMW':2, 'Audi':1, 'Mitsubishi':20, 'Jeep':13, 'Isuzu':11,
       'Kia':14, 'Volvo':28, 'Jaguar':12, 'MG':16, 'Land':15
}
seller_type = {'Individual':1, 'Dealer':0, 'Trustmark Dealer':2}
owner = {'Second Owner':2, 'First Owner':0, 'Third Owner':4 ,'Fourth & Above Owner':1, 'Test Drive Car':3}
km_range = { 'high':0, 'medium':2, 'low':1}
year_range = { 'Scrap':3, 'Buy':1, 'Best':0, 'Junk':2}
ex_range = { 'Affordable':0,'family':3,'Luxury':1,'Premium':2}

objs = {
    0:{'year':2010,'km_driven':120000,'fuel':'Diesel','seller_type':'Individual','transmission':'Manual','owner':'Third Owner','Rating':9,'company_name':'Maruti','ExShowroom Price':378262,'selling_price':300000, 'name':'Maruti Swift Desire 2010','link':'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKqu1Pa5_BwAVuENd-2_DQzcEUGv_zba9aoA&usqp=CAU'},
    1:{'year':2014,'km_driven':107143,'fuel':'Diesel','seller_type':'Individual','transmission':'Manual','owner':'Second Owner','Rating':10.53,'company_name':'Maruti','ExShowroom Price':365953,'selling_price':200000, 'name':'Maruti Swift Desire 2014','link':'https://www.motoroids.com/wp-content/uploads/2018/01/2018-Maruti-Suzuki-Swift-India-Booking-4.jpg'},
    2:{'year':2013,'km_driven':70000,'fuel':'Diesel','seller_type':'Individual','transmission':'Manual','owner':'Second Owner','Rating':13,'company_name':'Toyota','ExShowroom Price':640781,'selling_price':450000, 'name':'Toyota Etios','link':'https://stat.overdrive.in/wp-content/uploads/2013/05/Toyota-Etios.jpg'},
    3:{'year':2016,'km_driven':64672,'fuel':'Diesel','seller_type':'Trustmark Dealer','transmission':'Manual','owner':'First Owner','Rating':14,'company_name':'Maruti','ExShowroom Price':1313572,'selling_price':770000, 'name':'Maruti Baleno','link':'https://th.bing.com/th/id/OIP.6Ui0d8VVEcwwbHLN2wv6xwHaE2?pid=Api&rs=1'},
    4:{'year':2017,'km_driven':120000,'fuel':'Diesel','seller_type':'Individual','transmission':'Manual','owner':'First Owner','Rating':9,'company_name':'Mahindra','ExShowroom Price':746101,'selling_price':628000, 'name':'Mahindra Bolero','link':'https://th.bing.com/th/id/OIP.NQYPHAdRxUwm8kANvLx_qQHaEl?pid=Api&rs=1'},
    5:{'year':2017,'km_driven':40000,'fuel':'Diesel','seller_type':'Individual','transmission':'Manual','owner':'First Owner','Rating':12,'company_name':'Hyundai','ExShowroom Price':1504858,'selling_price':1200000, 'name':'Hyundai Verna','link':'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRILew6KBPBHI1_ibndrOFq_5zXcIEPJ0uzpA&usqp=CAU'},
    6:{'year':2017,'km_driven':70000,'fuel':'Diesel','seller_type':'Individual','transmission':'Manual','owner':'First Owner','Rating':10.37,'company_name':'Toyota','ExShowroom Price':1457909,'selling_price':1300000, 'name':'Toyota Innova 2017','link':'https://cdni.autocarindia.com/ExtraImages/20190408115805_INNOVA1.png'},
    7:{'year':2018,'km_driven':50000,'fuel':'Diesel','seller_type':'Individual','transmission':'Manual','owner':'First Owner','Rating':9.96,'company_name':'Toyota','ExShowroom Price':1539213,'selling_price':1500000, 'name':'Toyota Innova 2018','link':'https://th.bing.com/th/id/OIP.Hwh2EJJvAtUaJoG5P9pFnwHaFj?pid=Api&rs=1'},
    8:{'year':2014,'km_driven':135000,'fuel':'Diesel','seller_type':'Individual','transmission':'Manual','owner':'Third Owner','Rating':10,'company_name':'Skoda','ExShowroom Price':1410297,'selling_price':1200000, 'name':'Skoda Octavia','link':'https://th.bing.com/th/id/R4d87d63503fbb659c357000f545d17c7?rik=YaJmcI%2bJ5brjzw&riu=http%3a%2f%2f1.bp.blogspot.com%2f-VTW6y00EHe0%2fUkv-P7ymr6I%2fAAAAAAAAa8I%2fTp7WlK5TzCw%2fs1600%2fSkoda-Octavia_RS_2014_20.jpg&ehk=rAl2yIfPMbxI9ojQ4WeSaxFsKozZ7nvpj3Bw%2fIXp76A%3d&risl=&pid=ImgRaw'},
    9:{'year':2016,'km_driven':126000,'fuel':'Diesel','seller_type':'Dealer','transmission':'Automatic','owner':'First Owner','Rating':10,'company_name':'Ford','ExShowroom Price':3575337,'selling_price':1800000, 'name':'Ford Endeavour','link':'https://di-uploads-pod2.dealerinspire.com/riverviewford2/uploads/2017/03/2017-Ford-Explorer-Ruby-Red.png'},
    10:{'year':2014,'km_driven':62237,'fuel':'Petrol','seller_type':'Dealer','transmission':'Automatic','owner':'First Owner','Rating':13,'company_name':'Audi','ExShowroom Price':3341783,'selling_price':1850000, 'name':'Audi Q5 2.0 2014','link':'https://st.motortrend.com/uploads/sites/10/2015/11/2014-audi-q5-suv-angular-front.png'},
    11:{'year':2012,'km_driven':35000,'fuel':'Diesel','seller_type':'Individual','transmission':'Automatic','owner':'First Owner','Rating':11.10,'company_name':'Mercedes','ExShowroom Price':4117559,'selling_price':2500000, 'name':'Mercedes-Benz E-Class','link':'https://s.aolcdn.com/dims-global/dims3/GLOB/legacy_thumbnail/788x525/quality/85/https://s.aolcdn.com/commerce/autodata/images/USC30MBCA61A021001.jpg'},
    12:{'year':2015,'km_driven':35000,'fuel':'Diesel','seller_type':'Individual','transmission':'Automatic','owner':'First Owner','Rating':11,'company_name':'Audi','ExShowroom Price':4119188,'selling_price':3500000, 'name':'Audi Q5 2.0 2015','link':'https://th.bing.com/th/id/OIP.6klvVKR_Q_nEj2wWBUZcbAHaE4?pid=Api&w=2100&h=1386&rs=1'},
    13:{'year':2020,'km_driven':1500,'fuel':'Diesel','seller_type':'Individual','transmission':'Automatic','owner':'First Owner','Rating':9,'company_name':'Audi','ExShowroom Price':9027725,'selling_price':4700000, 'name':'Audi A5 Sportback','link':'https://assets.newcars.com/images/car-pictures/original/2020-Audi-A5-Coupe-Hatchback-2.0T-Premium-2dr-All-wheel-Drive-quattro-Coupe-Photo-4.png'},
    14:{'year':2019,'km_driven':30000,'fuel':'Diesel','seller_type':'Dealer','transmission':'Automatic','owner':'First Owner','Rating':11,'company_name':'BMW','ExShowroom Price':9857238,'selling_price':4950000, 'name':'BMW XS XDrive','link':'https://assets.newcars.com/images/car-pictures/original/2019-BMW-X5-SUV-xDrive40i-4dr-All-wheel-Drive-Sports-Activity-Vehicle-Photo-14.png'},
    15:{'year':2016,'km_driven':77350,'fuel':'Diesel','seller_type':'Dealer','transmission':'Automatic','owner':'First Owner','Rating':11,'company_name':'Mercedes','ExShowroom Price':8585599,'selling_price':5500000, 'name':'Mercedes-Benz GLS','link':'https://th.bing.com/th/id/OIP.cv-Thmm8oEfEmh1OFKHNaQHaE6?pid=Api&rs=1'},
    16:{'year':2017,'km_driven':6500,'fuel':'Diesel','seller_type':'Dealer','transmission':'Automatic','owner':'First Owner','Rating':9,'company_name':'Mercedes','ExShowroom Price':15538153,'selling_price':8150000, 'name':'Mercedes-Benz S-Class','link':'https://www.allvehiclecontracts.co.uk/Images/183788/600.jpg'},
    17:{'year':2016,'km_driven':13000,'fuel':'Petrol','seller_type':'Dealer','transmission':'Automatic','owner':'First Owner','Rating':14.09,'company_name':'Audi','ExShowroom Price':14235729,'selling_price':8900000, 'name':'Audi RS7','link':'https://icdn3.digitaltrends.com/image/2016-audi-rs7-performance-product-1200x630-c-ar1.91.jpg?ver=1'}
}

'''images={
    0:{'link':'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKqu1Pa5_BwAVuENd-2_DQzcEUGv_zba9aoA&usqp=CAU'}

}'''
'''
encoded_objs = [['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','',''],
                ['','','','','','','','','','','']
               ]
'''
max=[2020.00,806599.00,4.00,2.00,1.0,4.00,15.00,28.00,2.00,3.00,3.00]
def home(request):
    return render(request, 'home.html')

accuracy=None
def available_cars(request):
    global accuracy
    return render(request, 'used-cars.html',{'objs':objs,'model_details_flag':model_details_flag,'accuracy':accuracy})


def year_convert(year_var):
    if year_var > 1991 and year_var <= 2005:
        return year_range['Junk']
    elif year_var > 2005 and year_var <= 2010:
        return year_range['Scrap']
    elif year_var > 2010 and year_var <= 2015:
        return year_range['Buy']
    elif year_var > 2015 and year_var <= 2020:
        return year_range['Best']


def km_convert(km_var):
    if km_var > 0 and km_var <= 35000:
        return km_range['low']
    elif km_var > 35000 and km_var <= 100000:
        return km_range['medium']
    elif km_var > 100000 and km_var <= 2000000:
        return km_range['high']
  
def ex_convert(ex_var):
    if ex_var > 0 and ex_var <= 500000:
        return ex_range['Affordable']
    elif ex_var > 500000 and ex_var <= 1000000:
        return ex_range['family']
    elif ex_var > 1000000 and ex_var <= 1500000:
        return ex_range['Luxury']
    elif ex_var > 1500000 and ex_var <= 20000000:
        return ex_range['Premium']


def show_more(request, id):
    encoded_objs = [[0,0,0,0,0,0,0,0,0,0,0]]
    temp_dict = objs[id]
    year_var = temp_dict['year']
    encoded_objs[0][9] = year_convert(int(year_var))/max[9]
   # print(encoded_objs[0][9])
    km_var = temp_dict['km_driven']
    encoded_objs[0][8] = km_convert(int(km_var))/max[8]
    #print(encoded_objs[0][8])

    ex_var = temp_dict['ExShowroom Price']
    encoded_objs[0][10] = ex_convert(int(ex_var))/max[10]
    #print(encoded_objs[0][10])

    encoded_objs[0][0]=round((temp_dict['year']/max[0]),6)
    encoded_objs[0][1]=round((temp_dict['km_driven']/max[1]),6)
    encoded_objs[0][2]=fuel[temp_dict['fuel']]/max[2]
    #print(encoded_objs[0][2])
    encoded_objs[0][3]=seller_type[temp_dict['seller_type']]/max[3]
    encoded_objs[0][4]=transmission[temp_dict['transmission']]/max[4]
    encoded_objs[0][5]=owner[temp_dict['owner']]/max[5]
    encoded_objs[0][6]=temp_dict['Rating']/max[6]
    encoded_objs[0][7]=company_name[temp_dict['company_name']]/max[7]

    print(encoded_objs)
    poly_reg=PolynomialFeatures(degree=2)
    #X_train_p=poly_reg.fit_transform(X_train)
    fin_list=poly_reg.fit_transform(encoded_objs)
    print(fin_list)

    filename='Most_Accurate_Reg_Model.sav'
    model_load=pickle.load(open(file=filename,mode='rb'))
    y_pred=model_load.predict([fin_list[0]])
    print(y_pred)
    temp_dict['predicted_value']=y_pred
    temp_dict['flag']=False
    temp_dict['car_id']=id
    #print("This is y pred value"+str(y_pred))
    return render(request, 'car-details.html', temp_dict)

def predict_value(request, id):
    temp_dict = objs[id]
    temp_dict['flag']=True
    return render(request, 'car-details.html', temp_dict)

def gallery(request):
    return render(request,'gallery.html', objs)
    
def user_accuracy(request):
    data=pd.read_csv('SecondCar.csv')
    ##print(data.mode())
    for col in data.columns:
        if data[col].dtype == 'int32' or data[col].dtype == 'int64' or data[col].dtype == 'float32' or data[col].dtype == 'float64':
            #median = data[col].median()
            data[col].fillna(data[col].median(), inplace= True)
            
        else:
            #print('Hello') 
            #input()
            data = data.fillna(data.mode().iloc[0])
            #print(col,end=" ")
    print(data.median())
    print(data.isna().sum())
    print(data[data.isnull().any(axis=1)==True])
    data['company_name']=data['name'].str.split(' ').str[0]
    print(data.head())
    data['name'].str.split(' ').str[1]
    print(data.company_name.nunique())
    km_ranges=['low','medium','high']
    limits=[0,35000,100000,2000000]
    data['km_range']=pd.cut(data['km_driven'],bins=limits,labels=km_ranges)
    print(round(data.describe(),0))
    year_ranges=['Junk','Scrap','Buy','Best']
    limits=[1991,2005,2010,2015,2020]
    data['year_range']=pd.cut(data['year'],bins=limits,labels=year_ranges)
    ex_range=['Affordable','family','Luxury','Premium']
    limits=[0,500000,1000000,1500000,20000000]
    data['ex_range']=pd.cut(data['ExShowroom Price'],bins=limits,labels=ex_range)
    EN=LabelEncoder()
    data['fuel']=EN.fit_transform(data['fuel'])
    data['transmission']=EN.fit_transform(data['transmission'])
    data['name']=EN.fit_transform(data['name'])
    data['seller_type']=EN.fit_transform(data['seller_type'])
    data['owner']=EN.fit_transform(data['owner'])
    data['company_name']=EN.fit_transform(data['company_name'])
    data['km_range']=EN.fit_transform(data['km_range'])
    data['year_range']=EN.fit_transform(data['year_range'])
    data['ex_range']=EN.fit_transform(data['ex_range'])
    data=data.drop(['name','ExShowroom Price'],axis=1)
    print(data.head(2))
    X=data.drop('selling_price',axis=1)
    y=data.selling_price
    round(X.describe(),2)
    all_x=list(X.columns)
    X[all_x]=X[all_x]/(X[all_x].max())
    print(round(X.describe(),2))
    ##data['transmission'].unique()
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=1234)
    poly_reg=PolynomialFeatures(degree=2)
    X_train_p=poly_reg.fit_transform(X_train)
    X_test_p=poly_reg.fit_transform(X_test)
    ########## Define Model #################
    model=RandomForestRegressor()

    ########## Fit the train data ##############

    model.fit(X_train_p, y_train)


    ########### Predict The values ##############

    y_pred=model.predict(X_test_p)

    ########### Accuracy ####################
    global accuracy
    accuracy=r2_score(y_test,y_pred)*100
    
    print(round(accuracy,2))
    #accuracy=round(accuracy,2)
    return redirect('SecondCarApp:available-cars')

model_details_flag = False 
def model_details(request):
    global model_details_flag
    model_details_flag=True
    return redirect('SecondCarApp:available-cars')

def analysis(request):
    return render(request, 'analysis.html')

def bar_graph_1(request):
    '''
    df = pd.read_csv('SecondCar.csv')
    df['company_name']=df['name'].str.split(' ').str[0]
    
    table = df.groupby(by='company_name').agg('count')
    #x = df['company_name']
    #y = table.drop(['company_name'],axis=1)
    #print('@@@@@@@')
    #print(x)
    #print('!!!!!!')
    #print(y)
    plt.xlabel('Company Name', fontsize=10)
    plt.ylabel('Count', fontsize=12)
    temp_list = []
    key_list = list(company_name.keys())
    key_list.sort()
    print('sorted key list')
    print(key_list)
    plt.bar( key_list, table['year'])
    plt.show()
    '''

    return render(request, 'analysis.html',{'bar1_flag':True, 'bar2_flag':False,'pie1_flag':False})

def bar_graph_2(request):
    '''
    df=pd.read_csv('SecondCar.csv')
    table=df.groupby(by='year').agg('count')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Count', fontsize=12)
    x=df.year.unique()
    y=table['fuel']
    print('@@@@')
    print(x)
    print('!!!!')
    print(y)
    plt.bar(x,y)
    plt.show()
    '''
    return render(request, 'analysis.html',{'bar1_flag':False,'bar2_flag':True,'pie1_flag':False})

def pie_graph_1(request):
     return render(request, 'analysis.html',{'bar1_flag':False,'bar2_flag':False,'pie1_flag':True})

def back_to_model_btn(request):
    global model_details_flag
    model_details_flag=False
    return redirect('SecondCarApp:available-cars')

def back_to_accuracy_btn(request):
    global accuracy
    accuracy = None
    return redirect('SecondCarApp:available-cars')