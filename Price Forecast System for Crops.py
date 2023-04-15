from pathlib import Path

#Tkinter
from tkinter import Tk, Toplevel, Canvas, Label, Entry, Text, Button, PhotoImage
from tkinter import StringVar

# Requests
import requests

# SQLite Connector
import sqlite3
conn = sqlite3.connect('Database\Price_Forecast_System.db')
cur = conn.cursor()

# pandas
import pandas as pd

# numpy
import numpy as np

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH0 = OUTPUT_PATH / Path(r"assets\frame0")
ASSETS_PATH2 = OUTPUT_PATH / Path(r"assets\frame2")
ASSETS_PATH1 = OUTPUT_PATH / Path(r"assets\frame1")
ASSETS_PATH5 = OUTPUT_PATH / Path(r"assets\frame5")
ASSETS_PATH6 = OUTPUT_PATH / Path(r"assets\frame6")
ASSETS_PATH4 = OUTPUT_PATH / Path(r"assets\frame4")
ASSETS_PATH7 = OUTPUT_PATH / Path(r"assets\frame7")


def relative_to_assets0(path: str) -> Path:
    return ASSETS_PATH0 / Path(path)

def relative_to_assets2(path: str) -> Path:
    return ASSETS_PATH2 / Path(path)

def relative_to_assets1(path: str) -> Path:
    return ASSETS_PATH1 / Path(path)

def relative_to_assets5(path: str) -> Path:
    return ASSETS_PATH5 / Path(path)

def relative_to_assets6(path: str) -> Path:
    return ASSETS_PATH6 / Path(path)

def relative_to_assets4(path: str) -> Path:
    return ASSETS_PATH4 / Path(path)

def relative_to_assets6(path: str) -> Path:
    return ASSETS_PATH7 / Path(path)


# API key for OpenWeatherMap
owm_api_key = "650f4e9ae8de0f2259ff7ff657967de5"

# OWM base URL
owm_base_url = "http://api.openweathermap.org/data/2.5/weather?"


def Predict_price_result_page_continue():
    win7.destroy()
    win6.destroy()
    win5.destroy()
    win4.destroy()
    Predict_crop_page()

def Predict_price_result_page(cost_msg):
    
    global win7
    
    win7 = Toplevel(win6)
    win6.withdraw()

    win7.geometry("1000x750")
    win7.configure(bg = "#FFFFFF")


    canvas = Canvas(
        win7,
        bg = "#FFFFFF",
        height = 750,
        width = 1000,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    image_image_1 = PhotoImage(
        file=relative_to_assets6("image_1.png"))
    image_1 = canvas.create_image(
        500.0,
        375.0,
        image=image_image_1
    )

    button_image_1 = PhotoImage(
        file=relative_to_assets6("button_1.png"))
    button_1 = Button(
        win7,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Predict_price_result_page_continue(),
        relief="flat"
    )
    button_1.place(
        x=414.0,
        y=500.0,
        width=201.0,
        height=77.0
    )

    '''
    entry_image_1 = PhotoImage(
        file=relative_to_assets6("entry_1.png"))
    entry_bg_1 = canvas.create_image(
        514.5,
        350.0,
        image=entry_image_1
    )
    entry_1 = Text(
        win7,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        highlightthickness=0
    )
    entry_1.place(
        x=165.0,
        y=160.0,
        width=699.0,
        height=378.0
    )
    '''
    
    canvas.create_text(
        275.0,
        280.0,
        anchor="nw",
        text=cost_msg,
        fill="#FFFFFF",
        font=("NATS", 26 * -1)
    )

    canvas.create_text(
        290.0,
        47.0,
        anchor="nw",
        text="PREDICTED CROP PRICE",
        fill="#FFFFFF",
        font=("NATS", 38 * -1)
    )

    button_image_2 = PhotoImage(
        file=relative_to_assets6("button_2.png"))
    button_2 = Button(
        win7,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Predict_price_result_page_back(),
        relief="flat"
    )
    button_2.place(
        x=26.0,
        y=25.0,
        width=51.0,
        height=22.0
    )
    win7.resizable(False, False)
    win7.mainloop()
    
def Predict_price_result_page_back():
    win6.deiconify()
    win7.destroy()


def Predict_price():
    
    global State, Market, Month, Year
    global crop

    State_pdf = State.get()
    Mkt = Market.get()
    MM = int(Month.get())
    YY = int(Year.get())

    # Reading the Dataset for the crop
    crop_price_df = pd.read_csv('Datasets/Price/' + crop + '.csv')
    crop_price_df.head()

    # Outlier Removal
    # IQR - interquartile range - the range of values that resides in the middle of the scores
    Q1 = np.percentile(crop_price_df['Price'], 25, interpolation='midpoint')
    Q3 = np.percentile(crop_price_df['Price'], 75, interpolation='midpoint')
    IQR = Q3 - Q1
    # upper bound
    upper = np.where(crop_price_df['Price'] >= (Q3 + 1.5 * IQR))
    # lower bound
    lower = np.where(crop_price_df['Price'] <= (Q1 - 1.5 * IQR))
    # Removing the outliers, i.e., the upper and lower bounds
    crop_price_df.drop(upper[0], inplace=True)
    crop_price_df.drop(lower[0], inplace=True)

    # convert necessary object data to int data
    # State - Price Data Frame (pdf)
    unq_state_pdf = list(set(crop_price_df['State']))
    dictOfWords_state_pdf = {unq_state_pdf[i]: i for i in range(len(unq_state_pdf))}
    crop_price_df['State'] = crop_price_df['State'].map(dictOfWords_state_pdf)
    # Market
    unq_market_pdf = list(set(crop_price_df['Market']))
    dictOfWords_market_pdf = {unq_market_pdf[i]: i for i in range(len(unq_market_pdf))}
    crop_price_df['Market'] = crop_price_df['Market'].map(dictOfWords_market_pdf)

    # Separate independent and dependent columns
    X = crop_price_df[['State', 'Market', 'Month', 'Year']]
    Y = crop_price_df[['Price']]

    # Split data into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Random Forest Regressor
    rfr = RandomForestRegressor()
    rfr.fit(X_train, Y_train.values.ravel())

    try:
        # Predicting price using user input
        State_pdf = dictOfWords_state_pdf[State_pdf]
        Mkt = dictOfWords_market_pdf[Mkt]
        user_input = [[State_pdf, Mkt, MM, YY]]
    
        predicted_price = rfr.predict(user_input)[0]
        predicted_price = round(predicted_price, 2)
        cost_message = "The predicted cost for " + crop + " in " + str(MM) + "/" + str(YY) + "\n\t is Rs." + str(predicted_price) + " per Quintal"
        Predict_price_result_page(cost_message)
    except KeyError:
        canvas_win6.create_text(
            425.0,
            625.0,
            anchor="nw",
            text="Invalid Market!",
            #This market does not sell the predicted crop!\nPlease try another market
            fill="#FFFFFF",
            font=("NATS", 26 * -1)
        )
    

def Predict_price_page():
    
    global win6, canvas_win6
    global Month, Year, Market
    
    win6 = Toplevel(win5)
    win5.withdraw()

    win6.geometry("1000x750")
    win6.configure(bg = "#FFFFFF")


    canvas_win6 = Canvas(
        win6,
        bg = "#FFFFFF",
        height = 750,
        width = 1000,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas_win6.place(x = 0, y = 0)
    image_image_1 = PhotoImage(
        file=relative_to_assets4("image_1.png"))
    image_1 = canvas_win6.create_image(
        500.0,
        375.0,
        image=image_image_1
    )

    button_image_1 = PhotoImage(
        file=relative_to_assets4("button_1.png"))
    button_1 = Button(
        win6,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Predict_price(),
        relief="flat"
    )
    button_1.place(
        x=382.0,
        y=514.0,
        width=259.99853515625,
        height=71.03274536132812
    )

    # Year - Entry
    Year = StringVar()
    entry_image_1 = PhotoImage(
        file=relative_to_assets4("entry_1.png"))
    entry_bg_1 = canvas_win6.create_image(
        512.0,
        418.5,
        image=entry_image_1
    )
    entry_1 = Entry(
        win6,
        textvariable=Year,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_1.place(
        x=272.0,
        y=395.0,
        width=480.0,
        height=45.0
    )

    canvas_win6.create_text(
        272.0,
        367.0,
        anchor="nw",
        text="Year *",
        fill="#000000",
        font=("NATS", 22 * -1)
    )

    # Month - Entry
    Month = StringVar()
    entry_image_2 = PhotoImage(
        file=relative_to_assets4("entry_2.png"))
    entry_bg_2 = canvas_win6.create_image(
        512.0,
        321.5,
        image=entry_image_2
    )
    entry_2 = Entry(
        win6,
        textvariable=Month,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_2.place(
        x=272.0,
        y=298.0,
        width=480.0,
        height=45.0
    )

    canvas_win6.create_text(
        268.0,
        274.0,
        anchor="nw",
        text="Month *",
        fill="#000000",
        font=("NATS", 22 * -1)
    )

    # Market - Entry
    Market = StringVar()
    entry_image_3 = PhotoImage(
        file=relative_to_assets4("entry_3.png"))
    entry_bg_3 = canvas_win6.create_image(
        512.0,
        224.0,
        image=entry_image_3
    )
    entry_3 = Entry(
        win6,
        textvariable=Market,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_3.place(
        x=272.0,
        y=200.0,
        width=480.0,
        height=46.0
    )

    canvas_win6.create_text(
        269.0,
        175.0,
        anchor="nw",
        text="Enter Market *",
        fill="#000000",
        font=("NATS", 22 * -1)
    )

    canvas_win6.create_text(
        310.0,
        55.0,
        anchor="nw",
        text="PREDICT CROP PRICE",
        fill="#FFFFFF",
        font=("NATS", 38 * -1)
    )

    button_image_2 = PhotoImage(
        file=relative_to_assets4("button_2.png"))
    button_2 = Button(
        win6,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Predict_price_page_back(),
        relief="flat"
    )
    button_2.place(
        x=26.0,
        y=25.0,
        width=51.0,
        height=22.0
    )
    win6.resizable(False, False)
    win6.mainloop()

def Predict_price_page_back():
    win5.deiconify()
    win6.destroy()


def Predict_crop_result_page_continue():
    Predict_price_page()

def Predict_crop_result_page(msg):
    
    global win5
    
    win5 = Toplevel(win4)
    win4.withdraw()

    win5.geometry("1000x750")
    win5.configure(bg = "#FFFFFF")


    canvas = Canvas(
        win5,
        bg = "#FFFFFF",
        height = 750,
        width = 1000,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    image_image_1 = PhotoImage(
        file=relative_to_assets6("image_1.png"))
    image_1 = canvas.create_image(
        500.0,
        375.0,
        image=image_image_1
    )

    button_image_1 = PhotoImage(
        file=relative_to_assets6("button_1.png"))
    button_1 = Button(
        win5,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Predict_crop_result_page_continue(),
        relief="flat"
    )
    button_1.place(
        x=404.0,
        y=608.0,
        width=243.0,
        height=78.0
    )
    
    canvas.create_text(
        250.0,
        300.0,
        anchor="nw",
        text=msg,
        fill="#FFFFFF",
        font=("NATS", 26 * -1)
    )

    canvas.create_text(
        373.0,
        61.0,
        anchor="nw",
        text="SUITABLE CROP",
        fill="#FFFFFF",
        font=("NATS", 38 * -1)
    )

    button_image_2 = PhotoImage(
        file=relative_to_assets6("button_2.png"))
    button_2 = Button(
        win5,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Predict_crop_result_page_back(),
        relief="flat"
    )
    button_2.place(
        x=26.0,
        y=25.0,
        width=51.0,
        height=22.0
    )
    win5.resizable(False, False)
    win5.mainloop()

def Predict_crop_result_page_back():
    win4.deiconify()
    win5.destroy()
    

def Predict_crop():
    
    global State, City, Nitrogen, Phosphorous, Potassium, pH
    global crop
    
    State_str = State.get()
    City_str = City.get()
    N = int(Nitrogen.get())
    P = int(Phosphorous.get())
    K = int(Potassium.get())
    #T = float(Temp.get())
    #H = float(Humidity.get())
    ph = float(pH.get())
    #R = float(Rainfall.get())
    
    # OWM URL
    owm_url = owm_base_url + "appid=" + owm_api_key + "&q=" + City_str
    
    response = requests.get(owm_url)
    x = response.json()
    
    y = x["main"]
    
    T = float(y["temp"]) - 273.15
    #H = float(y["humidity"])
    H = 20

    # Reading the dataset
    crop_df = pd.read_csv('Datasets/best_crop.csv')

    # convert object data to int data
    # State
    unq_state = list(set(crop_df['State']))
    dictOfWords_state = {unq_state[i]: i for i in range(len(unq_state))}
    crop_df['State'] = crop_df['State'].map(dictOfWords_state)
    # Crop
    unq_crop = list(set(crop_df['Crop']))
    dictOfWords_crop = {unq_crop[i]: i for i in range(len(unq_crop))}
    crop_df['Crop'] = crop_df['Crop'].map(dictOfWords_crop)

    # Separate independent and dependent columns
    x = crop_df.drop(['Crop', 'Rainfall'], axis=1)
    y = crop_df[['Crop']]

    # Split data into train and test data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2)

    # Feature Scaling
    scalar = StandardScaler()
    Xtrain = scalar.fit_transform(Xtrain)
    Xtest = scalar.transform(Xtest)

    # Using Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(Xtrain, Ytrain.values.ravel())

    State_code = dictOfWords_state[State_str]

    user_input = [[State_code, N, P, K, T, H, ph]]

    sc = StandardScaler()
    sc.fit(user_input)
    user_input = scalar.transform(user_input)

    op = rfc.predict(user_input)

    crop = [k for k, v in dictOfWords_crop.items() if v == op[0]][0]

    crop_message = "The best crop for the given conditions is " + crop
    Predict_crop_result_page(crop_message)

def Predict_crop_page():
    
    global win4
    global State, City, Nitrogen, Phosphorous, Potassium, pH
    
    win4 = Toplevel(win3)
    win3.withdraw()

    win4.geometry("1000x750")
    win4.configure(bg = "#FFFFFF")

    canvas = Canvas(
        win4,
        bg = "#FFFFFF",
        height = 750,
        width = 1000,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    image_image_1 = PhotoImage(
        file=relative_to_assets5("image_1.png"))
    image_1 = canvas.create_image(
        500.0,
        375.0,
        image=image_image_1
    )

    #Continue Button
    button_image_1 = PhotoImage(
        file=relative_to_assets5("button_1.png"))
    button_1 = Button(
        win4,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Predict_crop(),
        relief="flat"
    )
    button_1.place(
        x=433.0,
        y=567.0,
        width=133.634765625,
        height=64.08514404296875
    )

    # Potassium - Entry
    Potassium = StringVar()
    entry_image_1 = PhotoImage(
        file=relative_to_assets5("entry_1.png"))
    entry_bg_1 = canvas.create_image(
        646.5,
        461.0,
        image=entry_image_1
    )
    entry_1 = Entry(
        win4,
        textvariable=Potassium,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_1.place(
        x=539.0,
        y=441.0,
        width=215.0,
        height=38.0
    )

    canvas.create_text(
        534.0,
        416.0,
        anchor="nw",
        text="Potassium*",
        fill="#000000",
        font=("NATS", 20 * -1)
    )

    # Phosphorus - Entry
    Phosphorous = StringVar()
    entry_image_2 = PhotoImage(
        file=relative_to_assets5("entry_2.png"))
    entry_bg_2 = canvas.create_image(
        646.5,
        359.0,
        image=entry_image_2
    )
    entry_2 = Entry(
        win4,
        textvariable=Phosphorous,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_2.place(
        x=539.0,
        y=339.0,
        width=215.0,
        height=38.0
    )

    canvas.create_text(
        534.0,
        316.0,
        anchor="nw",
        text="Phosphorus*",
        fill="#000000",
        font=("NATS", 20 * -1)
    )

    # Nitrogen - Entry
    Nitrogen = StringVar()
    entry_image_3 = PhotoImage(
        file=relative_to_assets5("entry_3.png"))
    entry_bg_3 = canvas.create_image(
        646.5,
        256.0,
        image=entry_image_3
    )
    entry_3 = Entry(
        win4,
        textvariable=Nitrogen,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_3.place(
        x=539.0,
        y=236.0,
        width=215.0,
        height=38.0
    )

    canvas.create_text(
        534.0,
        210.0,
        anchor="nw",
        text="Nitrogen*",
        fill="#000000",
        font=("NATS", 20 * -1)
    )

    # pH - Entry
    pH = StringVar()
    entry_image_4 = PhotoImage(
        file=relative_to_assets5("entry_4.png"))
    entry_bg_4 = canvas.create_image(
        358.5,
        461.0,
        image=entry_image_4
    )
    entry_4 = Entry(
        win4,
        textvariable=pH,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_4.place(
        x=251.0,
        y=441.0,
        width=215.0,
        height=38.0
    )

    canvas.create_text(
        246.0,
        417.0,
        anchor="nw",
        text="pH*",
        fill="#000000",
        font=("NATS", 20 * -1)
    )

    # City - Entry
    City = StringVar()
    entry_image_5 = PhotoImage(
        file=relative_to_assets5("entry_5.png"))
    entry_bg_5 = canvas.create_image(
        358.5,
        359.0,
        image=entry_image_5
    )
    entry_5 = Entry(
        win4,
        textvariable=City,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_5.place(
        x=251.0,
        y=339.0,
        width=215.0,
        height=38.0
    )

    canvas.create_text(
        246.0,
        315.0,
        anchor="nw",
        text="City*",
        fill="#000000",
        font=("NATS", 20 * -1)
    )

    # State Entry
    State = StringVar()
    entry_image_6 = PhotoImage(
        file=relative_to_assets5("entry_6.png"))
    entry_bg_6 = canvas.create_image(
        358.5,
        256.0,
        image=entry_image_6
    )
    entry_6 = Entry(
        win4,
        textvariable=State,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_6.place(
        x=251.0,
        y=236.0,
        width=215.0,
        height=38.0
    )

    canvas.create_text(
        246.0,
        213.0,
        anchor="nw",
        text="State*",
        fill="#000000",
        font=("NATS", 20 * -1)
    )

    canvas.create_text(
        184.0,
        92.0,
        anchor="nw",
        text="PREDICT SUITABLE CROP FOR PRODUCTION",
        fill="#FFFFFF",
        font=("NATS", 26 * -1)
    )

    button_image_2 = PhotoImage(
        file=relative_to_assets5("button_2.png"))
    button_2 = Button(
        win4,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Predict_crop_page_back(),
        relief="flat"
    )
    button_2.place(
        x=26.0,
        y=25.0,
        width=51.0,
        height=22.0
    )
    win4.resizable(False, False)
    win4.mainloop()


def Predict_crop_page_back():
    win3.deiconify()
    win4.destroy()
    


def Login():
    
    Username_login = User_login.get()
    Password_login = Pwd_login.get()
    
    print(Username_login, Password_login)
    
    sql = "select * from User where Username='"+Username_login+"' and Password='"+Password_login+"'"
    print(sql)
    cur.execute(sql)
    user = cur.fetchall()
    
    for i in user:
        print(i)
    
    if len(user) == 0:
        canvas_win3.create_text(
            380.0,
            150.0,
            anchor="nw",
            text="*Incorrect Username or Password*",
            fill="#D01010",
            font=("NATS", 20 * -1)
        )
    else:
        Predict_crop_page()

def Login_page():
    
    global win3, canvas_win3
    global User_login, Pwd_login
    
    win3 = Toplevel(win1)
    win1.withdraw()

    win3.geometry("1000x750")
    win3.configure(bg = "#FFFFFF")

    canvas_win3 = Canvas(
        win3,
        bg = "#FFFFFF",
        height = 750,
        width = 1000,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )


    canvas_win3.place(x = 0, y = 0)
    image_image_1 = PhotoImage(
        file=relative_to_assets1("image_1.png"))
    image_1 = canvas_win3.create_image(
        500.0,
        375.0,
        image=image_image_1
    )

    # Password - Entry
    Pwd_login = StringVar()
    entry_image_1 = PhotoImage(
        file=relative_to_assets1("entry_1.png"))
    entry_bg_1 = canvas_win3.create_image(
        500.0,
        422.0,
        image=entry_image_1
    )
    entry_1 = Entry(
        win3,
        textvariable=Pwd_login,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        show = "*",
        highlightthickness=0
    )
    entry_1.place(
        x=274.0,
        y=394.0,
        width=452.0,
        height=54.0
    )

    canvas_win3.create_text(
        274.0,
        367.0,
        anchor="nw",
        text="PASSWORD*",
        fill="#000000",
        font=("RobotoRoman Bold", 20 * -1)
    )

    # Username - Entry
    User_login = StringVar()
    entry_image_2 = PhotoImage(
        file=relative_to_assets1("entry_2.png"))
    entry_bg_2 = canvas_win3.create_image(
        500.0,
        279.5,
        image=entry_image_2
    )
    entry_2 = Entry(
        win3,
        textvariable=User_login,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_2.place(
        x=274.0,
        y=251.0,
        width=452.0,
        height=55.0
    )

    canvas_win3.create_text(
        274.0,
        225.0,
        anchor="nw",
        text="USERNAME*",
        fill="#000000",
        font=("RobotoRoman Bold", 20 * -1)
    )

    # Login Function Button
    button_image_1 = PhotoImage(
        file=relative_to_assets1("button_1.png"))
    button_1 = Button(
        win3,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=Login,
        relief="flat"
    )
    button_1.place(
        x=408.0,
        y=538.0,
        width=184.28302001953125,
        height=60.67543029785156
    )

    '''
    # Forgot Password Button - Not Functional
    button_image_2 = PhotoImage(
        file=relative_to_assets1("button_2.png"))
    button_2 = Button(
        win3,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_2 clicked"),
        relief="flat"
    )
    button_2.place(
        x=562.0,
        y=383.0,
        width=164.0,
        height=19.0
    )
    '''

    # Back Button
    button_image_2 = PhotoImage(
        file=relative_to_assets1("button_2.png"))
    button_2 = Button(
        win3,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Login_page_back(),
        relief="flat"
    )
    button_2.place(
        x=26.0,
        y=25.0,
        width=51.0,
        height=22.0
    )
    
    canvas_win3.create_text(
        435.0,
        61.0,
        anchor="nw",
        text="Login",
        fill="#FFFFFF",
        font=("NATS", 64 * -1)
    )
    
    win3.resizable(False, False)
    win3.mainloop()

def Login_page_back():
    win1.deiconify()
    win3.destroy()


def Signup():
    
    global Username, Password, Email, Phone
    global incomplete_err
    
    User = Username.get()
    Pwd = Password.get()
    Em = Email.get()
    Ph = Phone.get()
    
    print(User)
    
    incomplete_err = canvas_win2.create_text(
        400.0,
        135.0,
        anchor="nw",
        text="",
        fill="#D01010",
        font=("NATS", 20 * -1)
    )
    
    if len(User)==0 or len(Pwd)==0 or len(Em)==0 or len(Ph)==0:
        incomplete_err = canvas_win2.create_text(
            400.0,
            135.0,
            anchor="nw",
            text="*Please enter all the details*",
            fill="#D01010",
            font=("NATS", 20 * -1)
        )
    else:
        sql = "SELECT Username FROM User WHERE Username='"+str(User)+"'"
        cur.execute(sql)
        user = cur.fetchall()
        
        if len(user) == 0:
            sql = "INSERT INTO User (Username,Password,Email,Phone) VALUES ('"+str(User)+"', '"+str(Pwd)+"', '"+str(Em)+"', '"+str(Ph)+"' )"
            print(sql)
            cur.execute(sql)
            conn.commit()

            #tk_mb.showinfo('Data Saved','Sign Up successful')

            win1.deiconify()
            win2.destroy()

        else:
            if (incomplete_err):
                canvas_win2.delete(incomplete_err)
            user_exists_err = canvas_win2.create_text(
                400.0,
                135.0,
                anchor="nw",
                text="*Username already exist*",
                fill="#D01010",
                font=("NATS", 20 * -1)
            )

def Signup_page():
    global win2
    global Username, Password, Phone, Email
    global canvas_win2

    win2 = Toplevel(win1)
    win1.withdraw()

    win2.geometry("1000x750")
    win2.configure(bg = "#FFFFFF")


    canvas_win2 = Canvas(
        win2,
        bg = "#FFFFFF",
        height = 750,
        width = 1000,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas_win2.place(x = 0, y = 0)
    image_image_1 = PhotoImage(
        file=relative_to_assets2("image_1.png"))
    image_1 = canvas_win2.create_image(
        501.0,
        379.0,
        image=image_image_1
    )
    
    canvas_win2.create_text(
        408.0,
        47.0,
        anchor="nw",
        text="Sign Up",
        fill="#FFFFFF",
        font=("NATS", 64 * -1)
    )

    # SignUp Command Button
    button_image_1 = PhotoImage(
        file=relative_to_assets2("button_1.png"))
    button_1 = Button(
        win2,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=Signup,
        relief="flat"
    )
    button_1.place(
        x=389.0,
        y=582.0,
        width=214.1239013671875,
        height=76.96707916259766
    )

    # Phone Number - Entry
    Phone = StringVar()
    entry_image_1 = PhotoImage(
        file=relative_to_assets2("entry_1.png"))
    entry_bg_1 = canvas_win2.create_image(
        502.7240447998047,
        497.8312568664551,
        image=entry_image_1
    )
    entry_1 = Entry(
        win2,
        textvariable=Phone,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_1.place(
        x=284.2759704589844,
        y=474.172119140625,
        width=436.8961486816406,
        height=45.318275451660156
    )

    canvas_win2.create_text(
        282.0,
        449.0,
        anchor="nw",
        text="PHONE NUMBER",
        fill="#000000",
        font=("RobotoRoman Medium", 20 * -1)
    )

    # Password - Entry
    Password = StringVar()
    entry_image_2 = PhotoImage(
        file=relative_to_assets2("entry_2.png"))
    entry_bg_2 = canvas_win2.create_image(
        499.44911193847656,
        314.8436279296875,
        image=entry_image_2
    )
    entry_2 = Entry(
        win2,
        textvariable=Password,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        show = "*",
        highlightthickness=0
    )
    entry_2.place(
        x=281.0000305175781,
        y=290.884521484375,
        width=436.8981628417969,
        height=45.918212890625
    )

    canvas_win2.create_text(
        278.0,
        266.0,
        anchor="nw",
        text="PASSWORD*",
        fill="#000000",
        font=("RobotoRoman Medium", 20 * -1)
    )

    # Email ID - Entry
    Email = StringVar()
    entry_image_3 = PhotoImage(
    file=relative_to_assets2("entry_3.png"))
    entry_bg_3 = canvas_win2.create_image(
        500.2770080566406,
        406.29718017578125,
        image=entry_image_3
    )
    entry_3 = Entry(
        win2,
        textvariable=Email,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_3.place(
        x=281.82794189453125,
        y=382.33807373046875,
        width=436.89813232421875,
        height=45.918212890625
    )

    canvas_win2.create_text(
        281.0,
        358.0,
        anchor="nw",
        text="EMAIL ID",
        fill="#000000",
        font=("RobotoRoman Medium", 20 * -1)
    )

    # Username - Entry
    Username = StringVar()
    entry_image_4 = PhotoImage(
    file=relative_to_assets2("entry_4.png"))
    entry_bg_4 = canvas_win2.create_image(
        497.5519104003906,
        222.8568572998047,
        image=entry_image_4
    )
    entry_4 = Entry(
        win2,
        textvariable=Username,
        bd=0,
        bg="#2F91B7",
        fg="#000716",
        font = ('Geoegia 16'),
        highlightthickness=0
    )
    entry_4.place(
        x=279.10382080078125,
        y=199.19772338867188,
        width=436.89617919921875,
        height=45.318267822265625
    )

    canvas_win2.create_text(
        278.0,
        177.0,
        anchor="nw",
        text="USERNAME*",
        fill="#000000",
        font=("RobotoRoman Medium", 20 * -1)
    )

    # Back - Button
    button_image_2 = PhotoImage(
        file=relative_to_assets2("button_2.png"))
    button_2 = Button(
        win2,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: Signup_page_back(),
        relief="flat"
    )
    button_2.place(
        x=26.0,
        y=25.0,
        width=51.0,
        height=22.0
    )
    win2.resizable(False, False)
    win2.mainloop()
    
def Signup_page_back():
    win1.deiconify()
    win2.destroy()


def Home_page():
    global win1
    
    win1 = Tk()

    win1.geometry("1000x750")
    win1.configure(bg = "#FFFFFF")


    canvas = Canvas(
        win1,
        bg = "#FFFFFF",
        height = 750,
        width = 1000,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    image_image_1 = PhotoImage(
        file=relative_to_assets0("image_1.png"))
    image_1 = canvas.create_image(
        500.0,
        375.0,
        image=image_image_1
    )

    #Login Button
    button_image_1 = PhotoImage(
        file=relative_to_assets0("button_1.png"))
    button_1 = Button(
        win1,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=Login_page,
        relief="flat"
    )
    button_1.place(
        x=348.0,
        y=307.0,
        width=305.0,
        height=82.32188415527344
    )

    #SignUp Button
    button_image_2 = PhotoImage(
        file=relative_to_assets0("button_2.png"))
    button_2 = Button(
        win1,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=Signup_page,
        relief="flat"
    )
    button_2.place(
        x=348.0,
        y=424.6781311035156,
        width=305.0,
        height=82.32188415527344
    )
    

    canvas.create_text(
        260.0,
        115.0,
        anchor="nw",
        text="CROP PRICE DETECTOR",
        fill="#FFFFFF",
        font=("NATS", 45 * -1)
    )
    win1.resizable(False, False)
    win1.mainloop()

Home_page()