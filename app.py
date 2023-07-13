from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict_data',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            customer_type=request.form.get('customer_type'),
            age=int(request.form.get('age')),
            type_of_travel=request.form.get('type_of_travel'),
            Class=request.form.get('Class'),
            flight_distance=int(request.form.get('flight_distance')),
            inflight_wifi_service=int(request.form.get('inflight_wifi_service')),
            departure_arrival_time_convenient=int(request.form.get('departure_arrival_time_convenient')),
            ease_of_online_booking=int(request.form.get('ease_of_online_booking')),
            gate_location=int(request.form.get('gate_location')),
            food_and_drink=int(request.form.get('food_and_drink')),
            online_boarding=int(request.form.get('online_boarding')),
            seat_comfort=int(request.form.get('seat_comfort')),
            inflight_entertainment=int(request.form.get('inflight_entertainment')),
            on_board_service=int(request.form.get('on_board_service')),
            leg_room_service=int(request.form.get('leg_room_service')),
            baggage_handling=int(request.form.get('baggage_handling')),
            checkin_service=int(request.form.get('checkin_service')),
            inflight_service=int(request.form.get('inflight_service')),
            cleanliness=int(request.form.get('cleanliness')),
            departure_delay_in_minutes=int(request.form.get('departure_delay_in_minutes')),
            arrival_delay_in_minutes=float(request.form.get('arrival_delay_in_minutes')),
       )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
    res=int(results[0])
    if res==1:
            return render_template('home.html',results="satisfied")
    elif res==0:
            return render_template('home.html',results="not satisfied")
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000,debug=True)        

