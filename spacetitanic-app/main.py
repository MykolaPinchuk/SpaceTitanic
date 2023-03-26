#import Flask 
import numpy as np
import joblib, sklearn
from flask import Flask, render_template, request
#create an instance of Flask
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        planet_earth = request.form.get('planet_earth')
        planet_europa = request.form.get('planet_europa')
        cryosleep = request.form.get('cryosleep')
        deck_g = request.form.get('deck_g')
        starboard = request.form.get('starboard')
        zero_service = request.form.get('zero_service')
        room_service = request.form.get('room_service')
        food_court = request.form.get('food_court')
        shopping_mall = request.form.get('shopping_mall')
        spa = request.form.get('spa')
        vr_deck = request.form.get('vr_deck')        
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(planet_earth, planet_europa, cryosleep, deck_g, starboard, zero_service, room_service, food_court, shopping_mall, spa, vr_deck)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
        except ValueError:
            return "Please Enter valid values"
        pass        
    pass
def preprocessDataAndPredict(planet_earth, planet_europa, cryosleep, deck_g, starboard, zero_service, room_service, food_court, shopping_mall, spa, vr_deck):
    test_data = [planet_earth, planet_europa, cryosleep, deck_g, starboard, zero_service, room_service, food_court, shopping_mall, spa, vr_deck]
    print(test_data)
    test_data = np.array(test_data).astype(np.float) 
    test_data = test_data.reshape(1,-1)
    print(test_data)
    file = open("rf_model.pkl","rb")
    trained_model = joblib.load(file)
    prediction = trained_model.predict(test_data)
    return prediction
    pass
if __name__ == '__main__':
    app.run(debug=True)

