#import Flask 
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
        starboard = request.form.get('stargboard')
        zero_service = request.form.get('zero_service')
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(planet_earth, planet_europa, cryosleep, deck_g, starboard, zero_service)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
        except ValueError:
            return "Please Enter valid values"
        pass        
    pass
def preprocessDataAndPredict(planet_earth, planet_europa, cryosleep, deck_g, starboard, zero_service):
    test_data = [planet_europa,	planet_europa, cryosleep, deck_g, starboard, zero_service]
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

