from flask import Flask,render_template,request
from flask_bootstrap import Bootstrap
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib



app = Flask(__name__)

model1 = joblib.load("model/modelv2.pkl")


@app.route("/")
def index():
	return render_template('index.html')

@app.route("/premierleague")
def brasileirao():
	return render_template("premierleague.html")

@app.route("/submit",methods=["GET","POST"])
def prediction():
	features = [float(x) for x in request.form.values()]
	features = np.array(features)
	scaler = StandardScaler()
	scaled_features = scaler.fit_transform(features.reshape(-1,1))
	scaled_features = scaled_features.reshape(1,-1)
	prediction1 = (model1.predict_proba(scaled_features)[:,1] >=0.321779)
	prediction1 = prediction1[0]
	return render_template('prediction.html', prediction = prediction1)

if __name__ == 'main':
	app.run(debug=True)
bootstrap=Bootstrap(app)
