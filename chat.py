# from crypt import methods
import pandas as pd
from flask import Flask, render_template, request, jsonify
# import pickle
# with open('model_pickle','rb') as f:
#      logisticRegr=pickle.load(f)
app= Flask(__name__)


model = pd.read_pickle("./model_pickle.pkl")
le = pd.read_pickle("./le.pkl")
vec = pd.read_pickle("./vectorizer.pkl")

def get_res(model, vec,le, text):
    vect_txt = vec.transform([text])
    res = model.predict(vect_txt )
    predicted_text = le.inverse_transform(res)
    return predicted_text[0], res[0]


@app.route('/')
def home():
     return render_template("index.html")


@app.route('/chatbot', methods =["GET"])
def chatbot():
     return render_template("chatbot.html")

@app.route('/res', methods =["GET", "POST"])
def chat():
     if request.method =="POST":
          data = request.form.to_dict()
          data = request.get_json()
          print(data)
          print(get_res(model,vec, le, "Waht is mental health"))
          text, label = get_res(model,vec, le, data['Questions'])
          return jsonify({"text":text})
     return jsonify({"text":None, "label":None})


@app.get('/depression')
def dep():
     return render_template("depression.html")

@app.get('/anxiety')
def anxiety():
     return render_template("anxiety.html")

@app.get('/sch')
def sch():
     return render_template("sch.html")
@app.get('/disorder')
def order():
     return render_template("disorder.html")



if __name__ == "__main__":
 app.run(debug=True)
