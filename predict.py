
import argparse
from stroke_prediction import joblib, pd

def predict(age, avg_glucose_level):
    model = joblib.load("stroke_prediction/modeles/logistic_model.joblib")
    sample = pd.DataFrame({"age": [age], "avg_glucose_level": [avg_glucose_level]})
    y_pred = model.predict(sample)
    print("Le patient est atteint." if y_pred == 1 else "Le patient n'est pas atteint.")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("age", type=int, help="Indiquer votre age en entier")
    parse.add_argument("avg_glucose_level", type=float, help="Indiquer le niveau de glucose dans le sang en décimale (mg/dL)")
    args = parse.parse_args()
    predict(args.age, args.avg_glucose_level)