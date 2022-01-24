from stroke_prediction import pd

def recuperer_donnees():
    data_stroke = pd.read_csv("stroke_prediction/donnees/healthcare-dataset-stroke-data.csv")
    return data_stroke.copy()


    