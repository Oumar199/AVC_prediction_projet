import argparse
import argparse
from stroke_prediction.exploration import AnalyseFond
from stroke_prediction.modelisation import SelectionModele
from stroke_prediction.preprocess import Traitement
from stroke_prediction import sns, tree, pickle, plt, StratifiedKFold, recall_score, joblib

def run(preprocess, modelisation):
    exploration = AnalyseFond()
    exploration.visualisation_cible()
    exploration.transformation_colonnes_categorielles()
    if preprocess == "true":
        traitement = Traitement(exploration.data_frame, "stroke")
        traitement.supprimer_colonnes(['bmi', 'work_type', 'ever_married', 'smoking_status', 'id'])    
        print(traitement.dataframe.columns)
        traitement.supprimer_aberrantes("age")
        traitement.supprimer_aberrantes("avg_glucose_level")
        print(traitement.dataframe.shape[0])
        new_data_frame = traitement.encodage()
        X_train, X_test, y_train, y_test = traitement.separer_donnees(new_data_frame)
        print("Formes : \n", X_train.shape, X_test.shape)
        print("-"*20)
        sns.set_theme(style = "darkgrid")
        selector = traitement.selection_variables(tree.DecisionTreeClassifier(random_state = 4), X_train, y_train)
        colonnes_choisies = X_train.columns[selector.get_support()]
        X_train, X_test = X_train[colonnes_choisies], X_test[colonnes_choisies]
        modele = traitement.premier_entrainement(X_train, y_train)
        traitement.evaluation(modele, "Arbre de décision", X_train, X_test, y_train, y_test)
        with open("stroke_prediction/donnees/donnees_traitees/X_train.txt", "wb") as f:
            pick = pickle.Pickler(f)
            pick.dump(X_train)
        
        with open("stroke_prediction/donnees/donnees_traitees/X_test.txt", "wb") as f:
            pick = pickle.Pickler(f)
            pick.dump(X_test)
        
        with open("stroke_prediction/donnees/donnees_traitees/y_train.txt", "wb") as f:
            pick = pickle.Pickler(f)
            pick.dump(y_train)
        
        with open("stroke_prediction/donnees/donnees_traitees/y.test.txt", "wb") as f:
            pick = pickle.Pickler(f)
            pick.dump(y_test)
        
    elif modelisation == "true":
        with open("stroke_prediction/donnees/donnees_traitees/X_train.txt", "rb") as f:
            pick = pickle.Unpickler(f)
            X_train = pick.load()
        
        with open("stroke_prediction/donnees/donnees_traitees/X_test.txt", "rb") as f:
            pick = pickle.Unpickler(f)
            X_test = pick.load()
        
        with open("stroke_prediction/donnees/donnees_traitees/y_train.txt", "rb") as f:
            pick = pickle.Unpickler(f)
            y_train = pick.load()
    
        with open("stroke_prediction/donnees/donnees_traitees/y.test.txt", "rb") as f:
            pick = pickle.Unpickler(f)
            y_test = pick.load()
        selection_modele = SelectionModele()
        generateur_modeles = selection_modele.multiple_entrainements(X_train, X_test, y_train, y_test)
        for nom, modele in generateur_modeles:
            # plt.show()
            pass
        cv = StratifiedKFold(5)
        print(X_test)
        y_pred = selection_modele.modele_finale(modele, X_test, -4.7)
        
        print("La sensibilité du modèle choisi est de : {} %.".format(recall_score(y_test, y_pred).round(3)))
        selection_modele.evaluation(modele, nom, X_train, X_test, y_train, y_test, show_figure=True)
        joblib.dump(modele, "stroke_prediction/modeles/logistic_model.joblib")
        
        
        
        
        
    # analyse_forme = analyse_de_forme.AnalyseForme()
    # analyse_forme.identification_valeurs_manquantes()

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-preprocess", type=str, help="Donnez la valeur 'true' si vous voulez appliquer le preprocessing. Type string.", default="true")
    parse.add_argument("-modelisation", type=str, help="Donnez la valeur 'true' si vous voulez appliquer la modelisation. Type string.", default="true")
    args = parse.parse_args() 
    
    run(args.preprocess, args.modelisation)