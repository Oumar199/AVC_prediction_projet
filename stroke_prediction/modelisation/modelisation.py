
from stroke_prediction.preprocess.preprocess import Traitement
from stroke_prediction import plt, RandomizedSearchCV, np

class SelectionModele(Traitement):
    """Cette classe permet d'effectuer plusieurs entraînement avec un certain nombre de modèles et hérite
    des méthodes de la classe Traitement
    Attributes:
        None    
    """
    
    def __init__(self):
        pass
    
    def multiple_entrainements(self, X_train, X_test, y_train, y_test):
        """
        """
        from sklearn.compose import make_column_transformer, make_column_selector
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        
        variables_numeriques = make_column_selector(dtype_include=np.number)
        
        pipeline_numerique = make_pipeline(StandardScaler(), PolynomialFeatures())
        
        preprocesseur = make_column_transformer((pipeline_numerique, variables_numeriques))
        
        Forest = make_pipeline(preprocesseur, RandomForestClassifier(random_state = 4))
        Adaboost = make_pipeline(preprocesseur, AdaBoostClassifier(random_state = 4))
        Svc = make_pipeline(preprocesseur, SVC(random_state = 4))
        KNeighbors = make_pipeline(preprocesseur, KNeighborsClassifier())
        Logistic = make_pipeline(preprocesseur, LogisticRegression(random_state = 4))
        
        modeles_dict = {"Forêt aléatoire" : Forest,
                       "Adaboost" : Adaboost,
                       "SVC" : Svc,
                       "KNN" : KNeighbors,
                       "Logistic" : Logistic
                      }
        
        for nom, modele in modeles_dict.items():
            print(f"{'-'*15}\nPour le modèle {nom} :")
            modele.fit(X_train, y_train)
            self.evaluation(modele, nom, X_train, X_test, y_train, y_test, True)
            yield nom, modele
        
    def courbe_sensibilite_precision(self, modele, X_test, y_test):
        """Une fonction qui permet de tracer la courbe de Precision-Recall pour le choix d'un seuil.
        Args:
            modele (pipeline): Le modèle choisi
            X_test (pandas.DataFrame): Les caractéristiques de test
            y_test (pandas.DataFrame): Les données test de la cible
        
        Returns:
            None
        """
        from sklearn.metrics import precision_recall_curve
        
        precision, sensibilite, seuil = precision_recall_curve(y_test, modele.decision_function(X_test))
        
        plt.plot(seuil, precision[:-1], label = 'precision')
        plt.plot(seuil, sensibilite[:-1], label = 'sensibilité')
        plt.legend()
        
    def recherche_par_grilles(self, X_train, y_train, modele, params, cv = 5):
        grid = RandomizedSearchCV(modele, params, n_iter = 20, scoring = "f1", cv = cv, random_state = 21)
        grid.fit(X_train, y_train)
        print("Les meilleurs paramètres : \n", grid.best_params_)
        
    def modele_finale(self, modele, X, seuil = 0):
        """Modèle de prédiction basé sur le modèle fourni en paramètre.
        Args:
            modele (pipeline): Le modèle choisi
            X (pandas.DataFrame ou Autres): les données à prédire
            seuil (float): Le seuil de prédiction
        """
        return modele.decision_function(X) > seuil