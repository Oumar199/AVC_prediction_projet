
from stroke_prediction import train_test_split, plt, plot_roc_curve, pd, ttest_ind


class Traitement:
    """Cette classe va nous permettre d'effectuer des traitements sur les données.
    
    Attributes:
        dataframe(pandas.DataFrame): Les données avant tout traitement.
        cible(str): Le nom de la variable cible
    """
    
    def __init__(self, dataframe, cible):
        """Initialisation des attributs
        """
        if cible in dataframe.columns:
            self.dataframe = dataframe.copy()
            self.cible = cible
        else:
            raise ValueError("Le nom de la cible ne correspond à aucune colonne")
    
    def student_test(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame, colonnes: list, alpha: float):
        """Cette fonction permet d'effectuer un test de student.
        
        Args:
            dataframe1(pandas.DataFrame): Les données de la première classe.
            dataframe2(pandas.DataFrame): Les données de la deuxième classe.
            colonne(list): Les noms des colonnes sur les quelles on va réaliser le test.
            alpha(float): Le taux d'erreur alpha de l'hypothèse nulle.
        
        Returns:
            None
        """
        
        # Choix d'un échantillon pour le dataframe qui a la plus grande taille
        if dataframe1.shape[0] <= dataframe2.shape[0]:
            dataframe2 = dataframe2.sample(dataframe1.shape[0])
        else:
            dataframe1 = dataframe1.sample(dataframe2.shape[0])
            
        for colonne in colonnes:
            stat, p = ttest_ind(dataframe1[colonne], dataframe2[colonne])
            if  p < alpha:
                print(f"{colonne:-<30} : H0 rejetée")
            else:
                print(f"{colonne:-<30} : H0 non rejetée")
    
    def colonne_existe(self, colonne):
        """Fonction permettant de vérifier si une colonne existe.
        """
        return True if colonne in self.dataframe.columns else False
    
    def supprimer_colonnes(self, colonnes):
        """Cette fonction permet de supprimer des colonnes.
        """
        for colonne in colonnes:
            if self.colonne_existe(colonne):
                self.dataframe.drop(colonne, axis = 1, inplace = True)
            else:
                raise ValueError(f"La colonne {colonne} n'existe pas.")
    
    def modification_categorie(self, colonne, categorie_a_modif, categorie_de_modif):
        """Cette fonction permet de modifier une catégorie par une autre catégorie au sein
        d'une variable catégorielle.
        """
        modif = lambda x: categorie_de_modif if x == categorie_a_modif else x
        if self.colonne_existe(colonne):
            if self.dataframe[colonne].dtype == "category":
                self.dataframe[colonne] = self.dataframe[colonne].apply(modif)
                self.dataframe[colonne] = self.dataframe[colonne].astype('category')
            else:
                raise TypeError(f"La colonne {colonne} n'est pas une colonne catégorielle.")
        else:
            raise ValueError(f"La colonne {colonne} n'existe pas.")
    
    def supprimer_aberrantes(self, colonne):
        """Cette fonction permet de remplacer les valeurs aberrantes par la médiane 
        des données suivant une colonne à préciser
        """
        if self.colonne_existe(colonne):
            quantile_1 = self.dataframe[colonne].quantile(0.25)
            quantile_2 = self.dataframe[colonne].quantile(0.75)
            mediane = self.dataframe[colonne].median()
            inter_quantile = quantile_2 - quantile_1
            limite_bas = quantile_1 - 1.5 * inter_quantile
            limite_haut = quantile_2 + 1.5 * inter_quantile
#             remplacer_aberrante = lambda x : mediane if (x < limite_bas or x > limite_haut) else x
#             self.dataframe[colonne] = self.dataframe[colonne].apply(remplacer_aberrante)
            self.dataframe = self.dataframe[(self.dataframe[colonne] > limite_bas) & (self.dataframe[colonne] < limite_haut)]
        else:
            raise ValueError(f"La colonne {colonne} n'existe pas.")
    
    def separer_donnees(self, dataframe):
        """Cette fonction permet de séparer les données d'entraînement des données de test
        """
        y = dataframe[self.cible]
        
        X = dataframe.drop(self.cible, axis = 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
        
        return X_train, X_test, y_train, y_test
    
    def encodage(self):
        """Cette fonction permet d'encoder les valeurs qualitatives
        """
        dataframe = self.dataframe.copy()
        for colonne in dataframe.select_dtypes("category").columns:
            dataframe[colonne] = self.dataframe[colonne].cat.codes
        return dataframe
    
    def selection_variables(self, modele, X_train, y_train):
        """Cette fonction permet de sélectionner les variables qui apportent le plus d'information
        au modèle donné en paramètre.
        """
        from sklearn.feature_selection import SelectFromModel
        selector = SelectFromModel(modele)
        selector.fit(X_train, y_train)
        print(f"Colonnes choisies : \n", X_train.columns[selector.get_support()])
        return selector
    
    def premier_entrainement(self, X_train, y_train):
        """Cette fonction permet d'entraîner le premier modèle pour vérifier si le traitement effectué
        est bon.
        """
        from sklearn.tree import DecisionTreeClassifier
        
        modele = DecisionTreeClassifier(random_state = 4)
        
        modele.fit(X_train, y_train)
        
        return modele
    
    def evaluation(self, modele, nom_modele, X_train, X_test, y_train, y_test, plot_roc = False, show_figure = False):
        """Cette fonction permet d'évaluer un modèle
        """
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        from sklearn.model_selection import learning_curve
        import numpy as np
        
        y_pred = modele.predict(X_test)
        
        print(f"La precision du modèle : {accuracy_score(y_test, y_pred)}")
        print(f"Matrice de confusion : \n{confusion_matrix(y_test, y_pred)}")
        print(f"Rapport de classification : \n{classification_report(y_test, y_pred)}")
        
        N, train_score, test_score = learning_curve(modele, X_train, y_train, cv = 5, train_sizes = np.linspace(0.1, 1, 10))
        
        fig, axs = plt.subplots(1, 2, figsize = (17, 8))  
        
        axs[0].plot(N, train_score.mean(axis = 1), label = "score_entrainement")
        axs[0].plot(N, test_score.mean(axis = 1), label = "score_test")
        axs[0].set_title(f"Evaluation {nom_modele}")
        axs[0].set_xlabel(f"Tailles")
        axs[0].set_ylabel(f"Scores")
        axs[0].legend()
        
        plot_roc_curve(modele, X_test, y_test, ax = axs[1])
        axs[1].set_title(f"Courbe ROC {nom_modele}")
         
        fig.tight_layout(w_pad = 3)
        if show_figure:
            plt.show()
        