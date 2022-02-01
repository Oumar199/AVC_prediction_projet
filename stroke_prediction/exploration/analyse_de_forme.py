from stroke_prediction import pd, px


class AnalyseForme:
    def __init__(self, path = "stroke_prediction/donnees/healthcare-dataset-stroke-data.csv"):
        self.data_frame = pd.read_csv(path)
        print("Récupération des données effectuée avec succès :\n", self.data_frame.head(10))
            
    def identification_cible(self, nom_cible):
        print("Le type de la cible\n"+"-"*20+"\n", self.data_frame[nom_cible].dtype)
        print("Les valeurs possibles de la cible\n")
        
    def nombre_lignes_colonnes(self):
        print("Forme des données :\n", self.data_frame.shape)
        
    def types_variables(self):
        for colonne in self.data_frame.columns:
            print("Type de la colonne {} : {}".format(colonne, self.data_frame[colonne].dtype))
            print("-"*20)
            
    def recuperer_colonnes_categorielles(self, categorical_columns_sup = ["hypertension", "heart_disease"]):
        categorical_columns = self.data_frame.select_dtypes('object').columns.tolist()
        categorical_columns.extend(categorical_columns_sup)
        return categorical_columns
    
    def verifier_valeurs_object(self):
        for colonne in self.data_frame.select_dtypes('object').columns:
            print(f"Valeurs uniques de {colonne:-<20} {self.data_frame[colonne].unique()}\n")
    
    def verifier_valeurs_hyp_hear_dis(self):
        for colonne in self.recuperer_colonnes_categorielles()[:2]:
            print(f"Unique values of {colonne:-<20} {self.data_frame[colonne].unique()}\n")
            
    def recuperer_colonnes_non_categorielles(self):
        non_categorical_columns = ["age", "avg_glucose_level", "bmi"]
        return non_categorical_columns
    
    def description_donnees(self):
        print("La description des données non catégorielles : ", self.data_frame[self.recuperer_colonnes_non_categorielles].describe())
        
    def identification_valeurs_manquantes(self):
        print("Nombre de valeurs manquantes par colonne : \n", self.data_frame.isna().sum())
        print("-"*20)
        na_values = self.data_frame.isna()
        fig = px.imshow(na_values, title = "Visualisation des données manquantes")
        fig.show()
        
    def identification_valeurs_redondantes(self):
        print("Nombre de valeurs redondantes : ", self.data_frame.duplicated().sum())
        