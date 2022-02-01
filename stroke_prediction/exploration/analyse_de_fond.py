

from math import ceil
from stroke_prediction.exploration.analyse_de_forme import AnalyseForme
from stroke_prediction.exploration.analyse_de_forme import AnalyseForme
from stroke_prediction import plt, sns
import warnings
warnings.filterwarnings('ignore')

class AnalyseFond(AnalyseForme):
    def __init__(self, path = "stroke_prediction/donnees/healthcare-dataset-stroke-data.csv"):
        super().__init__(path)
    
    def visualisation_cible(self, cible):
        proportions = self.data_frame[cible].value_counts(normalize = True)*100
        print("Les proportions des classes de la cible sont : \n", proportions)
        proportions.plot.pie(labels=self.data_frame[cible].unique())
        plt.title("Diagramme en camembert de : ", cible)
        plt.ylabel('')
        plt.show()
        self.data_frame[cible] =  self.data_frame[cible].astype("category")
        
    def transformation_colonnes_categorielles(self):
        for colonne in self.recuperer_colonnes_categorielles():
            self.data_frame[colonne] = self.data_frame[colonne].astype('category')
    
    def visualisation_colonnes_categorielles(self):
        proportions = []
        for colonne in self.recuperer_colonnes_categorielles():
            proportion = self.data_frame[colonne].value_counts(normalize = True)*100
            proportions.append(proportion)
        fig, axs = plt.subplots(ceil(proportions.__len__()/2), 2)
        axs = axs.flat

        for i, p in enumerate(proportions):
            fig.tight_layout(pad = 3)
            sns.barplot(x = p.index, y = p.values, ax = axs[i])
            axs[i].set_title(f"Diagramme en barres {p.name}")
            axs[i].set_xlabel(p.name)
            axs[i].set_ylabel("Pourcentages de valeurs")
        fig.show()
    
    def visualisation_colonnes_non_categorielles(self):
        colonnes_non_categorielles = self.recuperer_colonnes_non_categorielles()
        fig, axs = plt.subplots(ceil(colonnes_non_categorielles.__len__()/2), 2)

        axs = axs.flat

        for i, colonne in enumerate(colonnes_non_categorielles):
            sns.histplot(data = self.data_frame, x = colonne, kde = True, ax = axs[i])
        fig.tight_layout(w_pad = 3, pad = 1.2)
        fig.show()
    
    def relation_categorielle_quantitative(self, colonne_categorielle, data_to_plot):
        """Fonction pour vérifier les relations pouvant exister entre les variables qualitatives
        et quantitatives. Nous allons utiliser la base de données ne comportant pas de valeurs 
        manquantes.
        
        Args:
            colonne_categorielle(str): Nom de la variable catégorielle
        
        Returns:
            None
        """
        colonnes_non_categorielles = self.recuperer_colonnes_non_categorielles()
        fig, axs = plt.subplots(ceil(colonnes_non_categorielles.__len__()/2), 2)

        axs = axs.flat

        for i,column in enumerate(colonnes_non_categorielles):

            sns.histplot(data = data_to_plot, x = column, kde = True, hue = colonne_categorielle, ax = axs[i], palette = "tab10")

            axs[i].set_title(f"Variable {column}", fontsize = 14)

        # fig.delaxes(axs[3])

        fig.tight_layout(pad = 3)
        
        fig.show()
        
    def identification_valeurs_aberrantes(self, data_to_plot, cible):
        
        colonnes_non_categorielles = self.recuperer_colonnes_non_categorielles()
        
        fig, axs = plt.subplots(ceil(colonnes_non_categorielles.__len__()/2), 2)

        axs = axs.flat

        for i,colonne in enumerate(colonnes_non_categorielles):

            sns.boxplot(data = data_to_plot, x = cible, y = colonne, hue = cible, ax = axs[i], palette = "tab10")

            axs[i].set_title(f"Variable {colonne}", fontsize = 14)

        fig.delaxes(axs[3])

        fig.tight_layout(pad = 3)