
def run():
    from stroke_prediction.exploration import analyse_de_forme
    analyse_forme = analyse_de_forme.AnalyseForme()
    analyse_forme.identification_valeurs_manquantes()

if __name__ == "__main__":
    run()