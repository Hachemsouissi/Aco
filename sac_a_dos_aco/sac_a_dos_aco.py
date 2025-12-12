import random
import math

# --- Données ---
poids = [2, 3, 6, 5, 1, 4]
valeurs = [8, 7, 6, 5, 4, 3]
capacite = 9
nb_objets = 6

# --- Hyper-paramètres ACO ---
nb_fourmis = 10
nb_iterations = 20
alpha = 1.0  # Importance de la phéromone (l'expérience)
beta = 2.0  # Importance de l'heuristique (le ratio valeur/poids)
rho = 0.5  # Evaporation (0.5 = 50% disparait)
Q = 100.0  # Quantité de phéromone déposée

# Initialisation des phéromones (une valeur par objet)
# Au début, on met une petite valeur partout (par ex 1.0)
pheromones = []
for i in range(nb_objets):
    pheromones.append(1.0)

# Initialisation de l'heuristique (eta)
# On utilise le ratio : Valeur / Poids
heuristiques = []
for i in range(nb_objets):
    ratio = valeurs[i] / poids[i]
    heuristiques.append(ratio)

meilleure_sol_globale = []
meilleure_valeur_globale = 0

# --- Boucle Principale ---
compteur = 0
while compteur < nb_iterations:

    solutions_fourmis = []
    scores_fourmis = []

    # 1. Chaque fourmi construit une solution
    for f in range(nb_fourmis):
        sac_actuel = []  # Liste de 0 ou 1
        poids_actuel = 0

        for obj in range(nb_objets):
            # La fourmi doit décider : je prends l'objet ou pas ?
            # On calcule une probabilité de le prendre

            tau = pheromones[obj]
            eta = heuristiques[obj]

            # Formule : score = tau^alpha * eta^beta
            attractivite = math.pow(tau, alpha) * math.pow(eta, beta)

            # Pour simplifier en "novice", on va dire que l'attractivité
            # définit la chance de prendre l'objet (entre 0 et 100 par exemple)
            # C'est une version simplifiée de ACO pour le sac à dos

            # Si le sac est plein, on ne peut pas prendre
            if poids_actuel + poids[obj] <= capacite:
                # Tirage au sort
                # Plus l'attractivité est haute, plus on a de chance
                seuil = attractivite / (attractivite + 1.0)  # Normalisation simple

                if random.random() < seuil:
                    sac_actuel.append(1)
                    poids_actuel = poids_actuel + poids[obj]
                else:
                    sac_actuel.append(0)
            else:
                sac_actuel.append(0)

        # Calcul valeur
        valeur_f = 0
        for k in range(nb_objets):
            if sac_actuel[k] == 1:
                valeur_f = valeur_f + valeurs[k]

        solutions_fourmis.append(sac_actuel)
        scores_fourmis.append(valeur_f)

        # Check meilleur global
        if valeur_f > meilleure_valeur_globale:
            meilleure_valeur_globale = valeur_f
            meilleure_sol_globale = list(sac_actuel)

    # 2. Mise à jour des Phéromones
    # D'abord l'évaporation
    for k in range(nb_objets):
        pheromones[k] = pheromones[k] * (1.0 - rho)

    # Ensuite le dépôt (seules les fourmis ajoutent)
    for f in range(nb_fourmis):
        score = scores_fourmis[f]
        sol = solutions_fourmis[f]

        # Si la solution est nulle, on ajoute rien
        if score > 0:
            depot = Q / (100.0 / score)  # Formule un peu bidouillée pour donner du poids

            for k in range(nb_objets):
                if sol[k] == 1:
                    pheromones[k] = pheromones[k] + depot

    compteur = compteur + 1
    print("Iteration " + str(compteur) + " - Best Value : " + str(meilleure_valeur_globale))

print("--- FIN ACO Sac a Dos ---")
print("Objets pris : " + str(meilleure_sol_globale))
print("Valeur : " + str(meilleure_valeur_globale))