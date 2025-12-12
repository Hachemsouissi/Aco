import random
import math

# --- Données ---
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
nb_villes = 4

# --- Hyper-paramètres ---
nb_fourmis = 5
max_iter = 20
alpha = 1.0
beta = 2.0
rho = 0.5
Q = 100.0

# Init Phéromones (Matrice)
matrice_phero = []
for i in range(nb_villes):
    ligne = []
    for j in range(nb_villes):
        ligne.append(1.0)  # Valeur initiale
    matrice_phero.append(ligne)

best_chemin = []
best_longueur = 999999

# --- Boucle ---
iterateur = 0
while iterateur < max_iter:

    chemins_fourmis = []
    longueurs_fourmis = []

    for f in range(nb_fourmis):
        # Départ ville 0 (pour faire simple)
        ville_actuelle = 0
        chemin = [0]
        villes_a_visiter = [1, 2, 3]  # On sait qu'il y a 4 villes

        longueur_totale = 0

        # Construction du chemin pas à pas
        while len(villes_a_visiter) > 0:

            # Calcul des probabilités pour chaque ville candidate
            probs = []
            somme_probs = 0.0

            for v_candidat in villes_a_visiter:
                # Tau = phéromone sur l'arête (actuelle -> candidat)
                tau = matrice_phero[ville_actuelle][v_candidat]

                # Eta = 1 / distance (visibilité)
                d = dist[ville_actuelle][v_candidat]
                eta = 1.0 / d

                # Score = tau^alpha * eta^beta
                score = math.pow(tau, alpha) * math.pow(eta, beta)

                probs.append(score)
                somme_probs = somme_probs + score

            # Roulette Wheel (Sélection aléatoire pondérée)
            r = random.uniform(0, somme_probs)
            cumul = 0.0
            choix = -1

            # On cherche sur quel candidat ça tombe
            index_trouve = 0
            for k in range(len(probs)):
                cumul = cumul + probs[k]
                if r <= cumul:
                    choix = villes_a_visiter[k]
                    index_trouve = k
                    break  # Sortir de la boucle for

            # Si jamais l'arrondi float fait rater (cas rare), on prend le dernier
            if choix == -1:
                choix = villes_a_visiter[len(villes_a_visiter) - 1]
                index_trouve = len(villes_a_visiter) - 1

            # Déplacement
            longueur_totale = longueur_totale + dist[ville_actuelle][choix]
            ville_actuelle = choix
            chemin.append(choix)

            # Enlever de la liste manuellement
            villes_a_visiter.pop(index_trouve)

        # Retour au début
        longueur_totale = longueur_totale + dist[ville_actuelle][0]

        chemins_fourmis.append(chemin)
        longueurs_fourmis.append(longueur_totale)

        if longueur_totale < best_longueur:
            best_longueur = longueur_totale
            best_chemin = list(chemin)

    # Mise à jour Phéromones
    # 1. Evaporation
    for i in range(nb_villes):
        for j in range(nb_villes):
            matrice_phero[i][j] = matrice_phero[i][j] * (1.0 - rho)

    # 2. Dépôt
    for f in range(nb_fourmis):
        l = longueurs_fourmis[f]
        ch = chemins_fourmis[f]
        contrib = Q / l

        # On dépose sur chaque arête du chemin
        for k in range(nb_villes - 1):
            v1 = ch[k]
            v2 = ch[k + 1]
            matrice_phero[v1][v2] = matrice_phero[v1][v2] + contrib
            matrice_phero[v2][v1] = matrice_phero[v2][v1] + contrib  # symétrique

        # Arête de retour
        vf = ch[nb_villes - 1]
        vd = ch[0]
        matrice_phero[vf][vd] = matrice_phero[vf][vd] + contrib
        matrice_phero[vd][vf] = matrice_phero[vd][vf] + contrib

    iterateur = iterateur + 1
    print("Tour " + str(iterateur) + " dist min : " + str(best_longueur))

print("Chemin final : " + str(best_chemin))