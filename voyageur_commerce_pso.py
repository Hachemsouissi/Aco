import random

matrice_dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
nb_villes = 4
nb_particules = 20
max_iter = 50
w = 0.7
c1 = 2.0
c2 = 2.0

# --- Initialisation ---
positions = []
vitesses = []
pbest_pos = []
pbest_valeur = []

gbest_pos = []
gbest_valeur = 999999

for i in range(nb_particules):
    p = []
    v = []
    for k in range(nb_villes):
        p.append(random.uniform(0, 10))  # Priorité entre 0 et 10
        v.append(random.uniform(-1, 1))
    positions.append(p)
    vitesses.append(v)
    pbest_pos.append(list(p))
    pbest_valeur.append(999999)

# --- Boucle ---
iterateur = 0
while iterateur < max_iter:

    for i in range(nb_particules):

        # 1. Reconstruction du chemin (Tri basé sur les valeurs de position)
        # On fait une liste de paires [valeur, index_ville]
        liste_priorites = []
        for k in range(nb_villes):
            pair = [positions[i][k], k]
            liste_priorites.append(pair)

        # Tri manuel (Tri à bulles basique) pour ordonner les villes
        for k in range(nb_villes):
            for m in range(0, nb_villes - k - 1):
                elem1 = liste_priorites[m]
                elem2 = liste_priorites[m + 1]
                # On compare les priorités (index 0)
                if elem1[0] > elem2[0]:
                    # Echange
                    temp = liste_priorites[m]
                    liste_priorites[m] = liste_priorites[m + 1]
                    liste_priorites[m + 1] = temp

        # Le chemin est l'ordre des index après le tri
        chemin = []
        for k in range(nb_villes):
            chemin.append(liste_priorites[k][1])

        # 2. Calcul distance
        dist = 0
        for k in range(nb_villes - 1):
            v1 = chemin[k]
            v2 = chemin[k + 1]
            dist = dist + matrice_dist[v1][v2]
        # Retour
        dist = dist + matrice_dist[chemin[nb_villes - 1]][chemin[0]]

        # 3. Mises à jour Best
        if dist < pbest_valeur[i]:
            pbest_valeur[i] = dist
            pbest_pos[i] = list(positions[i])

        if dist < gbest_valeur:
            gbest_valeur = dist
            gbest_pos = list(positions[i])

    # 4. Mouvement
    for i in range(nb_particules):
        for k in range(nb_villes):
            r1 = random.random()
            r2 = random.random()

            vit = vitesses[i][k]
            pos = positions[i][k]
            pb = pbest_pos[i][k]
            gb = gbest_pos[k]

            new_v = (w * vit) + (c1 * r1 * (pb - pos)) + (c2 * r2 * (gb - pos))
            new_p = pos + new_v

            vitesses[i][k] = new_v
            positions[i][k] = new_p

    iterateur = iterateur + 1
    print("Gen " + str(iterateur) + " Min Distance : " + str(gbest_valeur))

print("--- Meilleur Chemin ---")
# On doit reconstruire le chemin une dernière fois pour l'afficher
liste_fin = []
for k in range(nb_villes):
    liste_fin.append([gbest_pos[k], k])

for k in range(nb_villes):
    for m in range(0, nb_villes - k - 1):
        if liste_fin[m][0] > liste_fin[m + 1][0]:
            temp = liste_fin[m]
            liste_fin[m] = liste_fin[m + 1]
            liste_fin[m + 1] = temp

chemin_final = []
for k in range(nb_villes):
    chemin_final.append(liste_fin[k][1])

print(chemin_final)