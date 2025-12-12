import random

# --- Paramètres du problème ---
poids_objets = [2, 3, 6, 5, 1, 4]
valeur_objets = [8, 7, 6, 5, 4, 3]
capacite_sac = 9
nombre_objets = 6

# --- Hyper-paramètres PSO ---
nombre_particules = 20
iterations = 50

# Coefficients (Inertie, Cognitif, Social)
w = 0.7  # Poids de l'inertie (garde la vitesse d'avant)
c1 = 1.5  # Tendance à revenir vers son meilleur souvenir
c2 = 1.5  # Tendance à aller vers le meilleur du groupe

# --- Initialisation ---
# Structure : On va avoir des listes pour tout stocker
# positions[i] = liste des valeurs de la particule i
# vitesses[i] = liste des vitesses de la particule i
# pbest_pos[i] = la meilleure position connue par la particule i
# pbest_score[i] = le meilleur score connu par la particule i

positions = []
vitesses = []
pbest_pos = []
pbest_score = []

gbest_pos = []  # Meilleure position de tout le groupe
gbest_score = -1  # Meilleur score de tout le groupe

# Création des particules
for i in range(nombre_particules):
    pos_temp = []
    vit_temp = []
    for j in range(nombre_objets):
        # Position aléatoire entre 0 et 1
        pos_temp.append(random.random())
        # Vitesse petite au début
        vit_temp.append(random.uniform(-0.1, 0.1))

    positions.append(pos_temp)
    vitesses.append(vit_temp)
    pbest_pos.append(list(pos_temp))  # Copie
    pbest_score.append(0)

# --- Boucle Principale ---
compteur = 0
while compteur < iterations:

    for i in range(nombre_particules):

        # 1. EVALUATION (Décocage : Float -> 0 ou 1)
        poids_total = 0
        valeur_totale = 0
        solution_binaire = []  # Juste pour vérifier

        for j in range(nombre_objets):
            valeur_position = positions[i][j]

            # Si > 0.5 on prend l'objet (1), sinon 0
            if valeur_position > 0.5:
                poids_total = poids_total + poids_objets[j]
                valeur_totale = valeur_totale + valeur_objets[j]
                solution_binaire.append(1)
            else:
                solution_binaire.append(0)

        # Pénalité si trop lourd
        if poids_total > capacite_sac:
            valeur_totale = 0

        # 2. Mise à jour PBEST (Meilleur personnel)
        if valeur_totale > pbest_score[i]:
            pbest_score[i] = valeur_totale
            pbest_pos[i] = list(positions[i])

        # 3. Mise à jour GBEST (Meilleur global)
        if valeur_totale > gbest_score:
            gbest_score = valeur_totale
            gbest_pos = list(positions[i])

    # 4. Mise à jour Vitesse et Position
    for i in range(nombre_particules):
        for j in range(nombre_objets):
            r1 = random.random()
            r2 = random.random()

            vitesse_actuelle = vitesses[i][j]
            pos_actuelle = positions[i][j]
            meilleur_perso = pbest_pos[i][j]
            meilleur_global = gbest_pos[j]

            # Formule PSO classique
            nouvelle_vitesse = (w * vitesse_actuelle) + (c1 * r1 * (meilleur_perso - pos_actuelle)) + (
                        c2 * r2 * (meilleur_global - pos_actuelle))

            # Limiter la vitesse pour ne pas exploser (clamping)
            if nouvelle_vitesse > 1.0:
                nouvelle_vitesse = 1.0
            if nouvelle_vitesse < -1.0:
                nouvelle_vitesse = -1.0

            vitesses[i][j] = nouvelle_vitesse

            # Nouvelle position
            nouvelle_pos = pos_actuelle + nouvelle_vitesse

            # On garde les positions entre 0 et 1 pour simplifier
            if nouvelle_pos > 1.0:
                nouvelle_pos = 1.0
            if nouvelle_pos < 0.0:
                nouvelle_pos = 0.0

            positions[i][j] = nouvelle_pos

    compteur = compteur + 1
    print("Iteration " + str(compteur) + " - Meilleur Score : " + str(gbest_score))

# Résultat final
print("--- Solution Trouvée ---")
final_choix = []
for val in gbest_pos:
    if val > 0.5:
        final_choix.append(1)
    else:
        final_choix.append(0)
print(final_choix)
print("Valeur : " + str(gbest_score))