# Projet de Programmation 2 Universite de Montpellier

**Sujet: Suivi d'objet avec filtre a particule.**

## Introduction

Le projet se concentrer sur le suivi de seiches grace au filtre a particule sur des videos. 
Les informations initiales sur la seiche suivie sont donnees par une premiere detection, effectue par le reseau de neurone YOLOv7. Ce modele de reseau de neurone a ete fine-tune au prealable sur un dataset de seiche.
Ces informations sont alors utilises pour initialiser les particules du filtre, afin d'avoir une chance plus faible de diverger et une convergence plus rapide.
L'etape de mis a jour des particules est faites en passant un patch de la frame en cours de traitement dans un descripteur, puis une mesure de similarite est effectue sur le resultat du descripteur avec le resultat du descripteur correspondant au patch de la meilleure particule a l'etape precedente.


## Objectifs

L'objectif du projet est de suivre un objet dans une sequence d'image en utilisant un agorithme de filtre a particule.
Il se decoupera donc en plusieurs sous objectifs:

- [x] Creer le repo initial.
- [x] Organiser la structure fichier du projet.
- [ ] Creer une template de rapport en Latex.
- [ ] Creer un dataset de seiches pour YOLOv7.
- [ ] Fine-tuning de YOLOv7 sur notre dataset.
- [ ] Creer un programme pour la detection initiale de seiches.
- [ ] Implementer plusieurs descripteurs d'image (environ 5).
- [ ] Implementer plusieurs mesures de similarite (environ 5).
- [ ] Definir un ou plusieurs vecteurs de representation de ce que l'on suit.
- [x] Implementer un filtre a particule modulable.
- [ ] Creer un programme principal qui va gerer les transmissions de donnees entre modules.
- [ ] Implementer des metriques d'evaluation pour le programme principal.
- [ ] Test des differentes combinaisons entre les descripteurs et les mesures de similarite.
- [ ] Evaluation de notre application sur le terrain (Aquarium). 
- [ ] (Optionnelle) Implementer des metriques d'evaluation pour chaque modules.


## Dataset
TODO


## Fine-tuning
TODO


## Descripteurs
TODO


## Mesures de similarite
TODO


## Filtre a particule
Le principe general du filtre a particule implemente est le suivant:
  1. On genere N particules, chaque particule represente un etat possible dans lequel ce trouve l'objet a suivre.
  2. A chaque particule est associe une probabilite initial, donnee par notre premiere detection, representant notre confiance en une particule de bien represente l'etat de notre objet.
  3. On predit l'etat de nos particules grace a une modelisation du systeme que l'on suppose assez proche du systeme reel.
  4. On met a jour la distribution de probabilite modelise par nos particules en calculant la similarite entre notre estimation a la frame precedente et chacune des particules. Ce calcul nous permet alors de modifier notre confiance dans chaque particule.
  5. On regarde ensuite combien de particules participent reellement au suivi de l'objet. Si ce nombre passe en dessous d'un certain seuil, passe a l'etape 6, sinon, on passe a l'etape 7.
  6. On resample les particules pour que plus de particules participent au suivi de l'objet (duplication des meilleures particules et rejet des autres). 
  7. On calcule l'etat estime dans lequel notre objet ce trouve en faisant la moyenne des particules ponderes par leur probabilite.
  8. On revient a l'etape 3 et on recommence le processus.


## Evaluations
TODO
