
[x] Modifier prompt pour 
    -assigner un score de douleur/besoin de 1 à 10
    -donner une reformulation plus complète et explicite (identifier les informations importantes)

[] Modifier le workflow d'extraction de douleurs besoins pour enlever les doublons.
    -ajout d'un nouveau prompt pour identifier en une fois rapidement tous les pain points 
    du thread  avec nombre de douleurs et description rapide pour chaque douleur 10 mot max.
    -ajout d'un nouveau model pydantic (post_pains_needs) pour modéliser l'output de ce prompt.
    -modification des ancien noeud et prompt pour reformuler avec beaucoup plus de
    précision chaque douleur/besoin inclue dans post_pains_needs pour être sur de ne 
    pas avoir de doublons

[] Créer un agent qui créer des catégories pour chaque point de douleur/besoin identifiés.
    -classer les catégories par nombre de personnes ayant exprimés leurs douleurs/besoin
    -classer par catégorie les plus douleureuses.

[]  Tracing Langsmith

[] Productionniser les scripts pour Création interface

[] Estimation coût par recherche

[] Création interface

[] Mettre en prod (à spécifier quoi faire exactement)
