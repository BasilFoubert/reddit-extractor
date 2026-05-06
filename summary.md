CAHIER DES CHARGES
Reddit Pain Point Detector
Projet principal — Système ML de monitoring continu

**Objectif**

Construire un système de monitoring continu des besoins non satisfaits exprimés sur Reddit dans un domaine paramétrable. Le système ingère des données de manière incrémentale, recalcule l'analyse périodiquement, et expose en permanence une vision à jour via une API REST et un serveur MCP.

---

**COLLECTE INCRÉMENTALE**

- Collecte régulière depuis la dernière exécution sur des communautés ciblées
- Nettoyage, déduplication, horodatage, conservation des métadonnées
- Filtrage : bots, modérateurs, supprimés, trop courts

→ Quels critères permettent de distinguer un contenu porteur d'un pain point d'un bruit anecdotique ?

---

**EXTRACTION & STRUCTURATION DES PAIN POINTS**

- Appel LLM avec chain-of-thought pour chaque document retenu
- Structure produite en trois couches : verbatim exact / reformulation directe / raisonnement ancré dans le texte source
- Score de confiance + flag is_pain_point — documents non fondés écartés
- Rôle de LLM-as-judge : le raisonnement doit citer des éléments concrets du verbatim
- Documents validés alimentent le dataset de test annoté

→ Comment garantir que la reformulation reste fidèle au sens original sans extrapoler ?

Exemple à retravailler:
si sur les 4300 commentaire je génére 10000 données structuré comme celle ci est-ce assez pour un premier clustering?

{
  "subreddit":"r/ciso",
  "permalink": "/r/ciso/comments/1nii7l/cyber_security_skills_gap/",
    "verbatim": "Les CISOs sont submergés de sollicitations non ciblées",
    "pain_point_reformulated": "Manque de ..." 
}

---

**GÉNÉRATION D'EMBEDDINGS**

- Embedding calculé sur verbatim + reformulation concaténés (Option C)
- Verbatim, reformulation et raisonnement stockés séparément en métadonnées
- Seuls les nouveaux documents sont traités — embeddings existants conservés

→ Quelle concaténation verbatim + reformulation préserve le mieux la cohérence sémantique pour le clustering ?

---

**RECALCUL HEBDOMADAIRE DU CLUSTERING**

- Relance périodique sur l'ensemble du corpus
- Labellisation automatique de chaque cluster
- Archivage de l'état précédent pour comparaison dans le temps

→ Comment valider que les clusters reflètent des réalités business et non de simples proximités lexicales ?

---

**COUCHE AGENTIQUE**

- Enrichissement des clusters par RAG sur le vector store
- Exemples remontés toujours en verbatim — jamais en reformulation
- Estimation du volume et de la récurrence par cluster
- Production d'un rapport structuré, traçable et auditable

→ Comment garantir que l'agent s'appuie uniquement sur les verbatims sources sans extrapoler ?

---

**ÉVALUATION & MLOPS**

- Évaluation indépendante de chaque brique
- Dataset de test alimenté par les extractions LLM validées manuellement
- Au moins cinq configurations comparées : chunking, modèle d'embedding, algorithme de clustering
- Traçabilité des runs, monitoring de la dérive dans le temps

→ Quelles métriques reflètent réellement l'utilité du système pour un utilisateur final ?

---

**DÉPLOIEMENT — DEUX INTERFACES, UN SEUL MOTEUR**

- API REST : documentée, containerisée, déployée publiquement, état courant des pain points
- Serveur MCP : outils appelables par un agent IA — état courant, évolution, détail d'un thème, verbatims récents

→ Quels outils MCP ont une valeur réelle pour un agent qui interroge ce système ?

---

**Cycle de fonctionnement**

- Fréquent — collecte, extraction LLM, embeddings, mise à jour du vector store
- Hebdomadaire — recalcul clustering, relabellisation, archivage, mise à jour API et MCP

---

**Livrables**

- Pipeline d'ingestion incrémental — fonctionnel, automatisé, reproductible
- API REST déployée — documentée, containerisée, accessible publiquement
- Serveur MCP — état courant et historique
- Comparaison de configurations — cinq runs documentés avec métriques
- Dataset de test annoté — ground truth validée manuellement
- GitHub & write-up — README technique, démo visuelle, article publié

---

La valeur de ce projet tient à sa complétude et à sa tenue dans le temps. Un système qui ingère, extrait, analyse, évalue et expose de manière fiable sur plusieurs semaines est plus convaincant qu'un système sophistiqué qui s'arrête au notebook. Les choix techniques sont laissés ouverts — ce sont eux qui devront être défendus.