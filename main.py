"""
=============================================================================
  CARTOGRAPHIE SÉMANTIQUE DES PROGRAMMES LÉGISLATIFS — MULTI-ANNÉES
=============================================================================

OBJECTIF :
    Extraire les thèmes dominants des professions de foi des candidats aux
    élections législatives, projeter ces documents dans un espace thématique
    2D, et comparer l'évolution des partis et des thèmes au fil du temps.

STRUCTURE DES DONNÉES ATTENDUE :
    - Un répertoire par année : legislatives_1981/, legislatives_1988/, etc.
    - Chaque répertoire contient des fichiers .txt (une profession de foi = 1 fichier)
    - Un fichier CSV de métadonnées :
        https://minio.lab.sspcloud.fr/sim2023/mlfornlp/archelec_metadata.csv
      avec les colonnes : "titulaire-nom", "titulaire-soutien"

COMPRENDRE LES PROJECTIONS 2D :

  ┌─ BERTopic + UMAP ───────────────────────────────────────────────────────┐
  │  BERTopic :                                                             │
  │    • Utilise des embeddings SBERT (transformers multilingues).          │
  │    • Capture le sens contextuel, pas juste la fréquence des mots.       │
  │    • Cluster les documents sémantiquement proches → thèmes.             │
  │                                                                         │
  │  UMAP (Uniform Manifold Approximation and Projection) :                 │
  │    • Alternative à t-SNE, plus rapide et plus stable.                   │
  │    • Préserve MIEUX la structure globale que t-SNE.                     │
  │    • ✅  Les distances inter-clusters ont plus de sens qu'en t-SNE.     │
  │    • ✅  Reproductible (random_state fixé).                             │
  │    • Paramètre clé : n_neighbors (structure locale vs globale).         │
  │                                                                         │
  │  LECTURE DU GRAPHIQUE :                                                 │
  │    • Points proches = discours thématiquement similaires.               │
  │    • Clusters séparés = familles politiques avec vocabulaire distinct.  │
  │    • Mélange de partis dans un cluster = discours convergents.          │
  └─────────────────────────────────────────────────────────────────────────┘

UTILISATION :
    # Analyser une seule année :
    python semantic_mapping_multiyear.py --annees 1981

    # Analyser plusieurs années :
    python semantic_mapping_multiyear.py --annees 1981 1988 1993 2002

    # Choisir la méthode :
    python semantic_mapping_multiyear.py --annees 1981 --methode bertopic_umap

    # Comparer toutes les années avec comparaison temporelle :
    python semantic_mapping_multiyear.py --annees 1981 1988 1993 --comparaison

=============================================================================
"""

import argparse
import glob
import os
import warnings
import requests
import zipfile
import io
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
import umap
import spacy
from hdbscan import HDBSCAN
import string
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd


warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

URL_METADATA = "https://minio.lab.sspcloud.fr/sim2023/mlfornlp/archelec_metadata.csv"
REPERTOIRE_BASE = "."   # Répertoire racine contenant les dossiers legislatives_ANNEE/
MIN_CANDIDATS_PAR_PARTI = 30  # Seuil minimal pour retenir un parti

# Couleurs cohérentes par parti politique (clé = sous-chaîne du nom du parti)
PALETTE_PARTIS = {
    "PCF":        "#e63946",   # Rouge
    "PS":         "#f4a261",   # Rose-orange
    "RPR":        "#457b9d",   # Bleu gaullist
    "UDF":        "#1d3557",   # Bleu foncé
    "FN":         "#2d6a4f",   # Vert sombre
    "MRG":        "#e9c46a",   # Jaune
    "Verts":      "#52b788",   # Vert
    "LO":         "#9d0208",   # Rouge foncé
    "LFI":        "#ef233c",   # Rouge vif
    "En Marche":  "#4cc9f0",   # Bleu ciel
    "default":    "#adb5bd",   # Gris neutre
}

# --- GESTION GLOBALE DES COULEURS ---
GLOBAL_COLOR_MAP = {}
PALETTE_DYNAMIQUE = itertools.cycle(sns.color_palette("tab20", 20))

def obtenir_couleur_parti(nom_parti):
    if nom_parti in GLOBAL_COLOR_MAP:
        return GLOBAL_COLOR_MAP[nom_parti]

    for cle, couleur in PALETTE_PARTIS.items():
        if cle != "default" and cle.lower() in nom_parti.lower():
            GLOBAL_COLOR_MAP[nom_parti] = couleur
            return couleur

    nouvelle_couleur = next(PALETTE_DYNAMIQUE)
    GLOBAL_COLOR_MAP[nom_parti] = nouvelle_couleur
    return nouvelle_couleur

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES RESSOURCES NLP
# ─────────────────────────────────────────────────────────────────────────────

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    from spacy.cli import download as spacy_download
    spacy_download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

STOP_WORDS_FR = stopwords.words("french")
STOP_WORDS_DE = stopwords.words("german")

# Fusion des deux listes
STOP_WORDS_FR = STOP_WORDS_FR + STOP_WORDS_DE
MOTS_SUPPLEMENTAIRES = [
    "madame", "mademoiselle", "monsieur", "candidat", "élections", "suffrages",
    "député", "circonscription", "votez", "cevipof", "cevipov", "sciences", "po",
    "fonds", "suppléant", "législatives", "non", "libertés", "juin", "die", "und",
    "der", "ni", "développement", "tout", "toute", "toutes", "tous",
    "si", "für", "in", "den", "sie", "eine", "certes", "zu", "vers", "leurs",
    "no", "wir", "iiie", "nale", "faut", "arlette", "laguiller", "ich", "plus",
    "vie", "dé", "dem", "gilly", "das", "cette", "ce", "cet", "ans", "allons",
    "faire", "françois", "majorité", "ensemble", "14","10","1","2","3","france"," France", "donnons",
    "elections","législatives","tour","candidats","suppléants","maire","conseiller",
    "président","parti","comme","contre","fait","depuis","","être","falloir","vouloir",
    "mettre","donner","werden","auf","nicht","einer","dass","gegen","ihr","auch","mit",
    "von","ist","ein","sich","wird","haben","durch","ihre","als","frankreich","leben",
    "sind","mehr","einen","politik","mehrheit","hat","geben","juni","alsace","strasbourg",
    "mai","juin","mars","avril","juillet","septembre","national","front","vote",
    "nationale","mitterrand","facon","jean","gemeinsam","écologistes", "kiffer","rouen",
    "marseille","agira","neuf","1981","sans","entre","aujourd","hui","21","place","vu","paul","ceux","michel",
    "joseph","où","montivillie","fiszbin","lefort","vont","gauche","elles","1988","oui","allard","rouennais","changement","nouvelle","bien","voter","ecologie","communistes","gérard","lecanuet","général",
    "trop", "alors", "chambre", "côté","lalonde", "brice","gaudin","jusqu", "ici","12","anne","seine", "denis", "socialiste", "saint", "15", "claude", "neuilly", "neiertz", "véronique","calais", "allez",
     "poussy", "roland", "weisenhorn","marie","aussi","emile", "élisant", "patrie", "jarosz", "abord","wählt", "wähler", "jemenin", "am" , "deren" , "einwanderung" , "einsetzt" , "vertrauen" , "um" , "pensciences",
     "mrg", "baugeois",
    "herbiers" , "bourdier",  "briand", "vendée" , "bonnet","udf","patrice",
    "paris", "quilès" , "xiiie" , "quiles", "000","patrick","bonvoisin","jeanine","rpr","udf",'ouvrière',
    "Parti", "communiste", "français" , "Union" ,"pour" ,"la", "nouvelle" ,"République","Union" "démocratique", "du","Front" "travailliste","Républicains","indépendants" , "Fédération", "gauche" ,"démocrate","socialiste",
    "franzosen" , "uberzeugungen" , "wiederaufrichtung", "abschaffung", "überzeugungen", "kandidaten", "steuer", "schaffung",
    "binctin" , "boulen" , "canu" , "carrale" , "vieuxmaire", "pierrain",
    "odru", "bartolone", "ainsi", "mahéas","roatta",
    "sial", "50000", "69100", "villeurbannesciences", "jus", "npc", "32", "bruno", "aller" , "vorzug" , "dies"  ,  "stimmen" , "willen",
    "mulhouse" , "illzach", "marc", "schittly", "wittenheim",
    "nungesser", "conseil", "koehl", "guy", "nogent", "marne","électrices","suppleant","brive","parce","enfants"

]
STOP_WORDS_FR.extend(MOTS_SUPPLEMENTAIRES)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES : EXTRACTION NOM / PARTI (OPTIMISÉ)
# ─────────────────────────────────────────────────────────────────────────────
def charger_donnees_legislatives(url_zip, annee, extraction_path="."):
    if os.path.exists(os.path.join(extraction_path, f"legislatives_{annee}")):
        print(f"Les données {annee} semblent déjà présentes. Saut du téléchargement.")
        return

    print(f"Téléchargement de l'archive : {url_zip}...")
    try:
        r = requests.get(url_zip, stream=True)
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            print(f"Extraction des fichiers dans '{os.path.abspath(extraction_path)}'...")
            z.extractall(path=extraction_path)
            print("Extraction terminée.")
    except Exception as e:
        print(f"Erreur lors du traitement du ZIP : {e}")


def charger_metadata():
    print("Chargement des métadonnées...")
    df = pd.read_csv(URL_METADATA, low_memory=False)
    # Remplacer les valeurs nulles pour éviter les erreurs lors de la création du dictionnaire
    df["titulaire-nom"] = df["titulaire-nom"].fillna("")
    df["titulaire-soutien"] = df["titulaire-soutien"].fillna("Inconnu")
    return df[["titulaire-soutien", "titulaire-nom","id"]].copy()


def construire_url_to_parti(liste_fichiers, df_meta):
    """
    Version hautement optimisée avec SpaCy en batching et dictionnaire de métadonnées.
    """
    donnees = []
    # 1. Création d'un dictionnaire O(1) pour remplacer le "df_meta == nom" ultra-lent
    print("  Indexation des métadonnées...")
    dict_meta = {}
    for nom, parti in zip(df_meta["titulaire-nom"], df_meta["titulaire-soutien"]):
        nom = str(nom).strip().capitalize()
        if nom not in dict_meta:
            dict_meta[nom] = []
        if parti not in dict_meta[nom]:
            dict_meta[nom].append(parti)
    # 2. Lecture de tous les textes en mémoire pour le batching SpaCy
    print(f"  Lecture des {len(liste_fichiers)} fichiers...")
    textes = []
    for chemin in liste_fichiers:
        with open(chemin, "r", encoding="utf-8") as f:
            textes.append(f.read())
    # 3. Traitement NLP par lot (désactivation des modules inutiles pour x10 en vitesse)
    print("  Extraction des entités nommées (Optimisation NLP active)...")
    ponctuation = set(string.punctuation)
    # nlp.pipe est incroyablement plus rapide que de faire nlp(texte) dans une boucle
    pipes = nlp.pipe(textes, batch_size=100, disable=["parser", "tagger", "attribute_ruler", "lemmatizer"])
    for chemin, doc in zip(liste_fichiers, pipes):
        noms_uniques = set()
        for ent in doc.ents:
            if ent.label_ == "PER":
                parties = ent.text.strip().split()
                if not parties: continue
                nom_de_famille = parties[-1].strip()
                if nom_de_famille != "MITTERRAND":
                    if nom_de_famille.isupper() and not any(c in ponctuation for c in nom_de_famille):
                        noms_uniques.add(nom_de_famille)
        # 4. Association rapide au parti via le dictionnaire
        noms_filtres, partis_filtres = [], []
        for nom in list(noms_uniques):
            nom_cap = nom.capitalize()
            partis_trouves = dict_meta.get(nom_cap)
            if partis_trouves and partis_trouves[0] != "Inconnu":
                parti_final = partis_trouves[0] if len(partis_trouves) == 1 else partis_trouves
                noms_filtres.append(nom)
                partis_filtres.append(parti_final)
        nettoyage = []
        for p in partis_filtres:
            nettoyage.append(" / ".join(p) if isinstance(p, list) else str(p))
        donnees.append({
            "url": chemin,
            "nom": " | ".join(noms_filtres),
            "parti": " | ".join(nettoyage),
        })
    return pd.DataFrame(donnees)

# ─────────────────────────────────────────────────────────────────────────────
# PRÉPARATION DES DONNÉES POUR UNE ANNÉE
# ─────────────────────────────────────────────────────────────────────────────

def charger_annee(annee, df_meta, cache_csv=True):
    dossier = os.path.join(REPERTOIRE_BASE, f"legislatives_{annee}")
    if not os.path.isdir(dossier):
        raise FileNotFoundError(f"Répertoire introuvable : {dossier}")

    cache_path = f"url_to_nom_and_parti_{annee}.csv"

    if cache_csv and os.path.exists(cache_path):
        print(f"[{annee}] Chargement depuis le cache : {cache_path}")
        df_url = pd.read_csv(cache_path)
    else:
        print(f"[{annee}] Extraction des noms/partis depuis {dossier}...")
        liste_fichiers = glob.glob(os.path.join(dossier, "*.txt"))
        if not liste_fichiers:
            raise ValueError(f"Aucun fichier .txt dans {dossier}")
        df_url = construire_url_to_parti(liste_fichiers, df_meta)
        if cache_csv:
            df_url.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"[{annee}] Sauvegardé dans {cache_path}")

    # Éclater les partis multiples (séparés par | ou /)
    df_eclate = df_url.assign(
        parti=df_url["parti"].str.split(r"\s*[|/]\s*", regex=True)
    ).explode("parti").reset_index(drop=True)
    df_eclate["parti"] = df_eclate["parti"].str.strip()

    # Filtrer les partis valides
    df_valide = df_eclate[
        df_eclate["parti"].notna()
        & (df_eclate["parti"] != "")
        & (df_eclate["parti"] != "Inconnu")
        & (df_eclate["parti"] != "non mentionné")
    ]

    # Garder seulement les partis majeurs
    counts = df_valide["parti"].value_counts()
    partis_majeurs = counts[counts >= MIN_CANDIDATS_PAR_PARTI].index
    df_filtre = df_valide[df_valide["parti"].isin(partis_majeurs)].copy()

    print(f"[{annee}] {len(df_filtre)} entrées / {df_filtre['parti'].nunique()} partis majeurs.")
    return df_filtre


# ─────────────────────────────────────────────────────────────────────────────
# MÉTHODE 2 : BERTopic + UMAP (+ SAUVEGARDE MODÈLE)
# ─────────────────────────────────────────────────────────────────────────────

def preparer_donnees_bertopic_umap(df_metadata, annee):
    documents, noms_candidats, partis = _lire_documents(df_metadata)
    if not documents:
        print("no doc")
        return None, None, None, []

    # 1. OPTIMISATION : MISE EN CACHE DES EMBEDDINGS
    fichier_cache = f"embeddings_cache_{annee}.npy"
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    if os.path.exists(fichier_cache):
        print(f"Chargement instantané des embeddings depuis le cache ({fichier_cache})")
        embeddings = np.load(fichier_cache)
    else:
        embeddings = embedding_model.encode(documents, show_progress_bar=True, batch_size=32)
        np.save(fichier_cache, embeddings)

    meilleur_n = optimiser_nombre_topics(documents, embeddings, annee, range_topics=[6,7,8,10])
    vectorizer_model = CountVectorizer(stop_words=STOP_WORDS_FR)
    cluster_model = HDBSCAN(min_cluster_size=20, min_samples=10, prediction_data=True)

    print(f" Lancement de BERTopic ({meilleur_n} thèmes)...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        nr_topics=meilleur_n,
        calculate_probabilities=True,
        verbose=True,
        hdbscan_model=cluster_model
    )

    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

    # --- NOUVEAUTÉ : EXTRACTION ET SAUVEGARDE DES MOTS + PROBAS ---
    print(" Extraction des mots-clés et scores pour chaque thème...")

    all_topics_data = topic_model.get_topics()
    topic_info = topic_model.get_topic_info()

    mots_themes_list = []
    for topic_id, words_scores in all_topics_data.items():
        # On crée une chaîne formatée : "mot (0.045), mot2 (0.038)..."
        # On arrondit à 4 décimales pour la lisibilité
        mots_avec_scores = [f"{word} ({round(score, 4)})" for word, score in words_scores]

        # Récupération du nom du thème de manière sécurisée
        nom_theme = topic_info[topic_info.Topic == topic_id].Name.values[0]

        mots_themes_list.append({
            "Topic_ID": topic_id,
            "Annee": annee,
            "Nom_du_Theme": nom_theme,
            "Mots_et_Scores": ", ".join(mots_avec_scores)
        })

    # Sauvegarde en CSV
    df_mots_themes = pd.DataFrame(mots_themes_list)
    csv_filename = f"mots_cles_themes_{annee}.csv"
    df_mots_themes.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"Liste enrichie sauvegardée dans : {csv_filename}")
    # -----------------------------------------------------------------------

    # Sauvegarde du modèle
    save_dir = f"modele_bertopic_{annee}"
    print(f"Sauvegarde du modèle BERTopic dans le dossier : {save_dir}/")
    topic_model.save(save_dir, serialization="safetensors", save_ctfidf=True)

    # 3. GESTION ADAPTÉE DES SCORES
    if probs is not None:
        if len(probs.shape) == 2:
            score_thematique = np.max(probs, axis=1)
        else:
            score_thematique = probs
    else:
        score_thematique = np.ones(len(documents))

    mapping_themes = {
        row["Topic"]: f"T{row['Topic']}: {row['Name'].split('_', 1)[-1][:30]}"
        for _, row in topic_model.get_topic_info().iterrows()
    }

    # Projection UMAP
    reductor = umap.UMAP(n_neighbors=15, n_components=2, random_state=42, metric="cosine")
    coords = reductor.fit_transform(embeddings)

    df_plot = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "Candidat": noms_candidats,
        "Parti": partis,
        "Theme_Dominant": [mapping_themes.get(t, "Hors-thème") for t in topics],
        "Score": score_thematique
    })

    # Nettoyage et limitation pour l'affichage
    df_plot = df_plot.sort_values("Score", ascending=False)
    df_plot = df_plot.groupby("Parti").head(MAX_CANDIDATS_AFFICHÉS_PAR_PARTI).reset_index(drop=True)

    print(f"Traitement terminé. Points affichés : {len(df_plot)}")
    return df_plot, "Dimension UMAP 1", "Dimension UMAP 2", list(mapping_themes.values())


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE INTERNE : LECTURE DES FICHIERS
# ─────────────────────────────────────────────────────────────────────────────


def _lire_documents(df_metadata):
    documents, noms_candidats, partis = [], [], []
    for _, row in df_metadata.iterrows():
        try:
            with open(row["url"], "r", encoding="utf-8") as f:
                documents.append(f.read())
            nom_brut = str(row["nom"])
            noms_candidats.append(nom_brut.split("|")[0].strip())
            partis.append(str(row["parti"]).strip())
        except FileNotFoundError:
            continue
    return documents, noms_candidats, partis


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION : CARTE SÉMANTIQUE PAR ANNÉE
# ─────────────────────────────────────────────────────────────────────────────

def tracer_carte_semantique(df_plot, xlabel, ylabel, output_file, titre, noms_themes=None):
    if df_plot is None or df_plot.empty:
        print("Aucune donnée à tracer.")
        return

    df_plot["Parti_court"] = df_plot["Parti"].apply(
        lambda x: x[:28] + ".." if len(str(x)) > 28 else x
    )
    

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])

    partis_uniques = df_plot["Parti_court"].unique()
    color_map = {p: obtenir_couleur_parti(p) for p in partis_uniques}
    themes_uniques = df_plot["Theme_Dominant"].unique()
    df_plot=df_plot[df_plot['Theme_Dominant'] != "-1"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]
    marker_map = {t: markers[i % len(markers)] for i, t in enumerate(themes_uniques)}

    for parti in partis_uniques:
        for theme in themes_uniques:
            sub = df_plot[(df_plot["Parti_court"] == parti) & (df_plot["Theme_Dominant"] == theme)]
            if sub.empty:
                continue
            ax.scatter(
                sub["x"], sub["y"],
                c=[color_map[parti]], marker=marker_map[theme],
                s=80, alpha=0.75, edgecolors="w", linewidths=0.4,
                label=parti,
            )

    for parti in df_plot["Parti_court"].unique():
        sub_parti = df_plot[df_plot["Parti_court"] == parti].nlargest(3, "Score")
        for _, row in sub_parti.iterrows():
            ax.annotate(
                row["Candidat"][:12],
                (row["x"], row["y"]),
                fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.1),
                zorder=10
            )

    handles_partis = [mpatches.Patch(color=color_map[p], label=p) for p in partis_uniques]
    leg1 = ax.legend(
        handles=handles_partis, title="Partis", fontsize=8,
        loc="lower right", framealpha=0.85, title_fontsize=9,
    )
    ax.add_artist(leg1)

    from matplotlib.lines import Line2D
    handles_themes = [
        Line2D([0], [0], marker=marker_map[t], color="gray",
               markersize=8, linestyle="None", label=t[:40])
        for t in themes_uniques
    ]
    ax.legend(
        handles=handles_themes, title="Thèmes dominants", fontsize=7,
        loc="lower left", framealpha=0.85, title_fontsize=9,
    )

    ax.set_title(titre, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.grid(False)

    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")

    methode = "UMAP" if "UMAP" in xlabel else "t-SNE"
    algo_theme = "BERTopic" if "UMAP" in xlabel else "NMF"

    texte_explication = (
        f"LECTURE DU GRAPHIQUE\n"
        f"{'─'*34}\n\n"
        f"Chaque point = 1 document\n"
        f"(profession de foi d'un candidat)\n\n"
        f"COULEUR → Parti politique\n"
        f"FORME → Thème dominant\n\n"
        f"Deux points proches =\n"
        f"discours thématiquement\n"
        f"similaires.\n\n"
        f"MÉTHODE : {algo_theme} + {methode}\n"
        f"{'─'*34}\n\n"
    )

    if noms_themes:
        # On s'assure de ne pas afficher de thèmes "hors-sujet" (souvent mappés sur -1)
        # On filtre la liste pour exclure toute mention de "T-1" ou "Outliers"
        themes_propres = [t for t in noms_themes if "T-1" not in t and "Outliers" not in t]
        
        texte_explication += "\n\n" + "─"*34 + "\nTHÈMES DÉTECTÉS\n\n"
        
        # On affiche les 5 premiers thèmes valides (index 0 à 5)
        for t in themes_propres:
            texte_explication += f"• {t}\n"

    ax_info.text(
        0.05, 0.95, texte_explication,
        transform=ax_info.transAxes,
        fontsize=8.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(facecolor="#f8f9fa", alpha=0.9, edgecolor="#dee2e6", pad=10),
    )

    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"  Carte sauvegardée : {output_file}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION : COMPARAISON TEMPORELLE DES THÈMES
# ─────────────────────────────────────────────────────────────────────────────

def tracer_comparaison_temporelle(resultats_par_annee, output_file="comparaison_temporelle.png"):
    annees = sorted(resultats_par_annee.keys())
    n_annees = len(annees)

    if n_annees == 0:
        print("Aucune donnée pour la comparaison temporelle.")
        return

    fig = plt.figure(figsize=(7 * n_annees, 14))
    fig.suptitle(
        "Évolution des thèmes et des partis aux élections législatives",
        fontsize=18, fontweight="bold", y=0.98,
    )

    gs_top = gridspec.GridSpec(1, n_annees, top=0.88, bottom=0.52, hspace=0.4, wspace=0.35)
    gs_bot = gridspec.GridSpec(1, 1, top=0.46, bottom=0.06)

    for col, annee in enumerate(annees):
        data = resultats_par_annee[annee]
        df = data["df"]
        if df is None or df.empty:
            continue

        ax = fig.add_subplot(gs_top[col])
        partis_uniques = df["Parti"].unique()
        color_map = {p: obtenir_couleur_parti(p) for p in partis_uniques}

        for parti in partis_uniques:
            sub = df[df["Parti"] == parti]
            ax.scatter(
                sub["x"], sub["y"],
                c=[color_map[parti]], s=15, alpha=0.6,
                edgecolors="none", label=parti[:20],
            )

        ax.set_title(f"Législatives {annee}", fontsize=12, fontweight="bold")
        ax.set_xlabel(data.get("xlabel", "Dim 1"), fontsize=7)
        ax.set_ylabel(data.get("ylabel", "Dim 2"), fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, loc="upper right", framealpha=0.7, markerscale=1.5)
        sns.despine(ax=ax)

    ax_evo = fig.add_subplot(gs_bot[0])
    presence = {}
    for annee in annees:
        df = resultats_par_annee[annee]["df"]
        if df is None or df.empty:
            continue
        counts = df["Parti"].value_counts(normalize=True) * 100
        for parti, pct in counts.items():
            if parti not in presence:
                presence[parti] = {}
            presence[parti][annee] = pct

    totaux = {p: sum(v.values()) for p, v in presence.items()}
    top_partis = sorted(totaux, key=totaux.get, reverse=True)[:10]
    colors_evo = [obtenir_couleur_parti(p) for p in top_partis]
    
    for i, parti in enumerate(top_partis):
        xs = [a for a in annees if a in presence.get(parti, {})]
        ys = [presence[parti][a] for a in xs]
        if xs:
            ax_evo.plot(
                xs, ys, marker="o", linewidth=2.5,
                color=colors_evo[i], label=parti[:30],
            )
            ax_evo.annotate(
                f"{ys[-1]:.0f}%",
                (xs[-1], ys[-1]),
                textcoords="offset points", xytext=(5, 0),
                fontsize=7, color=colors_evo[i],
            )

    ax_evo.set_title(
        "Part des documents par parti (% des candidats majeurs) au fil des années",
        fontsize=12, fontweight="bold",
    )
    ax_evo.set_xlabel("Année des législatives", fontsize=10)
    ax_evo.set_ylabel("% de documents", fontsize=10)
    ax_evo.set_xticks(annees)
    ax_evo.legend(
        bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=8, title="Partis", title_fontsize=9,
    )
    ax_evo.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax_evo)

    plt.savefig(output_file, dpi=180, bbox_inches="tight")
    print(f"Comparaison temporelle sauvegardée : {output_file}")
    plt.close()



def optimiser_nombre_topics(documents, embeddings, annee, range_topics=[5, 10, 15, 20, 25, 30]):
    """
    Teste différentes valeurs de nr_topics et calcule le score de cohérence C_v.
    """
    print(f"Analyse de la cohérence pour les paliers : {range_topics}")
    
    # Préparation des données pour Gensim (tokenisation simple)
    texts = [doc.split() for doc in documents]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    coherence_scores = []

    for k in range_topics:
        print(f"  -> Test avec {k} thèmes...")
        
        # Initialisation du modèle
        model = BERTopic(nr_topics=k, verbose=False)
        topics, _ = model.fit_transform(documents, embeddings=embeddings)
        
        # --- CORRECTION ICI ---
        # On récupère tous les thèmes réellement créés par le modèle
        all_topics = model.get_topics() 
        
        topics_words = []
        for topic_id, words_scores in all_topics.items():
            # On ignore le topic -1 (outliers/bruit) car il nuit au calcul de cohérence
            if topic_id != -1:
                # On extrait les mots du thème
                words = [word for word, score in words_scores if word != ""]
                if words:
                    topics_words.append(words)
        # -----------------------

        # Calcul du score de cohérence C_v
        if len(topics_words) > 0:
            cm = CoherenceModel(topics=topics_words, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_scores.append(cm.get_coherence())
        else:
            coherence_scores.append(0)

    # Affichage du graphique "Méthode du coude"
    plt.figure(figsize=(10, 6))
    plt.plot(range_topics, coherence_scores, marker='o', linestyle='-', color='b')
    plt.title(f"Évolution de la cohérence en fonction du nombre de thèmes pour l'année {annee}")
    plt.xlabel("Nombre de thèmes (nr_topics)")
    plt.ylabel("Score de Cohérence $C_v$")
    plt.grid(True)
    plt.savefig(f"blabla{annee}", dpi=200, bbox_inches="tight")
    plt.show()

    best_k = range_topics[coherence_scores.index(max(coherence_scores))]
    print(f" ✅ Le score de cohérence maximal est de {max(coherence_scores):.3f} pour {best_k} thèmes.")
    
    return best_k
# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


def generer_statistiques_descriptives(df_metadata, annee):
    print(f"📊 Génération des statistiques descriptives pour {annee}...")

    # 1. Préparation des données
    documents, _, partis = _lire_documents(df_metadata)
    df_stats = pd.DataFrame({
        "texte": documents,
        "parti": partis,
        "longueur": [len(str(t).split()) for t in documents]
    })

    # --- MODIFICATION : Tronquer les noms des partis à 50 caractères ---
    df_stats["parti"] = df_stats["parti"].apply(
        lambda x: str(x)[:50] + ".." if len(str(x)) > 50 else str(x)
    )

    # Création d'une figure multi-stats
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # --- A. Histogramme des partis politiques ---
    ax1 = fig.add_subplot(gs[0, 0])
    # On récupère les couleurs via le dictionnaire global
    partis_presents = df_stats["parti"].unique()
    palette_couleurs = [obtenir_couleur_parti(p) for p in partis_presents]

    sns.countplot(data=df_stats, y="parti", ax=ax1, palette=palette_couleurs)
    ax1.set_title(f"Répartition des candidats par parti ({annee})", fontweight="bold", fontsize=14)
    ax1.set_xlabel("Nombre de documents")
    ax1.set_ylabel("")

    # --- B. Longueur moyenne des textes par parti ---
    ax2 = fig.add_subplot(gs[0, 1])
    longueur_moyenne = df_stats.groupby("parti")["longueur"].mean().sort_values()
    longueur_moyenne.plot(kind="barh", ax=ax2, color="#457b9d")
    ax2.set_title("Longueur moyenne des textes (nb mots) par parti", fontweight="bold", fontsize=14)
    ax2.set_xlabel("Nombre de mots moyen")
    ax2.set_ylabel("")

    # --- C. Global Bag of Words ---
    ax3 = fig.add_subplot(gs[1, 0])
    vectorizer = CountVectorizer(stop_words=STOP_WORDS_FR, max_features=20)
    word_counts = vectorizer.fit_transform(df_stats["texte"])
    sum_words = word_counts.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    df_bow = pd.DataFrame(words_freq, columns=["mot", "frequence"])
    sns.barplot(data=df_bow, x="frequence", y="mot", ax=ax3, color="#2a9d8f")
    ax3.set_title("Bag of Words : Top 20 mots (global)", fontweight="bold", fontsize=14)

    # --- D. Mots les plus cités par parti (Top 3 à 5 mots) ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    texte_top_mots = f"TOP MOTS PAR PARTI ({annee})\n" + "─"*70 + "\n"

    for parti in df_stats["parti"].unique():
        textes_parti = df_stats[df_stats["parti"] == parti]["texte"]
        # Vectorisation par parti
        vec_p = CountVectorizer(stop_words=STOP_WORDS_FR, max_features=5)
        try:
            vec_p.fit(textes_parti)
            top_mots = vec_p.get_feature_names_out()
            # Formatage propre : le nom du parti est déjà limité à 50
            texte_top_mots += f"• {parti:<50} : {', '.join(top_mots)}\n"
        except:
            continue

    ax4.text(0.01, 0.98, texte_top_mots, transform=ax4.transAxes, 
             fontsize=9, family="monospace", verticalalignment="top",
             bbox=dict(facecolor="#f8f9fa", alpha=0.8, edgecolor="#dee2e6"))

    # Sauvegarde finale
    output_path = f"statistiques_descriptives_{annee}.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"✅ Tableau de bord statistique sauvegardé : {output_path}")
    plt.close()


def analyser_annee(annee, df_meta, methode="bertopic_umap"):
    print(f"\n{'='*60}")
    print(f"  ANALYSE : Législatives {annee}")
    print(f"{'='*60}")

    try:
        df_filtre = charger_annee(annee, df_meta)
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERREUR : {e}")
        return None

    # generation des stat desc
    

    # On passe "annee" pour la sauvegarde du modèle
    df_plot, xlabel, ylabel, themes = preparer_donnees_bertopic_umap(df_filtre, annee)

    if df_plot is None:
        print(f"  Aucun document exploitable pour {annee}.")
        return None

    output_file = f"carte_semantique_{annee}_{methode}.png"
    titre = f"Cartographie Sémantique — Législatives {annee} ({methode.upper()})"
    tracer_carte_semantique(df_plot, xlabel, ylabel, output_file, titre, themes)
    if df_plot is not None:
        generer_graphique_distribution(df_plot, themes, annee)
    generer_statistiques_descriptives(df_filtre, annee)
    return {"df": df_plot, "xlabel": xlabel, "ylabel": ylabel, "themes": themes}


def generer_graphique_distribution(df, theme_labels, annee):
    """
    Génère et sauvegarde le graphique de distribution des thèmes par parti.
    Exclut le thème de bruit -1 (T-1).
    """
    # 1. Préparation des données
    # On retire "Hors-thème" ET on filtre pour exclure explicitement le thème -1
    # On s'assure de ne garder que les lignes où Theme_Dominant ne contient pas "T-1"
    df_clean = df[
        (df['Theme_Dominant'] != "Hors-thème") &
        (~df['Theme_Dominant'].str.startswith("T-1 :", na=False)) &
        (df['Theme_Dominant'] != "-1")
    ].copy()
    
    # 2. Création du tableau croisé
    # si tu as plus de 50 lignes dans ton DF. Utilise plutôt df_clean['Parti'].
    df_clean['Parti'] = df_clean['Parti'].astype(str).str[:50]
    tableau_croise = pd.crosstab(df_clean['Theme_Dominant'], df_clean['Parti'])

    # 3. Réorganisation selon l'ordre fourni (en excluant T-1 de la liste d'ordre aussi)
    order_themes = [t for t in theme_labels if t in tableau_croise.index and not t.startswith("T-1")]
    tableau_croise = tableau_croise.reindex(order_themes)

    # 4. Configuration esthétique et tracé
    sns.set_style("ticks")
    ax = tableau_croise.plot(
        kind='barh', 
        stacked=True, 
        figsize=(12, 8), 
        colormap='tab20', 
        width=0.8, 
        edgecolor='white',
        linewidth=0.5
    )

    # 5. Personnalisation des titres et axes
    plt.title(f"Distribution des Thèmes par Parti Politique - Législatives {annee}", 
              fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Nombre de candidats (Top scores par parti)", fontsize=11)
    plt.ylabel("")
    plt.gca().invert_yaxis() 

    # 6. Légende et finitions
    plt.legend(title="Partis Politiques", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    sns.despine()

    # 7. Sauvegarde
    plt.tight_layout()
    filename = f"distribution_themes_{annee}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f" 📊 Graphique de distribution sauvegardé : {filename}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Cartographie sémantique des programmes législatifs (multi-années)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--annees", nargs="+", type=int, default=[#'1973, 1978, 
        1981, 1988])
    parser.add_argument("--methode", choices=["nmf", "bertopic_umap"], default="bertopic_umap")
    parser.add_argument("--comparaison", action="store_true")
    parser.add_argument("--repertoire", default=".")
    parser.add_argument("--max-candidats-affiches-par-parti", type=int, default=20)
    parser.add_argument("--min-candidats", type=int, default=30)
    args = parser.parse_args()

    global REPERTOIRE_BASE, MIN_CANDIDATS_PAR_PARTI, MAX_CANDIDATS_AFFICHÉS_PAR_PARTI, COMPARAISON
    REPERTOIRE_BASE = args.repertoire
    MIN_CANDIDATS_PAR_PARTI = args.min_candidats
    MAX_CANDIDATS_AFFICHÉS_PAR_PARTI = args.max_candidats_affiches_par_parti
    COMPARAISON = args.comparaison

    charger_donnees_legislatives("https://minio.lab.sspcloud.fr/sim2023/mlfornlp/legislatives_1981.zip", 1981, extraction_path=".")
    charger_donnees_legislatives("https://minio.lab.sspcloud.fr/sim2023/mlfornlp/legislatives_1988.zip", 1988, extraction_path=".")
    charger_donnees_legislatives("https://minio.lab.sspcloud.fr/sim2023/mlfornlp/legislatives_1978.zip", 1978, extraction_path=".")
    charger_donnees_legislatives("https://minio.lab.sspcloud.fr/sim2023/mlfornlp/legislatives_1973.zip", 1973, extraction_path=".")
    df_meta = charger_metadata()
    resultats = {}

    for annee in args.annees:
        cache_path = f"url_to_nom_and_parti_{annee}.csv"
        if os.path.exists(cache_path):
            print(f"Fichier cache trouvé pour {annee}. Chargement de : {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            print(f"Pas de cache pour {annee}. Lancement de l'extraction des fichiers...")
            dossier_path = f"./legislatives_{annee}"
            if not os.path.exists(dossier_path):
                print(f"Erreur : Le dossier {dossier_path} est introuvable !")
                df = pd.DataFrame()
            else:
                liste_fichiers = glob.glob(os.path.join(dossier_path, "*.txt"))
                df = construire_url_to_parti(liste_fichiers, df_meta)
                df.to_csv(cache_path, index=False, encoding='utf-8-sig')
                print(f"Cache créé : {cache_path}")

        res = analyser_annee(annee, df_meta, methode=args.methode)
        if res is not None:
            resultats[annee] = res

    if args.comparaison and len(resultats) > 1:
        print("\nGénération de la comparaison temporelle...")
        tracer_comparaison_temporelle(
            resultats,
            output_file=f"comparaison_{'_'.join(str(a) for a in sorted(resultats.keys()))}.png",
        )
    elif args.comparaison and len(resultats) <= 1:
        print("⚠  La comparaison temporelle nécessite au moins 2 années avec des données.")

    print("\n✓ Analyse terminée.")


if __name__ == "__main__":
    main()
