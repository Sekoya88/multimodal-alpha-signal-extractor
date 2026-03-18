# 🚀 Déployer sur Hugging Face Spaces (Gratuitement)

Pour déployer ton dashboard Streamlit **Alpha-Signal Extractor** gratuitement sur Hugging Face Spaces, suis ces étapes simples.

## ⚠️ Avertissement sur les limites (Free Tier)
Les *Spaces gratuits* de Hugging Face offrent : **16 Go de RAM** et **2 vCPU**.
- Faire tourner simultanément **Ollama (LLaMA 3 8B ~4.7Go RAM)** + **llama.cpp (Qwen2.5-VL 3B ~2Go RAM)** va prendre ~7Go de RAM minimum au repos. C'est jouable mais l'inférence texte + vision va être **très lente** sur un pauvre dual-core CPU (attends-toi à ~1-2 tokens / sec et potentiellement un crash OOM si les contextes explosent).
- Si tu veux que ça soit plus fluide sur le Free Tier, change le backend de sentiment (de `llama3:8b` vers `qwen2:0.5b`) dans le fichier `config.py`.

---

## 🏗️ Étape par Étape

### 1. Créer le Space
1. Va sur [Hugging Face Spaces](https://huggingface.co/spaces) et connecte-toi.
2. Clique sur **Create new Space**.
3. **Space name**: `alpha-signal-extractor`
4. **License**: `mit` (ou autre).
5. **Select the Space SDK**: Choisis **Docker** 🐳 (Pas Streamlit, car on a besoin d'installer Ollama et compiler `llama-cpp-python`).
6. **Docker Template**: Laisse vide (Blank).
7. **Space Hardware**: Free.
8. Clique sur **Create Space**.

### 2. Pousser le Code source (via Git)

Copie l'URL de ton repo Hugging Face Space (ex: `https://huggingface.co/spaces/TON_PSEUDO/alpha-signal-extractor`).

Dans ton terminal, ajoute ce repo comme "remote" :
```bash
git remote add hf https://huggingface.co/spaces/TON_PSEUDO/alpha-signal-extractor
```

Puis force-push ta branche de dev vers HF :
```bash
git push hf dev:main
```

### 3. Gérer ton Modèle GGUF Fine-tuné (Crucial)

Vu que ton VLM (`alpha-signal-q4km.gguf`) pèse lourd, Github l'a probablement ignoré, et il n'a pas été envoyé à HF Spaces.
Il y a 2 façons de l'injecter sur le HF Space :

**Option A (La plus simple) : L'uploader via l'interface web**
1. Va sur ton HF Space, onglet **Files**.
2. Clique sur **Add File > Upload File**.
3. Dépose ton modèle `alpha-signal-q4km.gguf` à la racine.
4. Clique sur "Commit changes to main".

**Option B (Via le script Entrypoint)**
1. Modifie le fichier `scripts/entrypoint.sh` de ce projet.
2. Décommente la ligne avec `wget` et mets le vrai lien de téléchargement direct de ton GGUF si tu l'héberges ailleurs sur un repo HF.

### 4. Patienter (Le build Docker va commencer)
Hugging Face va détecter ton `Dockerfile` et compiler l'environnement (ça peut prendre 5-10 minutes la première fois car ça compile `llama-cpp-python` depuis les sources).
Une fois le build terminé, le Space passera au statut **Running**. Ton URL publique sera prête ! 🚀
