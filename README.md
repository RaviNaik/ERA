# Extensive Reimagining of AI (ERA) Course Archive

Welcome to the ERA course repository — a central hub for all versions and sessions of the ERA curriculum.

**🌐 Live site: [https://ravinaik.github.io/ERA/](https://ravinaik.github.io/ERA/)**

Each session is a fully self-contained, browser-based interactive web app. No server required — all model training runs client-side.

---

## Curriculum Index

| Version | Session | Description | Live Demo |
| :--- | :--- | :--- | :--- |
| **v5** | **Session 1** — Deep Learning Foundations | Four interactive proofs: why activations matter, why depth without nonlinearity is a lie, how embeddings learn similarity from co-occurrence, and memorization vs generalization. Uses TensorFlow.js. | [Open →](https://ravinaik.github.io/ERA/v5/session1/) |
| **v5** | **Session 2** — Multilingual BPE Tokenizer | Design a 10,000-token BPE vocabulary across English, Hindi, Telugu and Kannada Wikipedia pages. Minimize fertility ratio spread to maximize score. Interactive experiment dashboard. | [Open →](https://ravinaik.github.io/ERA/v5/session2/) |

*(Additional sessions will be appended as they are developed and released.)*

---

## Repository Structure

```
ERA/
├── index.html              ← Root landing page (GitHub Pages entry point)
├── style.css               ← Landing page styles
├── .github/
│   └── workflows/
│       └── deploy.yml      ← Auto-deploy to GitHub Pages on push to main
└── v5/
    ├── session1/
    │   ├── index.html      ← Redirect shim → webapp/
    │   ├── webapp/         ← All web app files (HTML, CSS, JS)
    │   └── README.md
    └── session2/
        ├── index.html      ← Redirect shim → webapp/
        ├── webapp/         ← All web app files
        ├── bpe_assignment/ ← Python scripts (non-web)
        ├── notebooks/
        └── README.md
```

**Convention**: all GitHub Pages web app code lives in `webapp/` inside each `sessionX/` folder. Python scripts, notebooks, and data files stay at the `sessionX/` level.

---

## GitHub Pages Setup (One-Time)

After pushing to `main` for the first time:

1. Go to **GitHub → ERA repo → Settings → Pages**
2. Under **Build and deployment → Source**, select **"GitHub Actions"**
3. The workflow in `.github/workflows/deploy.yml` will handle all future deploys automatically on every push to `main`

---

## Running Sessions Locally

Each `webapp/` folder is a pure static site. Run any local server:

```bash
# From the repo root
npx serve .

# Or from a specific session
cd v5/session1/webapp
python -m http.server 8000
```
