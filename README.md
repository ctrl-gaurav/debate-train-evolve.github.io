# Debate, Train, Evolve — Project Website

<div align="center">

[![EMNLP 2025](https://img.shields.io/badge/EMNLP_2025-Main_Conference-brightgreen?style=for-the-badge)](https://aclanthology.org/2025.emnlp-main.1666/)
[![Paper](https://img.shields.io/badge/Paper-ACL_Anthology-blue?style=for-the-badge)](https://aclanthology.org/2025.emnlp-main.1666/)
[![Framework](https://img.shields.io/badge/Framework-GitHub-orange?style=for-the-badge&logo=github)](https://github.com/ctrl-gaurav/Debate-Train-Evolve)
[![Website](https://img.shields.io/badge/Website-Live-00a1f0?style=for-the-badge)](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/)

**Self-Evolution of Language Model Reasoning via Multi-Agent Debate Traces**

**[Gaurav Srivastava](mailto:gks@vt.edu)**\* &nbsp;&bull;&nbsp; **[Zhenyu Bi](mailto:zhenyub@vt.edu)** &nbsp;&bull;&nbsp; **[Meng Lu](mailto:menglu@vt.edu)** &nbsp;&bull;&nbsp; **[Xuan Wang](mailto:xuanw@vt.edu)**&dagger;

[![Virginia Tech](https://img.shields.io/badge/Virginia_Tech-CS_Department-861F41?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJjdXJyZW50Q29sb3IiLz4KPC9zdmc+)](https://cs.vt.edu/)
&nbsp;
[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-2a4dff?style=flat-square&logo=academia)](https://2025.emnlp.org/)
&nbsp;
[![ACL Anthology](https://img.shields.io/badge/ACL_Anthology-2025.emnlp--main.1666-red?style=flat-square)](https://aclanthology.org/2025.emnlp-main.1666/)

<sub>\* Lead Author &nbsp;&nbsp; &dagger; Corresponding Author</sub>

</div>

---

This repository contains the project website for the DTE framework paper, accepted at **EMNLP 2025 Main Conference**.

## Links

| Resource | URL |
|----------|-----|
| Paper | [aclanthology.org/2025.emnlp-main.1666](https://aclanthology.org/2025.emnlp-main.1666/) |
| Website | [ctrl-gaurav.github.io/debate-train-evolve.github.io](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/) |
| Framework Code | [github.com/ctrl-gaurav/Debate-Train-Evolve](https://github.com/ctrl-gaurav/Debate-Train-Evolve) |
| Old Website | [ctrl-gaurav.github.io/debate-train-evolve.github.io/old_ui](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/old_ui/) |

## Website Stack

- **React 19** + TypeScript + Vite
- **Tailwind CSS 4** with custom theme (Navy, Electric, Cyan, Neon palette)
- **Framer Motion** for animations
- Dark/light mode, particle backgrounds, glassmorphism

## Development

```bash
npm install
npm run dev      # Development server
npm run build    # Production build
npm run preview  # Preview production build
```

## Deployment

Deployed automatically via GitHub Actions on push to `main`. The workflow builds the site and deploys to GitHub Pages. The `old_ui/` folder is automatically copied to `dist/` during build.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with authors, key results, pipeline overview |
| `/method` | Framework methodology (RCR, GRPO, evolution) |
| `/results` | Performance tables and benchmark results |
| `/docs` | Full framework documentation with interactive diagrams |
| `/team` | Research team profiles |
| `/citation` | BibTeX citation and paper metadata |

## Citation

```bibtex
@inproceedings{srivastava2025debate,
  title={Debate, Train, Evolve: Self-Evolution of Language Model Reasoning},
  author={Srivastava, Gaurav and Bi, Zhenyu and Lu, Meng and Wang, Xuan},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  url={https://aclanthology.org/2025.emnlp-main.1666/}
}
```

---

<div align="center">

Made with &#10084;&#65039; by the DTE Research Team

[![Virginia Tech](https://img.shields.io/badge/Virginia_Tech-CS_Department-861F41?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJjdXJyZW50Q29sb3IiLz4KPC9zdmc+)](https://cs.vt.edu/)
[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-2a4dff?style=flat&logo=academia)](https://2025.emnlp.org/)
[![ACL Anthology](https://img.shields.io/badge/ACL_Anthology-2025.emnlp--main.1666-red?style=flat)](https://aclanthology.org/2025.emnlp-main.1666/)

</div>
