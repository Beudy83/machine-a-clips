"""Ligne de commande : python -m machine_a_clips ma_video.mp4 --clips 5"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from machine_a_clips import __version__
from machine_a_clips.config import Config, appliquer_profil_court, appliquer_programme_live, charger
from machine_a_clips.demo import (
    BURSTS_PROGRAMME,
    SCENES_PROGRAMME,
    appliquer_reglages_demo,
    exporter_apercus,
    generer_video_essai,
)
from machine_a_clips.errors import ErreurUtilisateur
from machine_a_clips.media import nvenc_disponible, trouver_police, version_outil
from machine_a_clips.ollama_client import modele_present, ollama_disponible
from machine_a_clips.pipeline import lister_videos, traiter_entrees

logger = logging.getLogger("machine_a_clips")


def configurer_stdio() -> None:
    for flux in (sys.stdout, sys.stderr):
        reconfigurer = getattr(flux, "reconfigure", None)
        if reconfigurer is None:
            continue
        try:
            reconfigurer(encoding="utf-8", errors="replace")
        except Exception:
            pass


def configurer_logs(verbeux: bool) -> None:
    niveau = logging.DEBUG if verbeux else logging.INFO
    journal = logging.getLogger("machine_a_clips")
    journal.setLevel(niveau)
    if not journal.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        journal.addHandler(handler)
    journal.propagate = False


def construire_parser() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        prog="machine-a-clips",
        description=(
            "Transforme un live ou une VOD retrogaming en clips TikTok verticaux (9:16), "
            "entièrement sur cette machine."
        ),
    )
    parseur.add_argument("entree", nargs="?", help="Fichier vidéo (mp4, mkv…) ou dossier")
    parseur.add_argument("--clips", "-n", type=int, default=None, help="Nombre de clips (défaut : 5)")
    parseur.add_argument("--config", "-c", type=Path, default=None, help="Fichier config.yaml")
    parseur.add_argument("--sortie", "-o", default=None, help="Dossier de sortie")
    parseur.add_argument("--langue", default=None, help="Langue de la transcription (défaut : fr)")
    parseur.add_argument(
        "--modele-whisper",
        default=None,
        help="Modèle faster-whisper : tiny, base, small, medium",
    )
    parseur.add_argument("--modele-ollama", default=None, help="Modèle Ollama, par exemple llama3.2:3b")
    parseur.add_argument(
        "--sans-ollama",
        action="store_true",
        help="Ne contacte pas Ollama : classement sur l'énergie audio seulement",
    )
    parseur.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Transcription JSON déjà prête (saute Whisper)",
    )
    parseur.add_argument(
        "--mode-cadre",
        choices=("vertical", "gabarit", "plein", "flou", "crop", "empile"),
        default=None,
        help="vertical (gabarit 16:9 ou plein écran si la source est déjà 9:16), flou, crop, empile",
    )
    parseur.add_argument(
        "--profil",
        choices=("long", "court"),
        default=None,
        help="long : clips > 60 s. court : 15 à 45 s",
    )
    parseur.add_argument("--courts", action="store_true", help="Clips courts (15–45 s) au lieu du profil > 60 s")
    parseur.add_argument(
        "--programme-live",
        action="store_true",
        help=(
            "Campagne LIVE Incentive Program : 60 à 75 s, moments LIVE, "
            "sans autre plateforme, overlays TikTok gardés, #PaidPartnership"
        ),
    )
    parseur.add_argument(
        "--overlays",
        choices=("garder", "masquer", "auto"),
        default=None,
        help="Bandeaux TikTok déjà dans l'image : garder, masquer, ou auto",
    )
    parseur.add_argument("--sans-auto", action="store_true", help="N'adapte pas le cadrage au format de la source")
    parseur.add_argument("--sans-vision", action="store_true", help="Pas de passe image qwen2.5vl")
    parseur.add_argument("--sans-qualite", action="store_true", help="Garde aussi les extraits silencieux ou figés")
    parseur.add_argument("--duree-min", type=float, default=None, help="Durée minimale d'un clip, en secondes")
    parseur.add_argument("--duree-cible", type=float, default=None, help="Durée visée, en secondes")
    parseur.add_argument("--duree-max", type=float, default=None, help="Durée maximale d'un clip, en secondes")
    parseur.add_argument("--sans-sous-titres", action="store_true", help="N'incruste pas les sous-titres")
    parseur.add_argument("--demo", action="store_true", help="Fabrique un faux live d'une minute et le gabarit vertical")
    parseur.add_argument(
        "--interface",
        action="store_true",
        help="Ouvre la fenêtre (glisser-déposer). Sans console : pythonw -m machine_a_clips.gui",
    )
    parseur.add_argument("--verifier", action="store_true", help="Vérifie FFmpeg, Whisper, Ollama et le GPU")
    parseur.add_argument("--verbeux", "-v", action="store_true", help="Affiche plus de détails")
    return parseur


def appliquer_arguments(cfg: Config, args: argparse.Namespace) -> Config:
    if args.clips is not None:
        cfg.clips = args.clips
    if args.sortie:
        cfg.dossier_sortie = args.sortie
    if args.langue:
        cfg.langue = args.langue
    if args.modele_whisper:
        cfg.modele_whisper = args.modele_whisper
    if args.modele_ollama:
        cfg.ollama_modele = args.modele_ollama
    if args.sans_ollama:
        cfg.ollama_actif = False
    if args.mode_cadre:
        cfg.cadrage = "vertical" if args.mode_cadre == "gabarit" else args.mode_cadre
    if args.profil == "court" and args.duree_min is None and args.duree_cible is None and args.duree_max is None:
        appliquer_profil_court(cfg)
    elif args.profil:
        cfg.profil = args.profil
    if args.courts:
        appliquer_profil_court(cfg)
    if args.overlays:
        cfg.overlays = args.overlays
    if args.sans_auto:
        cfg.auto_cadrage = False
    if args.sans_vision:
        cfg.vision_actif = False
    if args.sans_qualite:
        cfg.qualite_actif = False
    if args.programme_live:
        cfg.programme_live.actif = True
    if cfg.programme_live.actif:
        appliquer_programme_live(cfg, forcer_durees=True, forcer_overlays=args.overlays is None)
    if args.duree_min is not None:
        cfg.duree_min_s = args.duree_min
    if args.duree_cible is not None:
        cfg.duree_cible_s = args.duree_cible
    if args.duree_max is not None:
        cfg.duree_max_s = args.duree_max
    if args.sans_sous_titres:
        cfg.sous_titres = False
    cfg.valider()
    return cfg


def rapport_verification(cfg: Config | None = None) -> tuple[str, bool]:
    cfg = cfg or Config()
    lignes = [f"machine-a-clips {__version__}"]
    version_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    lignes.append(f"Python {version_python}")
    ffmpeg = version_outil("ffmpeg")
    ffprobe = version_outil("ffprobe")
    if ffmpeg:
        lignes.append(f"FFmpeg : {ffmpeg}")
    else:
        lignes.append("FFmpeg : INTROUVABLE")
    if ffprobe:
        lignes.append(f"ffprobe : {ffprobe}")
    else:
        lignes.append("ffprobe : INTROUVABLE")
    try:
        _chemin, famille = trouver_police(cfg.police)
        lignes.append(f"Police sous-titres : {famille}")
    except ErreurUtilisateur as exc:
        lignes.append(f"Police sous-titres : INTROUVABLE ({exc})")
    try:
        import faster_whisper

        version = getattr(faster_whisper, "__version__", "installé")
        appareil = "cpu"
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                appareil = "cuda"
        except Exception:
            appareil = "cpu"
        from machine_a_clips.transcript import decrire_moteur_whisper

        lignes.append(f"faster-whisper {version} ({appareil})")
        lignes.append(decrire_moteur_whisper(cfg))
    except ImportError:
        lignes.append("faster-whisper : ABSENT — pip install -r requirements.txt")
    else:
        from machine_a_clips.whisper_local import decrire_modele

        etat = decrire_modele(cfg.modele_whisper)
        lignes.append(("Whisper prêt : " if etat.ok else "Whisper : ") + etat.detail)
    from machine_a_clips.installation import etat_dll_cuda

    _etat_dll, detail_dll = etat_dll_cuda()
    lignes.append(f"DLL CUDA : {detail_dll}")
    ok, modeles = ollama_disponible(cfg.ollama_url, delai_s=2.0)
    if not ok:
        lignes.append(
            f"Ollama : injoignable sur {cfg.ollama_url} "
            "(les clips sortiront quand même, classés sur le son)"
        )
    elif modele_present(cfg.ollama_modele, modeles):
        lignes.append(f"Ollama : OK, modèle {cfg.ollama_modele}")
    else:
        lignes.append(
            f"Ollama répond, mais « {cfg.ollama_modele} » manque. "
            f"Lance : ollama pull {cfg.ollama_modele}"
        )
    if nvenc_disponible():
        lignes.append("Encodeur : h264_nvenc est listé (repli automatique sur libx264 s'il refuse)")
    else:
        lignes.append("Encodeur : libx264 (h264_nvenc absent de ce FFmpeg)")
    pret = ffmpeg is not None and ffprobe is not None
    lignes.append("Prêt à découper." if pret else "Installe FFmpeg avant de lancer un live.")
    return "\n".join(lignes), pret


def commande_demo(cfg: Config, dossier: Path) -> int:
    appliquer_reglages_demo(cfg)
    dossier.mkdir(parents=True, exist_ok=True)
    video = dossier / "live_demo.mp4"
    transcript = dossier / "transcription_demo.json"
    cfg.dossier_sortie = str(dossier / "clips")
    if cfg.programme_live.actif:
        logger.info(
            "Démo programme LIVE : un clip de 60 à 75 s, overlays gardés, sans Whisper ni Ollama."
        )
        generer_video_essai(
            video,
            transcript,
            scenes=list(SCENES_PROGRAMME),
            bursts=list(BURSTS_PROGRAMME),
            motif="testsrc2",
            largeur=720,
            hauteur=1280,
            fps=12,
        )
    else:
        logger.info("Démo : un clip de plus d'une minute, gabarit vertical néon, sans Whisper ni Ollama.")
        generer_video_essai(video, transcript, motif="testsrc2", largeur=1280, hauteur=720, fps=12)
    logger.info("Pour un vrai live : python -m machine_a_clips ta_video.mp4 --clips 5")
    traiter_entrees([video], cfg, transcript)
    clips = sorted((Path(cfg.dossier_sortie) / "live_demo").glob("clip_*.mp4"))
    if clips:
        apercus = exporter_apercus(clips[0], clips[0].parent / "apercus")
        logger.info("Aperçus : %s", ", ".join(chemin.name for chemin in apercus))
    logger.info("Regarde les clips dans %s", cfg.dossier_sortie)
    return 0


def main(argv: list[str] | None = None) -> int:
    from machine_a_clips.installation import preparer_environnement

    preparer_environnement()
    configurer_stdio()
    parseur = construire_parser()
    args = parseur.parse_args(argv)
    configurer_logs(args.verbeux)
    try:
        if args.interface:
            from machine_a_clips.gui import main as ouvrir_interface

            return ouvrir_interface()
        if args.verifier and not args.entree and not args.demo:
            cfg = charger(args.config)
            texte, pret = rapport_verification(cfg)
            logger.info(texte)
            return 0 if pret else 1
        if args.demo:
            cfg = appliquer_arguments(charger(args.config), args)
            dossier = Path(args.sortie) if args.sortie else Path("demo")
            return commande_demo(cfg, dossier)
        if not args.entree:
            parseur.print_help()
            return 2
        cfg = appliquer_arguments(charger(args.config), args)
        videos = lister_videos(Path(args.entree))
        traiter_entrees(videos, cfg, args.transcript)
        return 0
    except ErreurUtilisateur as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrompu.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
