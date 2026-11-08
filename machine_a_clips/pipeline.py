"""Enchaîne sondage, audio, transcription, choix des moments et montage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from machine_a_clips.config import cadrage_effectif, source_deja_verticale, Config
from machine_a_clips.errors import ErreurUtilisateur
from machine_a_clips.media import enveloppe_video, exiger_ffmpeg, sonder
from machine_a_clips.moments import choisir, construire_candidats, enrichir_candidats_live, fusionner
from machine_a_clips.montage import (
    accroche_verifiee,
    accroches_uniques,
    bonus_chat,
    charger_chat,
    confiance_de,
    corriger_mots,
    habiller_legende,
    plan_de_coupe,
    preciser_raison,
    remapper_mots,
    temps_sortie,
)
from machine_a_clips.programme import ecarter_plateformes, sans_mentions
from machine_a_clips.ollama_client import annoter, choisir_modele, corriger_noms, decharger, ollama_disponible
from machine_a_clips.overlays import PlanOverlay, decider, mesurer_video
from machine_a_clips.qualite import (
    dynamique_entre,
    mouvement_moyen,
    raison_rejet,
    rejet_image_figee,
)
from machine_a_clips.render import presentation_jeu, rendre_clip
from machine_a_clips.subtitles import ecrire_ass, grouper_cartes, installer_polices, mots_du_clip
from machine_a_clips.vision import appliquer_vision, noter_images
from machine_a_clips.summary import ClipExporte, ecrire_clip, ecrire_resume
from machine_a_clips.suivi import Suivi
from machine_a_clips.textutil import format_horloge, nom_dossier_sur, tampon_fichier
from machine_a_clips.transcript import Transcript, charger_transcript, transcrire
from machine_a_clips.zone_jeu import ZoneJeu, mesurer_zone

logger = logging.getLogger("machine_a_clips")

EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv"}


def lister_videos(entree: Path) -> list[Path]:
    entree = entree.expanduser()
    if not entree.exists():
        raise ErreurUtilisateur(f"Introuvable : {entree}")
    if entree.is_file():
        if entree.suffix.lower() not in EXTENSIONS:
            formats = ", ".join(sorted(EXTENSIONS))
            raise ErreurUtilisateur(f"Format non pris en charge ({entree.suffix}). Utilise : {formats}.")
        return [entree]
    videos = sorted(
        fichier for fichier in entree.iterdir() if fichier.is_file() and fichier.suffix.lower() in EXTENSIONS
    )
    if not videos:
        raise ErreurUtilisateur(
            f"Aucune vidéo dans {entree}. Formats acceptés : {', '.join(sorted(EXTENSIONS))}."
        )
    return videos


def traiter_entrees(
    entrees: list[Path],
    cfg: Config,
    transcript: Path | None = None,
    suivi: Suivi | None = None,
) -> list[Path]:
    exiger_ffmpeg()
    if transcript is not None and len(entrees) != 1:
        raise ErreurUtilisateur("--transcript s'utilise avec une seule vidéo.")
    dossiers: list[Path] = []
    total = len(entrees)
    for index, video in enumerate(entrees, start=1):
        _ping(suivi, "fichier", f"{video.name} ({index}/{total})", index, total)
        dossiers.append(traiter_video(video, cfg, transcript, suivi))
    _ping(suivi, "termine", "Terminé.", total, total)
    return dossiers


def traiter_video(
    video: Path,
    cfg: Config,
    transcript_path: Path | None = None,
    suivi: Suivi | None = None,
) -> Path:
    return rendre_preparation(preparer_video(video, cfg, transcript_path, suivi), suivi=suivi)


def preparer_video(
    video: Path,
    cfg: Config,
    transcript_path: Path | None = None,
    suivi: Suivi | None = None,
) -> Preparation:
    exiger_ffmpeg()
    from machine_a_clips.installation import adapter_modeles_legers

    note_machine = adapter_modeles_legers(cfg)
    if note_machine:
        logger.info("%s", note_machine)
    info = sonder(video)
    if not info.a_audio:
        raise ErreurUtilisateur(f"Pas de piste audio dans {video.name}. OBS doit enregistrer le son.")
    if info.duree <= 0.5:
        raise ErreurUtilisateur(f"{video.name} est trop courte pour en tirer un clip.")
    logger.info(
        "Vidéo : %s (%.0f s, %dx%d)",
        video.name,
        info.duree,
        info.largeur,
        info.hauteur,
    )
    if info.duree < cfg.duree_min_s:
        logger.warning(
            "La vidéo dure %.0f s, sous la durée minimale (%.0f s). Le clip couvrira tout le fichier.",
            info.duree,
            cfg.duree_min_s,
        )

    effectif = cadrage_effectif(cfg, info.largeur, info.hauteur)
    if source_deja_verticale(info.largeur, info.hauteur):
        logger.info("Source déjà verticale (%dx%d) : mode %s.", info.largeur, info.hauteur, effectif)
    else:
        logger.info("Source paysage (%dx%d) : mode %s.", info.largeur, info.hauteur, effectif)
    plan = _plan_overlays(video, info.duree, cfg, effectif)
    if plan.raison:
        logger.info("Overlays : %s — %s.", plan.action, plan.raison)

    logger.info("Recherche des pics audio…")
    _ping(suivi, "audio", "Analyse audio…")
    serie = enveloppe_video(
        video,
        cfg.echantillons,
        cfg.fenetre_audio_s,
        cfg.passe_haut_hz,
        annulation=None if suivi is None else suivi.annuler,
    )
    logger.info("%d fenêtre(s) audio analysée(s).", len(serie))
    if serie and max(valeur for _, valeur in serie) < 30:
        logger.warning("Audio très faible : les moments forts seront moins fiables.")

    _ping(suivi, "transcription", "Transcription…")
    if transcript_path is not None:
        transcription = charger_transcript(transcript_path, cfg.langue)
        logger.info("Transcription lue depuis %s (%d passage(s)).", transcript_path.name, len(transcription.segments))
    else:
        transcription = transcrire(video, cfg, suivi)
    _polir_dictionnaire(transcription)

    montage = sans_mentions(transcription, cfg.programme_live.interdits) if cfg.programme_live.actif else transcription
    candidats = construire_candidats(serie, info.duree, cfg, montage)
    candidats = enrichir_candidats_live(candidats, serie, transcription, info.duree, cfg)

    annotations = None
    ollama_utilise = False
    modele_texte = cfg.ollama_modele
    if cfg.ollama_actif:
        _ping(suivi, "ia_texte", "IA texte…")
        annotations, ollama_utilise, modele_texte = _noter_ollama(candidats, transcription, cfg)
        if cfg.vision_actif:
            _ping(suivi, "ia_visuelle", "IA visuelle…")
            visions = noter_images(video, candidats, cfg)
            if visions:
                annotations = appliquer_vision(annotations, visions)
                ollama_utilise = True
    else:
        if cfg.programme_live.actif:
            logger.info("Ollama ignoré. Classement sur les moments LIVE du transcript et l'énergie audio.")
        else:
            logger.info("Ollama ignoré (--sans-ollama ou config). Classement sur l'énergie audio.")

    return preparer_suite(
        video,
        cfg,
        info,
        transcription,
        serie,
        candidats,
        annotations,
        ollama_utilise,
        modele_texte,
        plan,
        effectif,
        suivi,
    )


@dataclass
class Preparation:
    """Analyse terminée, rendu pas encore lancé. La GUI peut décocher des clips."""

    video: Path
    cfg: Config
    info: object
    transcription: Transcript
    choisis: list
    plans: list
    confiances: list
    zone: ZoneJeu | None
    plan_overlay: PlanOverlay
    ollama_utilise: bool
    modele_texte: str
    ecarts: list = field(default_factory=list)
    effectif: str = "vertical"


def preparer_suite(
    video,
    cfg,
    info,
    transcription,
    serie,
    candidats,
    annotations,
    ollama_utilise,
    modele_texte,
    plan,
    effectif,
    suivi: Suivi | None = None,
) -> Preparation:
    messages = _messages_chat(video)
    moments = fusionner(candidats, annotations, transcription, cfg)
    if messages:
        for moment in moments:
            moment.score = min(1.0, moment.score + 0.08 * bonus_chat(messages, moment.debut, moment.fin))
    ecarts = []
    if cfg.programme_live.actif:
        moments, ecarts = ecarter_plateformes(moments, cfg)
        for ecart in ecarts:
            logger.warning(
                "Programme LIVE : passage écarté %s → %s, mention de « %s ».",
                format_horloge(ecart.debut),
                format_horloge(ecart.fin),
                ecart.motif,
            )
    choisis = _selectionner(video, moments, serie, transcription, cfg)
    if not choisis:
        if cfg.programme_live.actif:
            dossier = Path(cfg.dossier_sortie) / nom_dossier_sur(video.stem)
            dossier.mkdir(parents=True, exist_ok=True)
            ecrire_resume(
                dossier,
                video.resolve(),
                info.duree,
                [],
                langue=transcription.langue,
                moteur_transcript=transcription.moteur,
                ollama_utilise=ollama_utilise,
                ollama_modele=modele_texte,
                programme_actif=True,
                rappel=cfg.programme_live.rappel,
                avertissements=ecarts,
            )
            raise ErreurUtilisateur(
                "Programme LIVE : aucun passage ne parle du LIVE sans citer une autre plateforme "
                f"ou un logiciel de stream. Détail dans {dossier / 'resume.md'}."
            )
        raise ErreurUtilisateur(f"Aucun moment exploitable dans {video.name}.")
    if len(choisis) < cfg.clips:
        logger.warning(
            "%d clip(s) demandé(s), %d seul(s) tiennent sans se chevaucher.",
            cfg.clips,
            len(choisis),
        )
    for moment in choisis:
        moment.accroche = accroche_verifiee(moment.accroche, _detail_image(moment.raison), moment.texte)
        if not cfg.programme_live.actif:
            moment.raison = preciser_raison(moment.texte, moment.raison)
    if not cfg.programme_live.actif:
        accroches_uniques(choisis)
        for index, moment in enumerate(choisis):
            habiller_legende(moment, index)
    plans = []
    confiances = []
    for moment in choisis:
        coupe = plan_de_coupe(moment, transcription, serie, cfg, choisis)
        plans.append(coupe)
        confiances.append(confiance_de(moment, coupe))
        logger.info("Coupe : %s (confiance %.0f%%).", coupe.raison, confiances[-1] * 100)
    zone = _zone_de_jeu(video, info, cfg, effectif)
    return Preparation(
        video=video,
        cfg=cfg,
        info=info,
        transcription=transcription,
        choisis=choisis,
        plans=plans,
        confiances=confiances,
        zone=zone,
        plan_overlay=plan,
        ollama_utilise=ollama_utilise,
        modele_texte=modele_texte,
        ecarts=ecarts,
        effectif=effectif,
    )


def rendre_preparation(
    prep: Preparation,
    indices: list[int] | None = None,
    suivi: Suivi | None = None,
) -> Path:
    if indices is None:
        retenus = list(range(len(prep.choisis)))
    else:
        retenus = [index for index in indices if 0 <= index < len(prep.choisis)]
    cfg = prep.cfg
    demande = cfg.cadrage
    cfg.cadrage = prep.effectif
    try:
        return _monter(prep, retenus, suivi)
    finally:
        cfg.cadrage = demande


def _monter(prep: Preparation, retenus: list[int], suivi: Suivi | None = None) -> Path:
    video = prep.video
    cfg = prep.cfg
    info = prep.info
    transcription = prep.transcription
    plan = prep.plan_overlay
    dossier = Path(cfg.dossier_sortie) / nom_dossier_sur(video.stem)
    travail = dossier / "travail"
    travail.mkdir(parents=True, exist_ok=True)
    famille, famille_accroche = installer_polices(travail / "fonts", cfg.police)

    exports: list[ClipExporte] = []
    total = max(1, len(retenus))
    presentation = presentation_jeu(cfg, prep.zone)
    for numero, index_moment in enumerate(retenus, start=1):
        moment = prep.choisis[index_moment]
        coupe = prep.plans[index_moment]
        confiance = prep.confiances[index_moment]
        index = numero
        logger.info(
            "Montage %d/%d  %s → %s  score %.2f  %s",
            index,
            total,
            format_horloge(moment.debut),
            format_horloge(moment.fin),
            moment.score,
            moment.type_moment,
        )
        logger.info("  %s", moment.raison)
        _ping(suivi, "rendu", f"Rendu du clip {index}/{total}", index, total)
        cartes = []
        duree_sortie = coupe.duree()
        if cfg.sous_titres:
            relatifs = remapper_mots(transcription.mots(), coupe.segments)
            if not relatifs:
                relatifs = mots_du_clip(transcription.mots(), moment.debut, moment.fin)
            cartes = grouper_cartes(relatifs, cfg.mots_par_carte, cfg.max_caracteres, cfg.majuscules)
        habillage = cfg.cadrage in {"vertical", "plein"} and (
            cfg.gabarit.marque_actif or cfg.gabarit.carte_fin or cfg.gabarit.titre_persistant
        )
        ass_path = None
        if cartes or (cfg.accroche and moment.accroche) or habillage:
            ass_path = travail / f"clip_{index:02d}.ass"
            ecrire_ass(
                ass_path,
                cartes,
                moment.accroche if cfg.accroche else "",
                cfg.accroche_duree_s,
                duree_sortie,
                cfg,
                famille,
                famille_accroche,
                presentation=presentation,
            )
        fichier = dossier / f"clip_{index:02d}_{tampon_fichier(moment.debut)}.mp4"
        payoff = None if coupe.payoff is None else temps_sortie(coupe.payoff, coupe.segments)
        codec = rendre_clip(
            video,
            fichier,
            moment.debut,
            moment.fin,
            cfg,
            info,
            ass_path,
            travail / "fonts" if ass_path else None,
            plan,
            annulation=None if suivi is None else suivi.annuler,
            zone=prep.zone,
            segments=coupe.segments,
            payoff=payoff,
        )
        detail = f"Montage : {coupe.raison}"
        export = ClipExporte(
            index=index,
            source=video.resolve(),
            fichier=fichier,
            debut=moment.debut,
            fin=moment.fin,
            score=moment.score,
            score_audio=moment.score_audio,
            score_llm=moment.score_llm,
            mode_score=moment.mode_score,
            type_moment=moment.type_moment,
            raison=moment.raison,
            accroche=moment.accroche,
            legende=moment.legende,
            hashtags=moment.hashtags,
            texte=moment.texte,
            cadrage=cfg.cadrage,
            codec=codec,
            cartes=[(carte.texte, carte.debut, carte.fin) for carte in cartes],
            rappel=cfg.programme_live.rappel if cfg.programme_live.actif else "",
            confiance=confiance,
            detail=detail,
        )
        ecrire_clip(export)
        exports.append(export)
        logger.info("  %s", fichier)

    ecrire_resume(
        dossier,
        video.resolve(),
        info.duree,
        exports,
        langue=transcription.langue,
        moteur_transcript=transcription.moteur,
        ollama_utilise=prep.ollama_utilise,
        ollama_modele=prep.modele_texte,
        programme_actif=cfg.programme_live.actif,
        rappel=cfg.programme_live.rappel if cfg.programme_live.actif else "",
        avertissements=prep.ecarts if cfg.programme_live.actif else None,
    )
    logger.info("Terminé. %d clip(s) dans %s", len(exports), dossier)
    return dossier


def _plan_overlays(video: Path, duree: float, cfg: Config, effectif: str) -> PlanOverlay:
    if effectif != "plein" or cfg.overlays == "garder":
        return PlanOverlay("garder", raison="")
    bandes = mesurer_video(video, duree, cfg.overlay_echantillons)
    return decider(bandes, cfg)


def _selectionner(video: Path, moments, serie, transcription: Transcript, cfg: Config):
    ecart = cfg.ecart_clips_s
    if not cfg.qualite_actif:
        return choisir(moments, cfg.clips, ecart_min_s=ecart)
    propres = []
    for moment in moments:
        raison = raison_rejet(serie, moment.debut, moment.fin, transcription, cfg)
        if raison:
            logger.info("Écarté %s → %s : %s", format_horloge(moment.debut), format_horloge(moment.fin), raison)
            continue
        propres.append(moment)
    if not propres:
        logger.warning("Contrôle qualité : tout a été écarté. Je garde le mieux noté.")
        propres = list(moments)
    exclus: set[int] = set()
    while True:
        propos = choisir(
            [moment for moment in propres if id(moment) not in exclus],
            cfg.clips,
            ecart_min_s=ecart,
        )
        if not propos or cfg.qualite_mouvement_min <= 0:
            return propos
        bloque = None
        for moment in propos:
            diff = mouvement_moyen(video, moment.debut, moment.fin)
            if diff is None:
                continue
            mots = len(transcription.texte_entre(moment.debut, moment.fin).split())
            raison = rejet_image_figee(diff, mots, dynamique_entre(serie, moment.debut, moment.fin), cfg)
            if raison:
                logger.info(
                    "Écarté %s → %s : %s (écart %.1f).",
                    format_horloge(moment.debut),
                    format_horloge(moment.fin),
                    raison,
                    diff,
                )
                exclus.add(id(moment))
                bloque = moment
                break
        if bloque is None:
            return propos
        if len(exclus) >= len(propres):
            logger.warning("Contrôle qualité : images trop figées. Je garde le mieux noté.")
            return choisir(moments, cfg.clips, ecart_min_s=ecart)


def _noter_ollama(candidats, transcription: Transcript, cfg: Config):
    ok, modeles = ollama_disponible(cfg.ollama_url, delai_s=2.0)
    if not ok:
        logger.warning(
            "Ollama ne répond pas sur %s. Les clips sont classés sur l'énergie audio seulement.\n"
            "  Ouvre l'application Ollama, ou lance : ollama pull %s",
            cfg.ollama_url,
            cfg.ollama_modele,
        )
        return None, False, cfg.ollama_modele
    choisi = choisir_modele(cfg, modeles)
    if choisi is None:
        logger.warning(
            "Ni « %s » ni « %s » ne sont dans Ollama. Classement sur l'énergie audio.\n"
            "  ollama pull %s",
            cfg.ollama_modele,
            cfg.ollama_repli,
            cfg.ollama_modele,
        )
        return None, False, cfg.ollama_modele
    if choisi != cfg.ollama_modele:
        logger.warning("« %s » absent, repli sur %s.", cfg.ollama_modele, choisi)
        cfg.ollama_modele = choisi
    logger.info("Ollama (%s) relit les extraits…", cfg.ollama_modele)
    try:
        corriger_noms(transcription, cfg)
    except Exception as exc:
        logger.info("Correction des noms ignorée (%s).", exc)
    notes = annoter(candidats, transcription, cfg)
    decharger(cfg.ollama_url, cfg.ollama_modele)
    logger.info("Modèle texte déchargé (keep_alive 0).")
    return notes, notes is not None, cfg.ollama_modele


def _polir_dictionnaire(transcription: Transcript) -> None:
    for segment in transcription.segments:
        segment.texte = corriger_mots(segment.texte)
        for mot in segment.mots:
            mot.texte = corriger_mots(mot.texte)


def _detail_image(raison: str) -> str:
    marque = "Image :"
    if marque not in (raison or ""):
        return ""
    return raison.split(marque, 1)[1].strip()


def _zone_de_jeu(video: Path, info, cfg: Config, effectif: str) -> ZoneJeu | None:
    if not cfg.auto_cadrage or effectif not in {"plein", "vertical"}:
        return None
    zone = mesurer_zone(video, info.duree)
    if zone is None:
        logger.info("Zone de jeu : l'image bouge sur presque tout le cadre, pas de recadrage.")
        return None
    logger.info(
        "Zone de jeu : x=%.2f y=%.2f w=%.2f h=%.2f (%s).",
        zone.x,
        zone.y,
        zone.w,
        zone.h,
        zone.raison,
    )
    return zone


def _messages_chat(video: Path) -> list[tuple[float, str]]:
    for chemin in (video.with_suffix(".chat.json"), video.with_name(video.stem + "_chat.json")):
        messages = charger_chat(chemin)
        if messages:
            logger.info("Chat : %d message(s) depuis %s.", len(messages), chemin.name)
            return messages
    return []


def _ping(suivi: Suivi | None, code: str, message: str, courant: int = 0, total: int = 0) -> None:
    if suivi is not None:
        suivi.etape(code, message, courant, total)
