# Machine à clips

Découpe un live ou une VOD (Mario 64, Sonic, OBS ou TikTok LIVE Studio) en clips TikTok verticaux, prêts pour le programme Creator Rewards.

Tout reste sur ton PC : pas de cloud, pas de compte, pas d'envoi de la vidéo.

Le programme :

1. lit un fichier (`mp4`, `mkv`, …) ou un dossier ;
2. transcrit la voix en français avec Whisper `large-v3-turbo` ;
3. repère les cris et les pics, puis construit un passage cohérent autour ;
4. demande à Ollama (`qwen2.5:7b`) une note, une accroche et une légende, puis à `qwen2.5vl:7b` de regarder quelques images ;
5. exporte les meilleurs passages en plus de 60 secondes (61 à 90 s), en 1080×1920, avec accroche, sous-titres karaoké et une carte de fin.

Ce dépôt présente l'idée du projet et deux fichiers du code réel : le pipeline (`machine_a_clips/pipeline.py`) et l'interface en ligne de commande (`machine_a_clips/cli.py`).
