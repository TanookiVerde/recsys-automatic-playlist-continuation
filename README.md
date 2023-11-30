# Sistemas de Recomendação - Trabalho Final

- Referência: https://github.com/hojinYang/spotify_recSys_challenge_2018
- Desafio ACM RecSys: http://www.recsyschallenge.com/2018/

# Setup

## Dependências

1. Baixar dados do desafio no [site](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#dataset)
   - Dataset completo: `spotify_million_playlist_dataset`
   - Dataset de Desafio: `spotify_million_playlist_dataset_challenge`
1. Extrair dados do desafio para dentro da pasta `dados/`.
   - Ficarão duas pastas: `spotify_million_playlist_dataset_challenge/` e `spotify_million_playlist_dataset/`
1. Criar ambiente virtual com `python -m venv env`
1. Ativando ambiente virtual
   - Windows: `env\Scripts\Activate.ps1`
   - Linux: `source env/bin/activate`
1. Instalando dependências com `pip install -r requirements.txt`


## Preparação de Dados

1. Rodar `implementacao-nova/0_processa_dados.ipynb`: Assim ocorrerá a geração dos dados preparados na pasta `dados-processados/`

# Planejamento

- Modelo 1: Personalized Random Walk
   - Entrada: [Playlist Atual]
   - Saída: [Ranking]
- Modelo 2: Character-level CNN
   - Entrada: [Título]
   - Saída: [Ranking]