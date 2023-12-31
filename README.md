# Sistemas de Recomendação - Trabalho Final

- Objetivo é implementar e avaliar um modelo Personalized PageRank para Continuação Automática de Playlists
- O dataset utilizado vem do desafio ACM RecSys: http://www.recsyschallenge.com/2018/

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


# Execução

1. Rodar `implementacao-nova/0_processa_dados.ipynb`: Assim ocorrerá a geração dos dados preparados na pasta `dados-processados/`
1. Rodar `implementacao-nova/1_treinamento_personalized_pagerank.ipynb`: Para gerar os pesos do modelo na pasta `dados-processados/`
1. Rodar `implementacao-nova/2_avaliacao_personalized_pagerank.ipynb`: Para avaliar o modelo isoladamente no conjunto de teste.

# Planejamento

- Modelo: Personalized Page Rank
   - Referências:
       - "Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time"
       - "Random Walk with Restart for Automatic Playlist Continuation and Query-Specific Adaptations"

