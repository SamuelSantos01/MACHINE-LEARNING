# Projeto de Análise de Emissões de CO2 e Potência de Motores

Este projeto utiliza as bibliotecas: `matplotlib`, `pandas`, `numpy`, `scikit-learn` e `math`. O onjetivo em Python é analisar a relação entre a potência dos motores e as emissões de CO2. Ele realiza uma análise usando um algoritmo de regressão linear para prever as emissões de CO2 com base na potência dos motores.

## Pré-requisitos

Certifique-se de ter instalado o Python e as seguintes bibliotecas:

- matplotlib
- pandas
- numpy
- scikit-learn

OBS: Todas essas bibliotecas listadas a cima não são padrão do python, logo é necessário a instalação delas no seu ambiente de desenvolvimento.

## Como utilizar o projeto

1. Clone este repositório:

2. Execute o script Python `analise_emissoes_co2.py`.

3. O programa irá:

    - Importar o conjunto de dados de um arquivo CSV chamado `FuelConsumptionCo2.csv`.
    - Dividir os dados em conjuntos de treino e teste.
    - Criar um modelo de regressão linear utilizando `scikit-learn`.
    - Treinar o modelo com os dados de treino.
    - Exibir gráficos com a visualização da relação entre potência dos motores e emissões de CO2.
    - Realizar previsões com os dados de teste.
    - Avaliar o desempenho do modelo usando métricas como Erro Quadrático Médio (MSE), Erro Médio Absoluto (MAE), Raiz do Erro Quadrático Médio (RMSE) e R²-score.

## Estrutura do Projeto

- `analise_emissoes_co2.py`: Script principal que contém o código para análise e visualização dos dados.
- `FuelConsumptionCo2.csv`: Conjunto de dados CSV contendo informações sobre o consumo de combustível e as emissões de CO2.
- `README.md`: Este arquivo, fornecendo uma visão geral do projeto e instruções de uso.

## Contribuições

Contribuições são bem-vindas! Se encontrar algum problema ou tiver sugestões para melhorias, sinta-se à vontade para abrir uma "issue" ou enviar um "pull request".
