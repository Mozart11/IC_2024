## IC_2024

### MNIST simulations
Os códigos relacionados ao MNIST, estão na pasta *MNIST_codes*.


**Antes de tudo**: Executar os códigos sets_gen.py e sets_gen_noise.py, para a parti dos arquivos binários baixados do site oficial do MNIST, que estão na pasta *MNIST_datasets*, serem gerados os nossos conjuntos de treino, teste e validação na mesma pasta.

Dentro da pasta dos códigos, existem as pastas dos testes do knn, svm e decision tree, cada um com seus respecitivos nomes. Dentro de cada uma dessas pastas existem os códigos:
- Validação.py, que são os códigos que usei para estimar os parâmetros ideais (cross-validation) e validar eles no conjunto de validação
- knn.py,svm.py ou tree.py são onde eu faço a avaliação final, usando o conjunto (train + validation) para treinar meu modelo com os hiperparâmetros ideais do código de validação no conjunto de teste e printo os resultados (matched e missmatched)
- os códigos com sample_complexity nas pastas são para gerar os NPZs com os valores para plotar a curva
- sample_complexity_plot.py, que está fora das pastas citadas, é o código que plota a complexity curve acessando os NPZs que são gerados depois de executar um a um dos códigos sample comeplexity das pastas.
- A pasta *Olds* são ideias e códigos antigos que eu tive em algum momento, **ignorar**.

### Olivetti simulations

Os códigos estão na pasta *Olivetti_faces_codes*.

Segue literalmente a mesma estrutura do MNIST, com a diferença em que aqui não existe conjunto de validação, apenas train e teste. Logo os arquivos de validação não existem, a unica coisa que mantive aqui é a parte de encontrar os hiperparâmetros ideais, mas somente para o KNN, que está no seu própio código, knn.py .
