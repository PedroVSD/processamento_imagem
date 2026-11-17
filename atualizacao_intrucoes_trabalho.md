# Trabalho Final — PathoShift Mini Challenge: Processamento Robusto de Imagens Histopatológicas
## 1. Visão geral
Neste trabalho, cada grupo implementará e avaliará um pipeline de processamento e classificação de imagens histopatológicas. A tarefa será realizada sobre imagens microscópicas individuais (patches ou tiles) fornecidas ou indicadas pela disciplina.

O objetivo é aplicar conceitos de processamento de imagens a um problema real: imagens de patologia variam em cor, iluminação, foco, compressão e origem. Portanto, não basta treinar um classificador; o grupo deverá investigar se uma escolha de pré-processamento torna a solução mais robusta.

O trabalho não tem finalidade de diagnóstico clínico. Todas as predições devem ser tratadas como resultados experimentais em dados públicos e anonimizados.

## 2. Objetivos de aprendizagem
- Ler, organizar e inspecionar um conjunto de imagens.
- Aplicar e justificar operações de processamento de imagens.
- Construir um baseline de classificação reproduzível.
- Comparar experimentalmente um baseline e uma proposta de pré-processamento.
- Avaliar resultados com métricas adequadas, análise de erros e limitações.
- Comunicar riscos de generalização em imagens médicas.
## 3. Dados e escopo
O dataset oficial é o pacote PathoShift-CRC e suas partições são definidas neste enunciado. O conjunto é composto por imagens individuais; o processamento de lâminas inteiras (Whole Slide Images) e agregação por lâmina não são requisitos deste trabalho.

- Usar apenas os dados, partições e fontes externas permitidos pela disciplina.
- Não usar dados de teste para selecionar hiperparâmetros ou escolher a melhor versão do modelo.
- Não usar identificadores, nomes de arquivos ou metadados como atalhos para inferir rótulos.
- Registrar origem dos dados, versões, seeds e configurações usadas.
Embora o dataset de origem tenha classes aproximadamente desbalanceadas, o subconjunto PathoShift-CRC é balanceado por classe. Ainda assim, acurácia isolada não será uma medida suficiente de qualidade; F1-macro e métricas por classe são obrigatórias.

## 4. Requisitos da entrega
- Análise inicial dos dados: distribuição de classes, dimensões, exemplos representativos e breve inspeção de cor ou qualidade.
- Baseline reproduzível: classificador simples treinado e avaliado nas partições fornecidas.
- Proposta de processamento de imagens: pelo menos uma transformação ou sequência de transformações justificada e implementada pelo grupo.
- Comparação controlada: resultados do baseline e da proposta sob o mesmo protocolo, com ao menos uma ablação simples quando aplicável.
- Teste de robustez: avaliação em ao menos uma condição adicional, como mudança de brilho/cor, desfoque, ruído ou compressão.
- Análise de erros: exemplos visuais de acertos e falhas, com interpretação das limitações observadas.
São exemplos de técnicas possíveis: conversão RGB–HSV/LAB, normalização ou aumentação de cor, equalização, filtragem, redução de ruído, avaliação de foco, rejeição de fundo, extração de textura, descritores clássicos, redes neurais leves ou modelos visuais pré-treinados autorizados. Não há uma técnica obrigatória; será avaliada a coerência entre hipótese, implementação e resultados.

## 5. Aproveitamento da etapa inicial do trabalho
O que foi desenvolvido na primeira etapa do enunciado anterior — organização e inspeção dos dados, pré-processamento, baseline de classificação, métricas e análise inicial de erros — pode e deve ser aproveitado como a base desta entrega única.

Não é necessário reimplementar do zero um componente que o grupo já tenha desenvolvido. Para aproveitá-lo, o grupo deverá:

incluir o código-fonte atual e instruções de reprodução;
indicar no relatório, em uma subseção breve, quais componentes foram desenvolvidos anteriormente e quais foram modificados agora;
executar o baseline no dataset e nos manifestos oficiais atuais, quando os dados anteriores forem diferentes;
usar o baseline reaproveitado como termo de comparação para a proposta de processamento e o teste de robustez.
O aproveitamento não transfere automaticamente uma nota anterior: a avaliação continuará sendo feita sobre o artefato final, sua execução no protocolo atual e a qualidade da comparação experimental. Porém, resultados, gráficos e decisões da etapa inicial podem ser incorporados ao relatório quando permanecerem pertinentes e forem identificados como tal.

## 6. Pergunta de investigação
Cada grupo deverá formular uma pergunta simples e mensurável. Exemplos:

- A normalização de cor melhora a classificação em imagens com coloração alterada?
- Qual espaço de cor separa melhor determinadas classes de tecido?
- Filtrar imagens desfocadas melhora a qualidade média das predições?
- O método continua funcionando após compressão ou ruído moderado?
- Aumentação de cor é mais robusta que normalização de cor?
- A hipótese não precisa ser confirmada. Um resultado negativo bem medido e interpretado é um resultado válido.

## 7. Avaliação
| Critério                                             | Peso sugerido |
|------------------------------------------------------|:-------------:|
| Pipeline funcional, organização e reprodutibilidade  | 25%           |
| Qualidade e justificativa do processamento de imagens | 25%         |
| Protocolo experimental e métricas                    | 20%           |
| Robustez, análise de erros e limitações               | 20%           |
| Relatório e apresentação                              | 10%           |
| **Total**                                            | **100%**      |

Métricas apropriadas poderão incluir F1-macro, precisão, revocação, matriz de confusão e, quando aplicável, calibração. A escolha deve ser justificada em função da tarefa e do balanceamento das classes.

O grupo não será avaliado apenas pelo maior número obtido. Resultados sem experimento comparativo, com vazamento entre partições ou que não possam ser reproduzidos não receberão pontuação integral.

## 8. O que deve ser entregue
- Código-fonte completo e instruções para executar o experimento.
- Arquivo README com dependências, comandos, seed e configuração principal.
- Requisitos do ambiente, por exemplo requirements.txt ou equivalente.
- Relatório em PDF, com no máximo seis páginas, excluindo referências e apêndice técnico.
- predictions_validation.csv e predictions_robustness.csv, conforme o contrato da Seção 11.2.
- Apresentação breve de todos os integrantes em 8 de dezembro de 2026.
O relatório deverá conter: objetivo e hipótese; dados e partições; pré-processamento; baseline e proposta; protocolo; resultados; análise de erros; limitações; e uma seção breve intitulada AI-Assisted Engineering.

## 9. Uso de IA generativa e modelos pré-treinados
Ferramentas de IA generativa podem ser usadas para programação, estudo, depuração, documentação e análise. Modelos pré-treinados são permitidos somente quando puderem ser executados localmente, forem declarados no relatório e não substituírem a comparação com o baseline do grupo.

O grupo continua integralmente responsável pela correção e deve conseguir explicar o código, as escolhas de processamento, as métricas e os próprios resultados. No relatório, deverá declarar resumidamente ferramentas de IA generativa, modelos externos e fontes de dados adicionais utilizados, bem como descrever como verificou os resultados.

Não é permitido apresentar resultado gerado por serviço externo ou modelo remoto como se fosse execução local do grupo, nem transferir dados de teste para serviços externos sem autorização.

## 10. Ética e comunicação de resultados
O trabalho não produz ferramenta médica, laudo ou recomendação clínica. Não devem ser feitas alegações de segurança, eficácia clínica ou substituição de profissionais de saúde. Os grupos devem explicitar limitações, possíveis vieses, risco de mudança de domínio e a diferença entre desempenho experimental e validação clínica. Dados identificáveis, não públicos ou obtidos sem autorização não podem ser usados.

## 11. Cronograma, infraestrutura e regras operacionais
| Marco                                                    | Data                                      |
|----------------------------------------------------------|-------------------------------------------|
| Disponibilização do pacote de dados, manifestos e avaliador | 16 de novembro de 2026                 |
| Entrega única no Moodle                                  | 7 de dezembro de 2026, 23h59 (horário de Brasília) |
| Apresentação em sala                                     | 8 de dezembro de 2026                    |
### 11.1 Dataset oficial
O dataset oficial será o pacote PathoShift-CRC, preparado a partir do conjunto público NCT-CRC-HE-100K-NONORM. O repositório oficial de origem, que também contém o conjunto externo CRC-VAL-HE-7K, está disponível em Zenodo — 100,000 histological images of human colorectal cancer and healthy tissue. Ele contém imagens H&E de tecido colorretal em nove classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR e TUM.

| Arquivo          | Conteúdo                                                        | Uso permitido                          |
|------------------|-----------------------------------------------------------------|----------------------------------------|
| `train.csv`      | 9.000 imagens, 1.000 por classe                                 | Treinamento                             |
| `validation.csv` | 2.250 imagens, 250 por classe                                   | Escolha de método e hiperparâmetros     |
| `robustness.csv` | Manifesto de perturbações determinísticas a aplicar às imagens de validação | Avaliação de robustez |
| `labels.csv`     | Rótulos de treino e validação                                   | Supervisão e avaliação local            |

As imagens TIFF de 224×224 e os manifestos serão distribuídos pela disciplina; não é necessário baixar o dataset completo original no Zenodo. A referência bibliográfica do dataset de origem está em CITATION.md e no registro Zenodo indicado acima. A divisão treino–validação é estratificada por classe. Como o dataset de origem não disponibiliza identificação individual de paciente para cada tile, essa divisão não deve ser interpretada como validação clínica por paciente; esta é uma limitação que deve ser reconhecida no relatório.

Atenção: o arquivo original do NCT-CRC-HE-100K-NONORM contém aproximadamente 100.000 tiles, mas o dataset desta disciplina é o subconjunto de 11.250 imagens definido pelos manifestos: 9.000 em train.csv e 2.250 em validation.csv. O grupo deve usar exclusivamente as imagens cujos caminhos aparecem nesses dois arquivos; não deve selecionar aleatoriamente nem incluir outros tiles do arquivo original. robustness.csv não adiciona novas imagens: ele descreve perturbações que devem ser aplicadas às 2.250 imagens de validação.

### 11.2 Protocolo de avaliação
Cada grupo deve treinar exclusivamente com train.csv. validation.csv pode ser usado para comparação e escolha de configurações. O resultado principal será a avaliação final da configuração escolhida em robustness.csv.

O conjunto de robustez aplica, de forma determinística, uma das seguintes perturbações a cada imagem: variação de cor/brilho, compressão JPEG, desfoque gaussiano ou ruído gaussiano. Os parâmetros e a seed de cada caso estão em robustness.csv. Não é permitido retreinar nem selecionar hiperparâmetros com base nos resultados desse arquivo.

O grupo deverá entregar dois arquivos CSV UTF-8, sem índice adicional:

````python
# predictions_validation.csv e predictions_robustness.csv
image_id,predicted_label,confidence
TUM_AAAABBBB,TUM,0.87
````
image_id deve corresponder exatamente ao manifesto: use os valores de validation.csv para predictions_validation.csv e os valores de robustness.csv para predictions_robustness.csv. predicted_label deve ser uma das nove classes; e confidence deve estar entre 0 e 1. O avaliador local calcula acurácia, F1-macro, F1 por classe e matriz de confusão; o relatório deve apresentar também a queda de F1-macro entre validação e robustez.

### 11.3 Baseline e avaliador local
Não há um modelo treinado de referência fornecido pela disciplina. O baseline obrigatório é o classificador simples desenvolvido pelo próprio grupo, preferencialmente reaproveitado da primeira etapa do trabalho anterior. Ele deverá ser comparado à proposta de processamento sob os mesmos manifestos e a mesma métrica.

O pacote inclui evaluate_pathoshift_crc.mjs, um avaliador local sem dependências externas. Ele valida o contrato dos CSVs e produz acurácia, F1-macro, F1 por classe e matriz de confusão. Redes neurais e modelos visuais pré-treinados são permitidos, desde que também apresentem a comparação com o baseline do grupo.

### 11.4 Limites de recursos e submissão
Não haverá Docker nem servidor de treinamento. O trabalho deve ser executável em computador local com, no mínimo, 4 núcleos de CPU, 8 GiB de RAM e 3 GiB de espaço livre, sem GPU. GPU pode ser usada voluntariamente, mas não pode ser requisito para reproduzir a inferência ou avaliar os resultados entregues.

A inferência sobre cada manifesto deve terminar em até 10 minutos no ambiente mínimo descrito. Não há limite de treinamento, mas o relatório deve informar hardware, tempo de treinamento e uso de GPU, quando houver. Serviços externos de inferência não são permitidos.

A submissão deverá conter um arquivo .zip com código, README, requirements.txt ou equivalente, relatório em PDF e os dois CSVs de predição. O pacote não deve incluir imagens do dataset nem pesos ou artefatos acima de 500 MiB; pesos maiores devem ter instruções de obtenção ou reprodução, quando autorizados.

### 11.5 Rubrica detalhada
| Critério                              | Pontos | Evidência esperada |
|---------------------------------------|:------:|--------------------|
| Pipeline funcional e reprodutibilidade | 2,5 | Código executa, README é suficiente e CSVs seguem o contrato. |
| Processamento de imagens e justificativa | 2,5 | Transformação própria, motivação e comparação justa com baseline. |
| Protocolo e resultados quantitativos | 2,0 | F1-macro, métricas por classe, matriz de confusão e condições controladas. |
| Robustez, erros e limitações | 2,0 | Comparação validação–robustez, exemplos visuais e interpretação honesta. |
| Relatório e defesa individual | 1,0 | Relatório claro e capacidade de explicar decisões e resultados. |

O relatório terá no máximo seis páginas, excluindo referências e apêndice técnico. A defesa ocorrerá em grupo, mas cada integrante responderá individualmente a pelo menos uma pergunta. A ausência de domínio mínimo sobre a própria contribuição poderá reduzir a parcela individual da nota.

Dados externos, rótulos adicionais e modelos treinados especificamente no conjunto de robustez não são permitidos.

Em caso de dúvida sobre a especificação, prevalece este enunciado e os arquivos distribuídos no pacote oficial.
