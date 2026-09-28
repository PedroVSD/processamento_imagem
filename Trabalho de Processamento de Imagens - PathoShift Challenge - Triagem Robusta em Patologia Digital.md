**Grupo:** grupos de três alunos  
**Entregas:** três etapas cumulativas - datas a definir!

### 1. Visão geral

Neste trabalho, cada grupo projetará, implementará e avaliará um **pipeline de processamento e análise de imagens histopatológicas**. O sistema receberá imagens microscópicas — inicialmente _tiles_ (recortes) e, se a infraestrutura permitir, coleções organizadas por lâmina — e deverá produzir classificações, medidas de confiança e evidências visuais que permitam inspecionar a decisão.

Patologia digital é uma aplicação importante de processamento de imagens e visão computacional: a variabilidade de coloração, aquisição, qualidade e procedência das imagens torna insuficiente demonstrar desempenho apenas em uma partição aleatória de um único conjunto de dados. O objetivo é investigar, de modo reproduzível, como escolhas de pré-processamento, representação, classificação, agregação e validação afetam a **robustez** do sistema.

O trabalho não tem finalidade de diagnóstico clínico nem autoriza uso clínico. As predições devem ser tratadas estritamente como resultados experimentais em dados públicos e anonimizados.

### 2. Glossário inicial

|Termo|Descrição breve|
|---|---|
|**Baseline**|Implementação simples e reproduzível usada como ponto de comparação.|
|**Calibração**|Correspondência entre a confiança declarada pelo modelo e a frequência com que ele está correto.|
|**Domínio**|Contexto de aquisição dos dados, como laboratório, scanner ou protocolo de coloração.|
|**H&E**|Coloração por hematoxilina e eosina, comum em imagens histopatológicas.|
|**Lâmina digital / WSI**|Imagem digital de uma lâmina inteira (_Whole Slide Image_), potencialmente com gigapixels.|
|**Mudança de domínio**|Alteração na distribuição das imagens entre treino e uso, causada por origem, coloração ou aquisição.|
|**Patch / tile**|Recorte menor extraído de uma imagem ou lâmina, processável individualmente.|
|**Split por paciente/lâmina**|Separação que impede imagens do mesmo paciente ou lâmina de aparecerem em treino e teste.|
|**Validação externa**|Avaliação em dados coletados de fonte diferente daquela usada para treinamento.|

### 3. Objetivos de aprendizagem

- Aplicar operações de processamento de imagens a um problema real, justificando as escolhas feitas.
- Analisar cor, iluminação, textura, ruído, resolução, foco e artefatos de aquisição.
- Construir e comparar baselines de [classificação de imagens](https://www.dcc.ufrrj.br/moodle/mod/resource/view.php?id=17638 "Classificação de Imagens").
- Organizar experimentos com separação apropriada entre desenvolvimento e avaliação.
- Medir qualidade preditiva, incerteza, custo computacional e robustez.
- Comunicar limitações, vieses e riscos de uma aplicação de visão computacional em saúde.

### 4. Entregas

|Entrega|Tema|Peso sugerido|
|---|---|---|
|1|Pathology Image Pipeline|25%|
|2|Slide-aware Analysis|30%|
|3|Robust Pathology System|45%|

As entregas são cumulativas. Datas, meio de submissão, ambiente oficial de execução e rubrica detalhada serão divulgados separadamente.

### 5. Dados e protocolo experimental

A disciplina fornecerá ou indicará datasets públicos, anonimizados e adequados ao escopo. A definição final dependerá de teste local de viabilidade, licença, tamanho, disponibilidade e infraestrutura. Poderão ser usados conjuntos de _patches_, um subconjunto pré-extraído de lâminas digitais, ou ambos.

- As partições oficiais devem evitar vazamento entre treino, validação e teste; quando houver identificadores, a separação será por paciente ou lâmina.
- Dados de teste privados não poderão ser usados para escolha de hiperparâmetros.
- O conjunto poderá conter classes desbalanceadas; acurácia isolada não será métrica suficiente.
- Identificadores, metadados e imagens de teste não devem ser usados para inferir rótulos por atalhos.
- O dataset distribuído pela disciplina prevalece sobre versões encontradas em repositórios de terceiros.

### 6. Interface de execução e submissão

O formato final de submissão será publicado antes da primeira entrega. A disciplina poderá adotar Docker Linux `amd64`, execução local assistida ou mecanismo equivalente que preserve reprodutibilidade e avaliação justa.

Como referência, caso Docker seja adotado, a interface poderá seguir o formato:

```
/pathoshift prepare --input /input --output /output
/pathoshift predict --input /input/manifest.jsonl --output /output/predictions.jsonl
/pathoshift evaluate --input /input/manifest.jsonl --output /output/metrics.json
/pathoshift describe
```

O sistema deverá operar sem rede durante a avaliação oficial e não deve depender de serviços externos de inferência. Os comandos, nomes finais, JSON Schemas e artefatos obrigatórios poderão ser ajustados na publicação técnica.

### 7. Entrega 1 — Pathology Image Pipeline

Nesta etapa, o grupo deverá construir um pipeline funcional para imagens individuais.

- Leitura validada do conjunto e geração de manifesto reprodutível.
- Análise exploratória breve: classes, dimensões, cor e exemplos de qualidade/artefatos.
- Pelo menos um procedimento de pré-processamento justificado.
- Pelo menos um baseline de classificação treinado e avaliado nas partições fornecidas.
- Métricas por classe e macro, matriz de confusão e exemplos de erros.
- Registro de configuração, _seed_, versões e custo de execução.

São exemplos de componentes possíveis: rejeição de fundo, avaliação de foco, transformação RGB–HSV/LAB, normalização de intensidade/cor, filtragem, extração de textura, classificador clássico ou rede neural. A escolha é livre, mas não substitui a justificativa experimental.

### 8. Entrega 2 — Slide-aware Analysis

Além dos requisitos anteriores, o grupo deverá evoluir a análise para considerar a estrutura de uma coleção de imagens associadas a uma mesma lâmina, caso tais dados sejam fornecidos. Quando o dataset oficial não incluir essa estrutura, a disciplina definirá uma tarefa equivalente de análise espacial, multirresolução ou agregação de amostras.

- Estratégia explícita de seleção, ordenação ou agregação de tiles.
- Predição no nível definido pela tarefa: tile, região ou lâmina.
- Visualização que conecte a predição às regiões de evidência.
- Comparação com um baseline que não use a estratégia proposta.
- Análise de erros de decisão local e agregação global.

São exemplos possíveis: votação e estatísticas robustas, seleção dos tiles mais informativos, agregação por atenção, múltiplas instâncias, pirâmide de resoluções ou modelos sensíveis a coordenadas.

### 9. Entrega 3 — Robust Pathology System

A entrega final investigará uma questão de pesquisa formulada pelo grupo e aprovada no escopo divulgado pela disciplina. A hipótese deve ser objetiva, mensurável e relacionada à confiabilidade do pipeline.

O grupo deverá escolher ao menos um eixo:

- mudança de domínio entre laboratórios, scanners ou protocolos de coloração;
- normalização de cor versus aumentação de dados;
- baixa disponibilidade de rótulos ou treinamento com subconjunto limitado;
- qualidade de imagem: desfoque, compressão, ruído ou presença de fundo;
- calibração de confiança, detecção de incerteza ou abstenção;
- explicabilidade e verificação da fidelidade de mapas de evidência;
- eficiência de memória, armazenamento ou inferência para grande número de tiles.

Serão exigidos: hipótese declarada, comparação controlada entre baseline e proposta, ablação, análise de erros e limitações, relatório final e defesa técnica individual.

### 10. Avaliação

A avaliação combinará correção do artefato, validade experimental e qualidade da análise. A distribuição exata de pontos será divulgada antes de cada entrega. Como referência, a avaliação final poderá considerar:

- correção, execução e reprodutibilidade;
- qualidade preditiva em métricas adequadas, incluindo métricas por classe e macro;
- robustez em conjuntos ou transformações não divulgados previamente;
- qualidade das evidências visuais e coerência de sua interpretação;
- custo computacional, dentro dos limites publicados;
- arquitetura, documentação, análise de erros e limitações;
- defesa técnica individual.

Melhorias grandes em uma única métrica não compensam falhas de reprodutibilidade, vazamento de dados, análise inadequada ou impossibilidade de explicar o sistema.

### 11. Uso de IA generativa e modelos pré-treinados

Ferramentas de IA generativa podem ser utilizadas para programação, depuração, estudo, documentação e análise. Modelos visuais pré-treinados poderão ser permitidos conforme as regras técnicas de cada entrega.

A responsabilidade pela correção, pelos resultados e pela compreensão permanece integralmente com o grupo. Durante a defesa, o grupo deve ser capaz de explicar decisões de pré-processamento, arquitetura, treino, métricas, código e limitações.

O relatório final deverá declarar ferramentas de IA generativa utilizadas, modelos externos/pesos pré-treinados e fontes de dados adicionais, participação substancial dessas ferramentas e como o grupo verificou os resultados. Não será permitido usar, durante a avaliação oficial, chamadas a serviços externos, modelos hospedados remotamente ou mecanismos que transfiram dados de teste para fora do ambiente.

### 12. Ética e comunicação de resultados

O trabalho não produz ferramenta médica, laudo ou recomendação clínica. Não devem ser feitas alegações de segurança, eficácia clínica ou substituição de profissionais de saúde. Os grupos devem explicitar possíveis vieses, limites de generalização e a diferença entre desempenho experimental e validação clínica. Dados não públicos, identificáveis ou obtidos sem autorização não podem ser usados.

### 13. Próximas publicações

- calendário, regras de submissão e rubricas detalhadas;
- dataset oficial, licença, _splits_ e procedimento de acesso;
- especificação de manifestos, saídas e JSON Schemas;
- ferramenta de avaliação, baseline e exemplos públicos;
- política definitiva para modelos pré-treinados e dados externos;
- procedimento de defesa e critérios de avaliação individual.

**Em caso de dúvida sobre a especificação, prevalecerão os documentos técnicos e comunicados publicados pela disciplina.**