# Amostra para revisão

Victor: 15 perguntas sorteadas (seed 15) do conjunto de avaliação. Para cada uma, confira se a pergunta
faz sentido, se o artigo esperado é mesmo o que responde e se a resposta de referência está certa.
Marque OK ou escreva a correção na última coluna. As duas últimas linhas não têm resposta no corpus de propósito:
o sistema precisa responder "NÃO ENCONTREI".

Como foi feito: 70 artigos sorteados (seed 7, estratificado por ato, em `scripts/sample_articles.py`);
para cada um, uma pergunta escrita à mão com outras palavras, como alguém perguntaria. Mais 12 perguntas
de fora do corpus (Selic, TED, IR, LGPD...).

| id | pergunta | artigo esperado | resposta de referência | revisão |
| --- | --- | --- | --- | --- |
| q002 | O que o banco precisa avisar a quem tem chave Pix de celular ou e-mail sobre a verificação de chaves registradas? | `res-bcb-1#regulamento#art-115` | Que outros usuários podem saber que a chave existe por causa da funcionalidade de verificação de chaves, com tempo para pedir a exclusão. | |
| q003 | Por qual rede os participantes se conectam ao DICT? | `res-bcb-1#regulamento#art-47` | Pela RSFN, contratando operadoras de telecomunicação ou PSTI autorizados. | |
| q004 | Um participante do Pix pode contratar outro participante para prover serviços específicos? Quem responde perante o cliente? | `res-bcb-1#regulamento#art-90-D` | Sim, por parceria; o contratante é integralmente responsável pelo cumprimento do Regulamento. | |
| q005 | O que valia entre 3 e 15 de novembro de 2020 no lançamento do Pix? | `res-bcb-1#regulamento#art-105` | Participação obrigatória para quem foi aprovado nas etapas cadastral e homologatória e todas as funcionalidades do DICT disponíveis. | |
| q010 | Como o DICT evita que alguém use as consultas de chave para coletar dados em massa? | `res-bcb-1#regulamento#art-83` | Mantém mecanismos de prevenção a ataques de leitura definidos no Manual Operacional do DICT. | |
| q021 | Por quanto tempo a marcação de fundada suspeita de fraude fica vinculada ao CPF ou CNPJ no DICT? | `res-bcb-1#regulamento#art-78-HD` | Cinco anos a partir do registro da marcação. | |
| q027 | O agente de saque paga tarifa para receber um Pix Saque? | `res-bcb-1#regulamento#art-87-F` | Não, é vedada a cobrança de tarifa dos agentes de saque nesse recebimento. | |
| q031 | Cooperativas que operam com contas de depósito de finalidade específica são equiparadas a singulares filiadas a uma central na adesão ao Pix? | `in-bcb-511#ato#art-102` | Sim, quando têm como liquidante no SPI a instituição que operacionaliza essas contas. | |
| q044 | Até quando posso cancelar um Pix Agendado? | `in-bcb-513#ato#art-14` | Até 23h59 do dia anterior à data de liquidação. | |
| q052 | Qual é o horário do período noturno para os limites do Pix de pessoa física? | `in-bcb-512#ato#art-3` | Em geral das 20h às 6h; o diurno das 6h às 20h. | |
| q055 | Órgãos públicos podem acessar diretamente os dados das chaves Pix? | `res-bcb-338#anexo-i#art-1, res-bcb-338#ato#art-5` | Sim, entes públicos no exercício de suas atribuições legais, conforme o regulamento de acesso ao DICT. | |
| q057 | Banco pode cobrar tarifa de pessoa física para enviar Pix? | `res-bcb-19#ato#art-4, res-bcb-1#regulamento#art-87-C` | Em regra não; da pessoa natural só no recebimento com finalidade de compra e no envio de saque/troco nas condições previstas. | |
| q067 | Quais participantes do Pix precisam enviar informações periódicas ao Banco Central pela IN 32? | `in-bcb-32#ato#art-1` | Provedores de conta transacional e liquidantes especiais. | |
| q076 | Qual é a tarifa máxima de um boleto bancário? | `— (fora do corpus)` | NÃO ENCONTREI | |
| q082 | Qual é o prazo do Procon para responder uma reclamação? | `— (fora do corpus)` | NÃO ENCONTREI | |
