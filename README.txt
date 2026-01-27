Legends of Heropolis DX - Layout Planner (GUI)

O que é
- Um app em Python (Tkinter) pra você montar e otimizar o layout do mapa
- Foco 100% em maximizar o TOTAL de buffs (%) dos “Spots” (adjacência)

Regras que o app respeita
1) Adjacência conta nos 8 ao redor (inclui diagonais)
2) Cada par de construções adjacentes conta NO MÁXIMO 1 vez por Spot (mesmo se encostar em vários tiles)
3) Sem cap: os bônus somam normalmente
4) Chunk 4x4:
   - Qualquer construção maior que 1x1 precisa caber inteira dentro de UM único chunk 4x4
   - Ou seja, 2x2 e 4x4 não podem atravessar a linha do chunk

Como usar (Windows)
1) Instale Python 3 (se não tiver)
2) Extraia o ZIP
3) Dê dois cliques em: run_gui.bat

Dentro do app
- O mapa é em tiles (quadradinhos)
- Linhas grossas mostram os chunks 4x4
- Clique para colocar construções (primeiro “Selecionar” no inventário)
- Arraste para mover
- Block Mode (ou botão direito) marca tiles vermelhos onde NÃO pode construir
- Cada construção no inventário tem:
  • Count = quantidade que você tem
  • Tamanho = 1x1 / 2x2 / 4x4 / 4x1 / 1x4 (se não souber, deixe “unset” por enquanto)

Optimizer (Meta)
- Clique em “Optimize (Meta)”
- Ele monta um layout do zero usando TODAS as construções com count>0 e tamanho definido
- Se algum item estiver com tamanho “unset”, ele vai avisar e não vai usar aquele item

Arquivos
- inventory.json: seu inventário (categorias, counts e tamanhos)
- Você pode salvar/carregar inventário e salvar/carregar layout pelo app

Obs.
- Se o .bat abrir e fechar rápido, normalmente é falta de Python instalado ou PATH.
  Se isso acontecer, rode pelo CMD:
    python heropolis_gui.py


Dica (Windows):
- Se voce nao quiser ver o CMD piscando, use: start_gui.vbs
- Se aparecer erro dizendo que Python nao foi encontrado, instale Python 3 e marque "Add to PATH".
