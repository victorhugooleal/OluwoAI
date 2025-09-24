# 🧙‍♂️ Oluwo AI

> **IA para interpretação dos 256 Odus de Ifá, preservando a sabedoria Yorubá com tecnologia moderna.**

---

## ✨ Visão Geral
- 🔍 Busca semântica nos 256 Odus
- 📚 Respostas contextualizadas com referências específicas
- 🌍 Interpretação multilíngue (Yorubá/Português)
- 🧠 Rastreabilidade do raciocínio
- 🛡️ Preservação da autenticidade cultural

---

## 🏗️ Arquitetura
- **Linguagem:** Python 3.11+
- **Framework Web:** FastAPI
- **LLM:** GPT-4/Claude 3 Opus (com backup Llama 3.1 70B)
- **Banco Vetorial:** Qdrant ou Weaviate
- **Processamento PDF:** Docling + PyMuPDF
- **Embeddings:** text-embedding-3-large (OpenAI) ou multilingual-e5-large
- **Cache:** Redis
- **Monitoramento:** Langfuse
- **Deploy:** Docker + Kubernetes

---

## 🔗 Pipeline RAG
1. Conversão dos PDFs para JSON estruturado
2. Indexação vetorial dos Odus
3. Busca semântica e reranking
4. Geração de respostas com citações e chain of thought

---

## 🚀 API REST
- `/api/v1/consult` — Consulta aos Odus
- `/api/v1/odu/{odu_id}` — Detalhes de cada Odu
- `/api/v1/index/refresh` — Reindexação dos dados

---

## 🧪 Testes e Monitoramento
- Suite de testes automatizados (pytest)
- Métricas via Prometheus
- Logs estruturados

---

## 🖥️ Como rodar localmente
```bash
# Instale as dependências
pip install -r requirements.txt

# Execute o servidor
uvicorn main:app --reload

# Para ambiente completo, use Docker Compose
```

---

## 📦 Observações Importantes
- **Os arquivos PDF dos Odus NÃO devem ser versionados no GitHub.**
- Consulte a pasta `256-odus-pdf/` localmente para processamento.

---

## 💡 Versão POC Barata
Se quiser rodar sem custos, siga as instruções do arquivo:
- `.github/instructions/projeto-oluwoai-poc-barata.instructions.md`

Stack sugerida para POC:
- Python, FastAPI, Qdrant local, embeddings MiniLM, LLM local via Ollama/LM Studio

---

## 📚 Licença
Projeto privado para preservação cultural e pesquisa. Consulte o autor para uso comercial.
